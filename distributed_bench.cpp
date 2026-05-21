// distributed_bench.cpp
//
// Pure-query throughput and recall benchmark for a static GP-ANN index.
// One MPI rank per shard.  No inserts or deletes.
//
// Each rank loads its shard from the partition file, builds a local HNSW,
// and loads the pre-built HNSW router.  Then a sweep over nprobe = 1 .. K
// is executed: for each value the full Alltoallv fan-out/fan-in pipeline
// is timed and recall@num_neighbors is computed against the ground-truth
// file.
//
// Usage
// -----
//   mpirun -np <K> ./DistributedBench
//          <input-points> <queries> <ground-truth-file>
//          <num-neighbors> <partition-file> <router-file>
//          [output-file]
//
//   K must equal the number of shards in the partition file.
//   output-file defaults to "distributed_bench_results.csv".
//
// CSV columns
// -----------
//   nprobe, time_s, qps, recall@<K>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

#include <mpi.h>

#include "distributed_query_benchmark.h"
#include "points_io.h"
#include "metis_io.h"

int main(int argc, const char* argv[]) {
    if (argc < 7 || argc > 8) {
        std::cerr << "Usage: mpirun -np <K> ./DistributedBench"
                     " <input-points> <queries> <ground-truth-file>"
                     " <num-neighbors> <partition-file> <router-file>"
                     " [output-file]\n";
        std::abort();
    }

    const std::string point_file      = argv[1];
    const std::string query_file      = argv[2];
    const std::string ground_truth_file = argv[3];
    const int         num_neighbors   = std::stoi(argv[4]);
    const std::string partition_file  = argv[5];
    const std::string router_file     = argv[6];
    const std::string output_file     = (argc == 8)
                                        ? argv[7]
                                        : "distributed_bench_results.csv";

    MPI_Init(nullptr, nullptr);

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Suppress ReadPoints / build chatter on non-root ranks.
    if (rank != 0) std::cout.setstate(std::ios_base::failbit);

    // ------------------------------------------------------------------
    // Build the per-shard index and load the router.
    // ------------------------------------------------------------------
    DistributedQueryBenchmark bench;
    bench.num_neighbors = num_neighbors;
    // Set num_voting_neighbors high enough that every shard gets a real
    // distance estimate for all nprobe values across the sweep.
    bench.num_voting_neighbors = comm_size;

    bench.LoadPartition(partition_file);
    bench.LoadShardPointSet(point_file);
    bench.BuildInShardIndex();
    bench.LoadRouter(router_file);

    PointSet queries = ReadPoints(query_file);

    if (rank == 0) {
        std::cout.clear();
        std::cout << "DistributedBench\n"
                  << "  ranks        : " << comm_size << "\n"
                  << "  queries      : " << queries.n << "\n"
                  << "  num_neighbors: " << num_neighbors << "\n"
                  << "  gt file      : " << ground_truth_file << "\n"
                  << "  output       : " << output_file << "\n"
                  << "  sweeping nprobe 1.." << comm_size << "\n\n";
    }

    // ------------------------------------------------------------------
    // Open CSV output on rank 0.
    // ------------------------------------------------------------------
    std::ofstream csv;
    if (rank == 0) {
        csv.open(output_file);
        if (!csv) {
            std::cerr << "Cannot open output file: " << output_file << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        csv << "nprobe,time_s,qps,recall@" << num_neighbors << "\n";
    }

    // ------------------------------------------------------------------
    // Run the sweep.
    // ------------------------------------------------------------------
    if (rank != 0) std::cout.setstate(std::ios_base::failbit);
    auto sweep = bench.ProcessSearchSweep(queries, ground_truth_file);
    if (rank == 0) std::cout.clear();

    // ------------------------------------------------------------------
    // Report and write CSV (rank 0 only).
    // ------------------------------------------------------------------
    if (rank == 0) {
        const bool has_gt = !ground_truth_file.empty() &&
                            std::filesystem::exists(ground_truth_file);
        for (const auto& r : sweep) {
            const double qps = r.elapsed > 0
                               ? (double)queries.n / r.elapsed : 0.0;
            std::cout << "nprobe=" << r.nprobe
                      << "  time=" << r.elapsed << " s"
                      << "  QPS=" << qps;
            if (has_gt)
                std::cout << "  recall@" << num_neighbors << "=" << r.recall;
            std::cout << "\n";

            csv << r.nprobe << "," << r.elapsed << "," << qps << ",";
            if (has_gt) csv << r.recall; else csv << "N/A";
            csv << "\n";
        }
        csv.flush();
        csv.close();
        std::cout << "\nDone. Results written to " << output_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
