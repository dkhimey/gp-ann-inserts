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
//   nprobe, time_s, qps, recall@<K>, theoretical_recall@<K>
//
// theoretical_recall@<K> is the point-level routing recall: the fraction of
// true k-NN whose owning shard is among the top-nprobe probed shards, i.e. the
// recall an oracle in-shard search would achieve. It upper-bounds recall@<K>.

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

    // Suppress library-level ReadPoints / HNSW chatter on non-root ranks.
    if (rank != 0) std::cout.setstate(std::ios_base::failbit);

    // ------------------------------------------------------------------
    // Build the per-shard index and load the router.
    // Per-rank progress goes to stderr so it is visible from all ranks
    // regardless of the stdout suppression above.
    // ------------------------------------------------------------------
    DistributedQueryBenchmark bench;
    bench.num_neighbors = num_neighbors;
    // Set num_voting_neighbors high enough that every shard gets a real
    // distance estimate for all nprobe values across the sweep.
    // Use 10 * comm_size voting neighbors so every shard gets a real distance
    // estimate for all nprobe values in the sweep (matches reference benchmark).
    bench.num_voting_neighbors = 10 * comm_size;

    double t_setup = MPI_Wtime();
    std::cerr << "[rank " << rank << "] loading partition\n";
    // The partition (point -> shard map) is always needed: it drives the shard
    // point selection on a build, and the point-level theoretical-recall
    // computation during the sweep.
    bench.LoadPartition(partition_file);

    // Per-shard index: load from the on-disk cache if one exists for this
    // (partition, M, ef_construction); otherwise build it and save it so the
    // next run can skip the read+build.  On a cache hit we only need `dim`
    // (cheap header read) — the full shard point set is never loaded.
    const std::string shard_cache = bench.ShardIndexCachePath(partition_file);
    if (std::filesystem::exists(shard_cache)) {
        std::cerr << "[rank " << rank << "] loading cached in-shard HNSW\n";
        bench.ReadDim(point_file);
        bench.LoadInShardIndex(shard_cache);
    } else {
        std::cerr << "[rank " << rank << "] loading shard points\n";
        bench.LoadShardPointSet(point_file);
        std::cerr << "[rank " << rank << "] building in-shard HNSW\n";
        bench.BuildInShardIndex(shard_cache);
    }

    std::cerr << "[rank " << rank << "] loading router\n";
    bench.LoadRouter(router_file);
    std::cerr << "[rank " << rank << "] setup done in "
              << (MPI_Wtime() - t_setup) << " s\n";

    PointSet queries = ReadPoints(query_file);

    // Synchronise: rank 0 must not enter the sweep until every rank has
    // finished building its shard index.
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout.clear();
        std::cout << "DistributedBench\n"
                  << "  ranks         : " << comm_size << "\n"
                  << "  queries       : " << queries.n << "\n"
                  << "  num_neighbors : " << num_neighbors << "\n"
                  << "  warmup rounds : " << bench.num_warmup_rounds << "\n"
                  << "  bench rounds  : " << bench.num_bench_rounds << "\n"
                  << "  gt file       : " << ground_truth_file << "\n"
                  << "  output        : " << output_file << "\n"
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
        csv << "nprobe,time_s,qps,recall@" << num_neighbors
            << ",theoretical_recall@" << num_neighbors << "\n";
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
            const double qps = r.min_elapsed > 0
                               ? (double)queries.n / r.min_elapsed : 0.0;
            std::cout << "nprobe=" << r.nprobe
                      << "  time=" << r.min_elapsed << " s"
                      << "  QPS=" << qps;
            if (has_gt)
                std::cout << "  recall@" << num_neighbors << "=" << r.recall
                          << "  theoretical_recall@" << num_neighbors << "=" << r.theoretical_recall;
            std::cout << "\n";

            csv << r.nprobe << "," << r.min_elapsed << "," << qps << ",";
            if (has_gt) csv << r.recall << "," << r.theoretical_recall;
            else        csv << "N/A,N/A";
            csv << "\n";
        }
        csv.flush();
        csv.close();
        std::cout << "\nDone. Results written to " << output_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
