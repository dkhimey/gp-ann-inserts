#include <mpi.h>
#include <fstream>
#include <numeric>
#include <unordered_set>

#include "distributed_query_benchmark.h"
#include "points_io.h"
#include "metis_io.h"

// Number of timed passes per nprobe value (results averaged, matching surge).
static constexpr int NUM_RUNS = 3;

namespace {
    size_t ComputeChunkSize(size_t a, size_t b) {
        return (a + b - 1) / b;
    }
}

int main(int argc, const char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: ./DistributedBench input-points queries ground-truth-file"
                     " num_neighbors clusters-file output-file\n"
                     "  clusters-file: GP partition output (<prefix>.k=P.GP)\n"
                     "  Pass - as ground-truth-file to skip recall computation." << std::endl;
        std::abort();
    }

    std::string point_file        = argv[1];
    std::string query_file        = argv[2];
    std::string ground_truth_file = argv[3];
    std::string k_string          = argv[4];
    std::string clusters_file     = argv[5];
    std::string output_file       = argv[6];

    MPI_Init(nullptr, nullptr);

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // ── Build index once; reuse across all nprobe values ─────────────────────────
    std::cout << "[DistributedBench] Building index on all ranks …\n";
    DistributedQueryBenchmark bench;
    bench.num_neighbors = std::stoi(k_string);
    bench.LoadPartitionFromClusters(clusters_file);
    bench.LoadShardPointSet(point_file);
    bench.BuildRouterFromSample(point_file);
    bench.BuildInShardIndex();
    std::cout << "[DistributedBench] Index built.\n";

    // ── Load queries (all ranks) ──────────────────────────────────────────────────
    std::cout << "[DistributedBench] Loading queries on all ranks …\n";
    PointSet queries = ReadPoints(query_file);
    std::vector<int> query_ids(queries.n);
    std::iota(query_ids.begin(), query_ids.end(), 0);
    size_t chunk_size = ComputeChunkSize(query_ids.size(), comm_size);
    std::vector<int> my_query_ids(
        query_ids.begin() + rank * chunk_size,
        query_ids.begin() + std::min(query_ids.size(), (rank + 1) * chunk_size));
    std::cout << "[DistributedBench] Rank " << rank << " has " << my_query_ids.size() << " queries.\n";

    // ── Ground truth (all ranks load for distributed recall computation) ─────────
    std::vector<NNVec> ground_truth;
    bool have_gt = false;
    if (!ground_truth_file.empty() && ground_truth_file != "-") {
        ground_truth = ReadGroundTruth(ground_truth_file);
        have_gt = !ground_truth.empty();
    }

    // ── Open output CSV (rank 0 only) ─────────────────────────────────────────────
    std::ofstream csv;
    if (rank == 0) {
        csv.open(output_file);
        if (!csv.is_open()) {
            std::cerr << "ERROR: cannot open output file: " << output_file << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        csv << "nprobe,recall@" << bench.num_neighbors << ",qps,avg_parts_searched\n";
        std::cout << "[DistributedBench] " << bench.num_shards << " shards"
                  << "  k=" << bench.num_neighbors
                  << "  queries=" << queries.n
                  << (have_gt ? "  (ground truth loaded)" : "  (no ground truth)")
                  << "\n"
                  << "  sweeping nprobe 1.." << bench.num_shards << "\n";
    }

    // ══════════════════════════════════════════════════════════════════════════════
    //  Sweep: for each nprobe in [1, num_shards]:
    //    1. Warmup pass  – collects results, computes recall (untimed)
    //    2. NUM_RUNS timed passes – barrier-to-barrier, MPI_Reduce(MAX) for QPS
    // ══════════════════════════════════════════════════════════════════════════════
    for (int nprobe = 1; nprobe <= bench.num_shards; nprobe++) {
        if (rank == 0)
            std::cout << "  nprobe=" << nprobe << " ...\n";

        // ── Warmup pass: collect results, measure recall ──────────────────────────
        long long warmup_parts = 0;
        auto results = bench.ProcessQueries(
            my_query_ids, queries, nprobe, warmup_parts, /*collect_results=*/true);

        uint64_t my_hits = 0;
        if (have_gt) {
            for (size_t i = 0; i < my_query_ids.size(); i++) {
                int qid = my_query_ids[i];
                if (qid >= static_cast<int>(ground_truth.size())) continue;
                const auto& gt_q = ground_truth[qid];
                std::unordered_set<uint32_t> gt_set;
                for (int j = 0; j < bench.num_neighbors && j < static_cast<int>(gt_q.size()); j++)
                    gt_set.insert(gt_q[j].second);
                for (uint32_t nb : results[i])
                    if (gt_set.count(nb)) my_hits++;
            }
        }

        uint64_t total_hits = 0;
        MPI_Reduce(&my_hits, &total_hits, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        long long global_warmup_parts = 0;
        MPI_Reduce(&warmup_parts, &global_warmup_parts, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        const double recall = (have_gt && queries.n > 0)
            ? static_cast<double>(total_hits) / (static_cast<double>(queries.n) * bench.num_neighbors)
            : -1.0;
        const double avg_parts_warmup = (queries.n > 0)
            ? static_cast<double>(global_warmup_parts) / static_cast<double>(queries.n)
            : -1.0;

        if (rank == 0)
            std::cout << "    [warmup] recall@" << bench.num_neighbors << "=" << recall
                      << "  avg_parts=" << avg_parts_warmup << "\n";

        // ── Timed passes: NUM_RUNS full sweeps, barrier-to-barrier ────────────────
        long long timed_parts_acc = 0;

        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();

        for (int run = 0; run < NUM_RUNS; run++) {
            long long pass_parts = 0;
            bench.ProcessQueries(
                my_query_ids, queries, nprobe, pass_parts, /*collect_results=*/false);
            timed_parts_acc += pass_parts;
        }

        const double t1 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        double elapsed = t1 - t0;
        double max_elapsed = 0.0;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        long long global_timed_parts = 0;
        MPI_Reduce(&timed_parts_acc, &global_timed_parts, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            const double qps = (max_elapsed > 0.0)
                ? static_cast<double>(queries.n) * NUM_RUNS / max_elapsed
                : 0.0;
            const double avg_parts_timed = (queries.n > 0 && NUM_RUNS > 0)
                ? static_cast<double>(global_timed_parts) / (static_cast<double>(queries.n) * NUM_RUNS)
                : -1.0;

            std::cout << "    [timed]  qps=" << qps
                      << "  avg_parts=" << avg_parts_timed << "\n";

            csv << nprobe << "," << recall << "," << qps << "," << avg_parts_timed << "\n";
            csv.flush();
        }
    }

    if (rank == 0) {
        csv.close();
        std::cout << "[DistributedBench] Done. Results written to " << output_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
