#include <mpi.h>
#include "distributed_query_benchmark.h"

#include "points_io.h"
#include "metis_io.h"

namespace {
    size_t ComputeChunkSize(size_t a, size_t b) {
        return (a+b-1) / b;
    }
}

int main(int argc, const char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage ./DistributedBench input-points queries ground-truth-file num_neighbors partition-file router-file" << std::endl;
        std::abort();
    }

    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
    std::string k_string = argv[4];
    int num_neighbors = std::stoi(k_string);
    std::string partition_file = argv[5];
    std::string router_file = argv[6];

    MPI_Init(nullptr, nullptr);

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    DistributedQueryBenchmark bench;
    bench.num_neighbors = num_neighbors;
    bench.LoadPartition(partition_file);
    bench.LoadShardPointSet(point_file);
    bench.BuildInShardIndex();
    bench.LoadRouter(router_file);

    PointSet queries = ReadPoints(query_file);
    std::vector<int> query_ids(queries.n);
    std::iota(query_ids.begin(), query_ids.end(), 0);
    size_t chunk_size = ComputeChunkSize(query_ids.size(), comm_size);
    std::vector<int> my_query_ids(query_ids.begin() + rank * chunk_size, query_ids.begin() + std::min(query_ids.size(), (rank + 1) * chunk_size));

    MPI_Barrier(MPI_COMM_WORLD);

    double t1, t2;
    t1 = MPI_Wtime();
    auto [rcv_rqids, rcv_rids, rcv_rdists] = bench.ProcessQueries(my_query_ids, queries);
    t2 = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);    // TODO is this necessary? the subsequent MPI_Reduce should enforce a sync
    double elapsed = t2 - t1;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Merge results per query (handles num_probes > 1).
    std::vector<NNVec> neighbors(nq);
    for (int src = 0; src < comm_size; ++src) {
        const size_t nres = rcv_rqids[src].size();
        for (size_t j = 0; j < nres; ++j) {
            const uint32_t qid = rcv_rqids[src][j];
            for (int nn = 0; nn < num_neighbors; ++nn)
                neighbors[qid].push_back({
                    rcv_rdists[src][j * num_neighbors + nn],
                    rcv_rids[src][j * num_neighbors + nn] });
        }
    }
    // Trim to top-k (sorted, deduped by label).
    for (size_t q = qs; q < qe; ++q) {
        auto& nv = neighbors[q];
        std::sort(nv.begin(), nv.end());
        nv.erase(std::unique(nv.begin(), nv.end(),
            [](const auto& a, const auto& b){ return a.second == b.second; }),
            nv.end());
        if ((int)nv.size() > num_neighbors) nv.resize(num_neighbors);
    }

    // Recall: each rank evaluates its own query chunk; reduce sum.
    double recall = 0.0;
    if (!gt_file.empty() && std::filesystem::exists(gt_file)) {
        auto gt = ReadGroundTruth(gt_file);
        // Use distances stored in the GT file directly (no base vectors needed).
        std::vector<float> dist_kth(nq, std::numeric_limits<float>::max());
        for (size_t q = 0; q < nq; ++q) {
            auto sorted = gt[q];
            std::sort(sorted.begin(), sorted.end());
            if ((int)sorted.size() >= num_neighbors)
                dist_kth[q] = sorted[num_neighbors - 1].first;
        }
        uint64_t my_hits = 0;
        for (size_t q = qs; q < qe; ++q)
            for (const auto& [dist, id] : neighbors[q])
                if (dist <= dist_kth[q]) ++my_hits;
        uint64_t total_hits = 0;
        MPI_Reduce(&my_hits, &total_hits, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
            recall = (double)total_hits / ((double)nq * num_neighbors);
    }

    // compute recall on rank 0

    if (rank == 0) {
        std::cout << "End-to-end time " << max_elapsed << std::endl;
        std::cout << "Throughput " << (double)queries.n / max_elapsed << " qps" << std::endl;
        if (!gt_file.empty() && std::filesystem::exists(gt_file))
            std::cout << "Recall@" << num_neighbors << " = " << recall << std::endl;
    }

    MPI_Finalize();
    return 0;
}