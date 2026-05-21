#pragma once

#include "points_io.h"
#include "metis_io.h"
#include "hnsw_router.h"
#include "../external/hnswlib/hnswlib/hnswlib.h"

#include <mpi.h>
#include <parlay/primitives.h>

#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Generic Alltoallv helper
// ---------------------------------------------------------------------------

namespace dqb_detail {

template<typename T>
inline void AllToAllV(const std::vector<std::vector<T>>& send_bufs,
                      std::vector<std::vector<T>>&       recv_bufs,
                      MPI_Datatype                       dtype,
                      int                                comm_size,
                      MPI_Comm                           comm)
{
    std::vector<int> sc(comm_size), sd(comm_size, 0);
    for (int r = 0; r < comm_size; ++r) sc[r] = (int)send_bufs[r].size();
    for (int r = 1; r < comm_size; ++r) sd[r] = sd[r-1] + sc[r-1];

    std::vector<T> sf;
    { size_t tot = 0; for (const auto& b : send_bufs) tot += b.size(); sf.reserve(tot); }
    for (int r = 0; r < comm_size; ++r)
        sf.insert(sf.end(), send_bufs[r].begin(), send_bufs[r].end());

    std::vector<int> rc(comm_size, 0), rd(comm_size, 0);
    MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, comm);
    for (int r = 1; r < comm_size; ++r) rd[r] = rd[r-1] + rc[r-1];

    const size_t total_recv = (size_t)rd[comm_size-1] + rc[comm_size-1];
    std::vector<T> rf(total_recv);
    MPI_Alltoallv(sf.data(), sc.data(), sd.data(), dtype,
                  rf.data(), rc.data(), rd.data(), dtype, comm);

    recv_bufs.assign(comm_size, {});
    for (int r = 0; r < comm_size; ++r)
        recv_bufs[r].assign(rf.begin() + rd[r], rf.begin() + rd[r] + rc[r]);
}

} // namespace dqb_detail


// ---------------------------------------------------------------------------
// DistributedQueryBenchmark
// ---------------------------------------------------------------------------

class DistributedQueryBenchmark {
public:
    int rank      = 0;
    int comm_size = 1;

    int num_shards    = 1;
    int dim           = 0;
    int num_neighbors = 10;

    PointSet          shard_points;
    std::vector<int>  partition;               // base-point → shard mapping
    std::vector<int>  routing_index_partition; // centroid → shard mapping (owned, keeps HNSWRouter ref alive)

#ifdef MIPS_DISTANCE
    using SpaceT = hnswlib::InnerProductSpace;
#else
    using SpaceT = hnswlib::L2Space;
#endif

    std::unique_ptr<SpaceT>                             space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>>    hnsw;
    std::unique_ptr<HNSWRouter>                         router;
    HNSWParameters hnsw_parameters;

    int num_voting_neighbors = 250;
    int num_probes           = 2;

    DistributedQueryBenchmark() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    }

    // -----------------------------------------------------------------------

    void LoadPartition(const std::string& partition_file) {
        // The file is written by WriteClusters (clusters format), NOT by
        // WriteMetisPartition.  ReadMetisPartition would parse the cluster
        // member IDs as shard labels and produce nonsense.
        Clusters clusters = ReadClusters(partition_file);
        num_shards = (int)clusters.size();

        // Find the total number of base points so we can size the array.
        size_t n = 0;
        for (const auto& c : clusters)
            for (uint32_t id : c)
                n = std::max(n, (size_t)(id + 1));

        // Build point → shard mapping.
        partition.assign(n, -1);
        for (int s = 0; s < num_shards; ++s)
            for (uint32_t id : clusters[s])
                partition[id] = s;
    }

    void LoadShardPointSet(const std::string& point_set_file) {
        size_t num_points_in_shard = 0;
        for (const auto& part : partition) {
            if (part == rank) ++num_points_in_shard;
        }

        uint32_t n = 0, d = 0;
        size_t offset = 0;

        std::ifstream in(point_set_file, std::ios::binary);
        in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
        offset += sizeof(uint32_t);
        in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        offset += sizeof(uint32_t);

        shard_points.n = num_points_in_shard;
        shard_points.d = d;
        shard_points.coordinates.resize(shard_points.n * shard_points.d);
        dim = (int)d;

        size_t coords_end = 0;
        for (uint32_t point_id = 0; point_id < n; ++point_id) {
            if (partition[point_id] == rank) {
                size_t begin = offset + (size_t)point_id * d * sizeof(float);
                size_t range_length = 0;
                for ( ; point_id < n && partition[point_id] == rank; ++point_id) {
                    ++range_length;
                }
                in.seekg((std::streamoff)begin);
                in.read(reinterpret_cast<char*>(&shard_points.coordinates[coords_end]),
                        (std::streamsize)(range_length * d * sizeof(float)));
                coords_end += range_length * d;
            }
        }
    }

    void BuildInShardIndex() {
        space = std::make_unique<SpaceT>(dim);
        hnsw  = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space.get(), shard_points.n,
            hnsw_parameters.M, hnsw_parameters.ef_construction,
            /* random seed = */ 555);
        parlay::parallel_for(0, shard_points.n, [&](size_t i) {
            hnsw->addPoint(shard_points.GetPoint(i), i);
        });
        hnsw->setEf(hnsw_parameters.ef_search);
        shard_points.Drop();
    }

    void LoadRouter(const std::string& hnsw_router_file) {
        // The HNSW router indexes centroids, not base points.  Its companion
        // file (*.routing_index_partition) maps centroid IDs → shard IDs.
        // We must pass that partition — NOT the base-point partition — so that
        // router.Query() returns the correct shard for each near centroid.
        routing_index_partition = ReadMetisPartition(hnsw_router_file + ".routing_index_partition");
        router = std::make_unique<HNSWRouter>(hnsw_router_file, dim, routing_index_partition);
    }

    // -----------------------------------------------------------------------
    // Per-nprobe result record returned by ProcessSearchSweep.
    // recall is meaningful only on rank 0; -1.0 means GT was unavailable.
    struct SearchResult {
        int    nprobe;
        double elapsed;   // wall-clock seconds (max across ranks)
        double recall;    // recall@num_neighbors, or -1.0
    };

    // -----------------------------------------------------------------------
    // Sweep nprobe from 1 to num_shards and measure time + recall at each
    // value.  Mirrors ProcessSearchSweep in distributed_insert_bench_sweep.cpp.
    //
    // Routing is computed once per query (full shard ordering).
    // num_voting_neighbors should be >= num_shards so every shard gets a real
    // distance estimate — call set_voting_neighbors_for_sweep() before this.
    //
    // gt_file may be empty or non-existent; recall is then reported as -1.0.
    std::vector<SearchResult> ProcessSearchSweep(PointSet& queries,
                                                 const std::string& gt_file) {
        using namespace dqb_detail;

        const size_t nq    = queries.n;
        const size_t chunk = (nq + comm_size - 1) / comm_size;
        const size_t qs    = (size_t)rank * chunk;
        const size_t qe    = std::min(nq, qs + chunk);

        // ------------------------------------------------------------------
        // Pre-compute full shard ordering for every query in this rank's chunk.
        // ------------------------------------------------------------------
        std::vector<std::vector<int>> routing_order(qe - qs);
        for (size_t q = qs; q < qe; ++q)
            routing_order[q - qs] =
                router->Query(queries.GetPoint(q), num_voting_neighbors).RoutingQuery();

        // ------------------------------------------------------------------
        // Load GT and compute per-query kth-distance threshold for recall.
        // ------------------------------------------------------------------
        const bool has_gt = !gt_file.empty() &&
                            std::filesystem::exists(gt_file);
        std::vector<float> dist_kth(nq, std::numeric_limits<float>::max());
        if (has_gt) {
            auto gt = ReadGroundTruth(gt_file);
            for (size_t q = 0; q < nq; ++q) {
                auto sorted = gt[q];
                std::sort(sorted.begin(), sorted.end());
                if ((int)sorted.size() >= num_neighbors)
                    dist_kth[q] = sorted[num_neighbors - 1].first;
            }
        }

        // ------------------------------------------------------------------
        // Sweep nprobe = 1 .. num_shards.
        // ------------------------------------------------------------------
        std::vector<SearchResult> results;
        results.reserve(num_shards);

        for (int nprobe = 1; nprobe <= num_shards; ++nprobe) {

            // Build per-shard send buffers using the top-nprobe shards.
            std::vector<std::vector<uint32_t>> send_qids(comm_size);
            std::vector<std::vector<float>>    send_qvecs(comm_size);
            for (size_t q = qs; q < qe; ++q) {
                float* Q = queries.GetPoint(q);
                const auto& order = routing_order[q - qs];
                const int np = std::min(nprobe, (int)order.size());
                for (int pi = 0; pi < np; ++pi) {
                    const int s = order[pi];
                    send_qids[s].push_back((uint32_t)q);
                    send_qvecs[s].insert(send_qvecs[s].end(), Q, Q + dim);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            const double t0 = MPI_Wtime();

            // Phase 1: fan queries out to target shards.
            std::vector<std::vector<uint32_t>> recv_qids;
            std::vector<std::vector<float>>    recv_qvecs;
            AllToAllV(send_qids,  recv_qids,  MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
            AllToAllV(send_qvecs, recv_qvecs, MPI_FLOAT,    comm_size, MPI_COMM_WORLD);

            // Search received queries against this rank's shard HNSW.
            struct QTask { int src; size_t j; uint32_t qid; };
            std::vector<QTask> qtasks;
            for (int src = 0; src < comm_size; ++src)
                for (size_t j = 0; j < recv_qids[src].size(); ++j)
                    qtasks.push_back({src, j, recv_qids[src][j]});

            using KNNResult = std::vector<std::pair<float, hnswlib::labeltype>>;
            std::vector<KNNResult> qresults(qtasks.size());
            parlay::parallel_for(0, qtasks.size(), [&](size_t ti) {
                const auto& qt = qtasks[ti];
                float* Q = recv_qvecs[qt.src].data() + qt.j * dim;
                auto pq = hnsw->searchKnn(Q, num_neighbors);
                auto& res = qresults[ti];
                res.reserve(num_neighbors);
                while (!pq.empty()) { res.push_back(pq.top()); pq.pop(); }
                // Pad to num_neighbors so the fan-in stride is always fixed.
                while ((int)res.size() < num_neighbors)
                    res.push_back({std::numeric_limits<float>::max(), 0});
            });

            // Pack results: one row per (shard_src, query) pair.
            std::vector<std::vector<uint32_t>> snd_rqids(comm_size);
            std::vector<std::vector<uint32_t>> snd_rids(comm_size);
            std::vector<std::vector<float>>    snd_rdists(comm_size);
            for (size_t ti = 0; ti < qtasks.size(); ++ti) {
                const int src = qtasks[ti].src;
                snd_rqids[src].push_back(qtasks[ti].qid);
                for (int nn = 0; nn < num_neighbors; ++nn) {
                    snd_rids[src].push_back((uint32_t)qresults[ti][nn].second);
                    snd_rdists[src].push_back(qresults[ti][nn].first);
                }
            }

            // Phase 2: fan results back to query owners.
            std::vector<std::vector<uint32_t>> rcv_rqids, rcv_rids;
            std::vector<std::vector<float>>    rcv_rdists;
            AllToAllV(snd_rqids,  rcv_rqids,  MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
            AllToAllV(snd_rids,   rcv_rids,   MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
            AllToAllV(snd_rdists, rcv_rdists, MPI_FLOAT,    comm_size, MPI_COMM_WORLD);

            const double elapsed = MPI_Wtime() - t0;
            double max_t = 0.0;
            MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            // Merge per-shard results into a single top-k list per query.
            std::vector<NNVec> neighbors(nq);
            for (int src = 0; src < comm_size; ++src) {
                const size_t nres = rcv_rqids[src].size();
                for (size_t j = 0; j < nres; ++j) {
                    const uint32_t qid = rcv_rqids[src][j];
                    for (int nn = 0; nn < num_neighbors; ++nn)
                        neighbors[qid].push_back({
                            rcv_rdists[src][j * num_neighbors + nn],
                            rcv_rids[src][j * num_neighbors + nn]});
                }
            }
            // Sort, dedup, and trim to top-k for each query this rank owns.
            for (size_t q = qs; q < qe; ++q) {
                auto& nv = neighbors[q];
                std::sort(nv.begin(), nv.end());
                nv.erase(std::unique(nv.begin(), nv.end(),
                    [](const auto& a, const auto& b) {
                        return a.second == b.second; }),
                    nv.end());
                if ((int)nv.size() > num_neighbors) nv.resize(num_neighbors);
            }

            // Compute recall: each rank counts hits in its own query slice,
            // then reduce-sum to rank 0.
            double recall = -1.0;
            if (has_gt) {
                uint64_t my_hits = 0;
                for (size_t q = qs; q < qe; ++q)
                    for (const auto& [dist, id] : neighbors[q])
                        if (dist <= dist_kth[q]) ++my_hits;
                uint64_t total_hits = 0;
                MPI_Reduce(&my_hits, &total_hits, 1, MPI_UINT64_T,
                           MPI_SUM, 0, MPI_COMM_WORLD);
                if (rank == 0)
                    recall = (double)total_hits / ((double)nq * num_neighbors);
            }

            results.push_back({nprobe, max_t, recall});
        }

        return results;
    }
};
