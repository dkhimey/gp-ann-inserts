#pragma once

#include "points_io.h"
#include "metis_io.h"
#include "hnsw_router.h"
#include "../external/hnswlib/hnswlib/hnswlib.h"

#include <mpi.h>
#include <parlay/primitives.h>

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
    std::vector<int>  partition;

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
        partition  = ReadMetisPartition(partition_file);
        num_shards = NumPartsInPartition(partition);
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
        router = std::make_unique<HNSWRouter>(hnsw_router_file, dim, partition);
    }

    // -----------------------------------------------------------------------
    // Route a single query vector; returns at most num_probes shard IDs.
    std::vector<int> Route(float* Q) {
        std::vector<int> probes = router->Query(Q, num_voting_neighbors).RoutingQuery();
        if ((int)probes.size() > num_probes)
            probes.resize(num_probes);
        return probes;
    }

    // -----------------------------------------------------------------------
    // Distributed query processing via MPI_Alltoallv (two-phase fan-out/fan-in).
    //
    // Each rank owns a disjoint slice of query_ids.  The method:
    //   1. Routes each query to target shards.
    //   2. Alltoallv fan-out: sends (query_id, query_vector) to target ranks.
    //   3. Each rank runs searchKnn on its local HNSW for received queries.
    //   4. Alltoallv fan-in: returns (query_id, neighbor_ids, distances) to owners.
    //
    // Results are collected but not further processed here; the caller measures
    // wall-clock time end-to-end.
    void ProcessQueries(const std::vector<int>& query_ids, PointSet& queries) {
        using namespace dqb_detail;

        // ------------------------------------------------------------------
        // Phase 1 fan-out: build per-shard send buffers.
        // ------------------------------------------------------------------
        std::vector<std::vector<uint32_t>> send_qids(comm_size);
        std::vector<std::vector<float>>    send_qvecs(comm_size);

        for (int qid : query_ids) {
            float* Q = queries.GetPoint(qid);
            for (int shard : Route(Q)) {
                send_qids[shard].push_back((uint32_t)qid);
                send_qvecs[shard].insert(send_qvecs[shard].end(), Q, Q + dim);
            }
        }

        std::vector<std::vector<uint32_t>> recv_qids;
        std::vector<std::vector<float>>    recv_qvecs;
        AllToAllV(send_qids,  recv_qids,  MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
        AllToAllV(send_qvecs, recv_qvecs, MPI_FLOAT,    comm_size, MPI_COMM_WORLD);

        // ------------------------------------------------------------------
        // Local HNSW search for all received queries.
        // ------------------------------------------------------------------
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
            res.reserve(pq.size());
            while (!pq.empty()) { res.push_back(pq.top()); pq.pop(); }
        });

        // ------------------------------------------------------------------
        // Phase 2 fan-in: pack results and return to query owners.
        // ------------------------------------------------------------------
        std::vector<std::vector<uint32_t>> snd_rqids(comm_size);
        std::vector<std::vector<uint32_t>> snd_rids(comm_size);
        std::vector<std::vector<float>>    snd_rdists(comm_size);

        for (size_t ti = 0; ti < qtasks.size(); ++ti) {
            const int src = qtasks[ti].src;
            snd_rqids[src].push_back(qtasks[ti].qid);
            for (const auto& [dist, label] : qresults[ti]) {
                snd_rids[src].push_back((uint32_t)label);
                snd_rdists[src].push_back(dist);
            }
        }

        std::vector<std::vector<uint32_t>> rcv_rqids, rcv_rids;
        std::vector<std::vector<float>>    rcv_rdists;
        AllToAllV(snd_rqids,  rcv_rqids,  MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
        AllToAllV(snd_rids,   rcv_rids,   MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
        AllToAllV(snd_rdists, rcv_rdists, MPI_FLOAT,    comm_size, MPI_COMM_WORLD);

        // Results available in rcv_r* — communication round-trip complete.
        (void)rcv_rqids; (void)rcv_rids; (void)rcv_rdists;
    }
};
