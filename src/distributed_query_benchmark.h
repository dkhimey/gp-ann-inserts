#pragma once

#include "points_io.h"
#include "metis_io.h"

#include "hnsw_router.h"
#include "../external/hnswlib/hnswlib/hnswlib.h"

#include <mpi.h>
#include <unordered_map>
#include <parlay/primitives.h>

class DistributedQueryBenchmark {
public:
    int rank;
    int comm_size;

    int num_shards;
    int dim;
    int num_neighbors;

    PointSet shard_points;
    std::vector<int> partition;
    std::vector<uint32_t> local_to_global;  // local HNSW ID i → global point ID

    #ifdef MIPS_DISTANCE
    using SpaceT = hnswlib::InnerProductSpace;
    #else
    using SpaceT = hnswlib::L2Space;
    #endif

    std::unique_ptr<SpaceT> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
    std::unique_ptr<HNSWRouter> router;
    HNSWParameters hnsw_parameters;
    int num_voting_neighbors = 250;
    int num_probes = 2;

    DistributedQueryBenchmark() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    }

    void LoadPartition(const std::string& partition_file) {
        partition = ReadMetisPartition(partition_file);
        num_shards = NumPartsInPartition(partition);
    }

    void LoadShardPointSet(const std::string& point_set_file) {
        size_t num_points_in_shard = 0;
        for (const auto& part : partition) {
            if (part == rank) {
                num_points_in_shard++;
            }
        }

        uint32_t n, d;
        size_t offset = 0;

        std::ifstream in(point_set_file, std::ios::binary);
        in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
        offset += sizeof(uint32_t);
        in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        offset += sizeof(uint32_t);

        shard_points.n = num_points_in_shard;
        shard_points.d = d;
        shard_points.coordinates.resize(shard_points.n * shard_points.d);
        dim = d;

        size_t coords_end = 0;
        for (uint32_t point_id = 0; point_id < n; ++point_id) {
            if (partition[point_id] == rank) {
                size_t begin = offset + point_id * d * sizeof(float);
                size_t range_length = 0;
                for ( ; point_id < n && partition[point_id] == rank; ++point_id) {
                    range_length += 1;
                }

                in.seekg(begin);
                in.read(reinterpret_cast<char*>(&shard_points.coordinates[coords_end]), range_length * d * sizeof(float));
                coords_end += range_length * d;
            }
        }

        // Build local-to-global ID map: local HNSW label i → global point index.
        // Points are loaded in ascending global-ID order, so this is just a filter.
        local_to_global.clear();
        local_to_global.reserve(shard_points.n);
        for (size_t i = 0; i < partition.size(); i++) {
            if (partition[i] == rank)
                local_to_global.push_back(static_cast<uint32_t>(i));
        }
    }

    void BuildInShardIndex() {
        space = std::make_unique<SpaceT>(dim);
        hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), shard_points.n, hnsw_parameters.M, hnsw_parameters.ef_construction, /* random seed = */ 555);
        parlay::parallel_for(0, shard_points.n, [&](size_t i) { hnsw->addPoint(shard_points.GetPoint(i), i); });
        hnsw->setEf(hnsw_parameters.ef_search);
        shard_points.Drop();    // TODO we could do the HNSW insert during IO --> halve the memory requirement
    }

    void LoadRouter(const std::string& hnsw_router_file) {
        router = std::make_unique<HNSWRouter>(hnsw_router_file, dim, partition);
    }

    std::vector<int> Route(float* Q) {
        std::vector<int> probes = router->Query(Q, num_voting_neighbors).RoutingQuery();
        probes.resize(std::min<int>(probes.size(), num_probes));
        return probes;
    }

    // Route and execute all queries in query_ids against the distributed index.
    //
    // nprobe           – number of shards to probe per query (sweepable, 1..num_shards).
    //                    Sets this->num_probes before routing so Route() picks it up.
    // out_parts_searched – set to total shard visits across all queries in query_ids
    //                      (sum before MPI_Reduce; caller reduces across ranks).
    // collect_results  – if true, translate local→global IDs and return merged
    //                    neighbor lists per query (used for recall on warmup passes).
    //                    if false, Phase 4 still runs (fair timing) but results are
    //                    discarded (used for timed QPS passes).
    std::vector<std::vector<uint32_t>> ProcessQueries(
            const std::vector<int>& query_ids, PointSet& queries,
            int nprobe, long long& out_parts_searched, bool collect_results = true) {

        num_probes = nprobe;    // Route() reads this->num_probes

        // ── Phase 1: route assigned queries into per-rank send buffers ───────────
        std::vector<std::vector<int>>   send_qids(comm_size);
        std::vector<std::vector<float>> send_vecs(comm_size);
        long long my_parts = 0;

        for (int qid : query_ids) {
            float* Q = queries.GetPoint(qid);
            std::vector<int> targets = Route(Q);
            my_parts += static_cast<long long>(targets.size());
            for (int shard : targets) {
                send_qids[shard].push_back(qid);
                send_vecs[shard].insert(send_vecs[shard].end(), Q, Q + dim);
            }
        }
        out_parts_searched = my_parts;

        // ── Phase 2: AllToAllV — dispatch query IDs and vectors to target shards ─
        std::vector<int> qid_send_counts(comm_size), qid_recv_counts(comm_size);
        for (int r = 0; r < comm_size; r++)
            qid_send_counts[r] = static_cast<int>(send_qids[r].size());
        MPI_Alltoall(qid_send_counts.data(), 1, MPI_INT,
                     qid_recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> qid_send_displs(comm_size, 0), qid_recv_displs(comm_size, 0);
        for (int r = 1; r < comm_size; r++) {
            qid_send_displs[r] = qid_send_displs[r-1] + qid_send_counts[r-1];
            qid_recv_displs[r] = qid_recv_displs[r-1] + qid_recv_counts[r-1];
        }
        const int total_recv = qid_recv_displs[comm_size-1] + qid_recv_counts[comm_size-1];

        std::vector<int> flat_send_qids, flat_recv_qids(total_recv);
        for (int r = 0; r < comm_size; r++)
            flat_send_qids.insert(flat_send_qids.end(), send_qids[r].begin(), send_qids[r].end());
        MPI_Alltoallv(flat_send_qids.data(), qid_send_counts.data(), qid_send_displs.data(), MPI_INT,
                      flat_recv_qids.data(), qid_recv_counts.data(), qid_recv_displs.data(), MPI_INT,
                      MPI_COMM_WORLD);

        std::vector<int> vec_send_counts(comm_size), vec_recv_counts(comm_size);
        std::vector<int> vec_send_displs(comm_size, 0), vec_recv_displs(comm_size, 0);
        for (int r = 0; r < comm_size; r++) {
            vec_send_counts[r] = qid_send_counts[r] * dim;
            vec_recv_counts[r] = qid_recv_counts[r] * dim;
        }
        for (int r = 1; r < comm_size; r++) {
            vec_send_displs[r] = vec_send_displs[r-1] + vec_send_counts[r-1];
            vec_recv_displs[r] = vec_recv_displs[r-1] + vec_recv_counts[r-1];
        }

        std::vector<float> flat_send_vecs, flat_recv_vecs(static_cast<size_t>(total_recv) * dim);
        for (int r = 0; r < comm_size; r++)
            flat_send_vecs.insert(flat_send_vecs.end(), send_vecs[r].begin(), send_vecs[r].end());
        MPI_Alltoallv(flat_send_vecs.data(), vec_send_counts.data(), vec_send_displs.data(), MPI_FLOAT,
                      flat_recv_vecs.data(), vec_recv_counts.data(), vec_recv_displs.data(), MPI_FLOAT,
                      MPI_COMM_WORLD);

        // ── Phase 3: search local HNSW, translate local→global IDs ──────────────
        // Owner of query qid = qid / chunk_size, matching the partition in main().
        const size_t chunk_size = (static_cast<size_t>(queries.n) + comm_size - 1) / comm_size;

        std::vector<std::vector<int>> snd_rqids(comm_size);
        std::vector<std::vector<int>> snd_rids(comm_size);

        for (int i = 0; i < total_recv; i++) {
            float* Q = flat_recv_vecs.data() + static_cast<size_t>(i) * dim;
            auto result = hnsw->searchKnn(Q, num_neighbors);
            const int qid   = flat_recv_qids[i];
            const int owner = static_cast<int>(static_cast<size_t>(qid) / chunk_size);
            snd_rqids[owner].push_back(qid);
            int n_returned = 0;
            while (!result.empty() && n_returned < num_neighbors) {
                uint32_t local_id = static_cast<uint32_t>(result.top().second);
                snd_rids[owner].push_back(static_cast<int>(local_to_global[local_id]));
                result.pop();
                n_returned++;
            }
            // Pad to num_neighbors so Phase 4 can use a fixed stride.
            while (n_returned < num_neighbors) {
                snd_rids[owner].push_back(-1);
                n_returned++;
            }
        }

        // ── Phase 4: AllToAllV — return global neighbor IDs to query owners ──────
        std::vector<int> rqid_send_counts(comm_size), rqid_recv_counts(comm_size);
        for (int r = 0; r < comm_size; r++)
            rqid_send_counts[r] = static_cast<int>(snd_rqids[r].size());
        MPI_Alltoall(rqid_send_counts.data(), 1, MPI_INT,
                     rqid_recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> rqid_send_displs(comm_size, 0), rqid_recv_displs(comm_size, 0);
        for (int r = 1; r < comm_size; r++) {
            rqid_send_displs[r] = rqid_send_displs[r-1] + rqid_send_counts[r-1];
            rqid_recv_displs[r] = rqid_recv_displs[r-1] + rqid_recv_counts[r-1];
        }
        const int total_rqid_recv = rqid_recv_displs[comm_size-1] + rqid_recv_counts[comm_size-1];

        std::vector<int> flat_snd_rqids, flat_rcv_rqids(total_rqid_recv);
        for (int r = 0; r < comm_size; r++)
            flat_snd_rqids.insert(flat_snd_rqids.end(), snd_rqids[r].begin(), snd_rqids[r].end());
        MPI_Alltoallv(flat_snd_rqids.data(), rqid_send_counts.data(), rqid_send_displs.data(), MPI_INT,
                      flat_rcv_rqids.data(), rqid_recv_counts.data(), rqid_recv_displs.data(), MPI_INT,
                      MPI_COMM_WORLD);

        std::vector<int> rid_send_counts(comm_size), rid_recv_counts(comm_size);
        std::vector<int> rid_send_displs(comm_size, 0), rid_recv_displs(comm_size, 0);
        for (int r = 0; r < comm_size; r++) {
            rid_send_counts[r] = rqid_send_counts[r] * num_neighbors;
            rid_recv_counts[r] = rqid_recv_counts[r] * num_neighbors;
        }
        for (int r = 1; r < comm_size; r++) {
            rid_send_displs[r] = rid_send_displs[r-1] + rid_send_counts[r-1];
            rid_recv_displs[r] = rid_recv_displs[r-1] + rid_recv_counts[r-1];
        }

        std::vector<int> flat_snd_rids, flat_rcv_rids(total_rqid_recv * num_neighbors);
        for (int r = 0; r < comm_size; r++)
            flat_snd_rids.insert(flat_snd_rids.end(), snd_rids[r].begin(), snd_rids[r].end());
        MPI_Alltoallv(flat_snd_rids.data(), rid_send_counts.data(), rid_send_displs.data(), MPI_INT,
                      flat_rcv_rids.data(), rid_recv_counts.data(), rid_recv_displs.data(), MPI_INT,
                      MPI_COMM_WORLD);

        // ── Phase 5: merge results per query (warmup / recall passes only) ───────
        if (!collect_results) return {};

        std::unordered_map<int, int> qid_to_idx;
        qid_to_idx.reserve(query_ids.size());
        for (int i = 0; i < static_cast<int>(query_ids.size()); i++)
            qid_to_idx[query_ids[i]] = i;

        std::vector<std::vector<uint32_t>> results(query_ids.size());
        for (int j = 0; j < total_rqid_recv; j++) {
            int qid = flat_rcv_rqids[j];
            auto it = qid_to_idx.find(qid);
            if (it == qid_to_idx.end()) continue;
            int idx = it->second;
            for (int nn = 0; nn < num_neighbors; nn++) {
                int rid = flat_rcv_rids[j * num_neighbors + nn];
                if (rid >= 0) results[idx].push_back(static_cast<uint32_t>(rid));
            }
        }
        // Deduplicate across shard contributions and trim to k.
        for (auto& r : results) {
            std::sort(r.begin(), r.end());
            r.erase(std::unique(r.begin(), r.end()), r.end());
            if (static_cast<int>(r.size()) > num_neighbors) r.resize(num_neighbors);
        }
        return results;
    }

};
