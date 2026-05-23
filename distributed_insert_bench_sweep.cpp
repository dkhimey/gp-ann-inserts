// distributed_insert_bench_sweep.cpp
//
// Same as distributed_insert_bench but sweeps nprobe from 1 to num_shards
// during every search step rather than using a fixed nprobe.  The index is
// built and mutated exactly once; each search step is repeated num_shards
// times (once per nprobe value), so per-nprobe recall and timing are measured
// without re-running the expensive build/insert/delete phases.
//
// num_voting_neighbors is set to num_shards so that every shard has a real
// distance estimate for all nprobe values (no shard is left at float::max).
//
// Usage
// -----
//   mpirun -np <K> ./DistributedInsertBenchSweep
//          <base.fbin> <queries.fbin> <partition-file>
//          <gt-prefix> <runbook.yaml>
//          <num-neighbors> [output-file]
//
//   K must equal the number of shards in the partition file.
//   GT files: <gt-prefix>/step<i>.gt100
//   Output CSV columns: step, operation, nprobe, range_start, range_end,
//                       time_s, qps_or_throughput, recall@K,
//                       shard_0_slots, shard_0_active, ...

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <parlay/primitives.h>

#include "../external/hnswlib/hnswlib/hnswlib.h"

#include "defs.h"
#include "hnsw_router.h"
#include "kmeans_tree_router.h"
#include "metis_io.h"
#include "points_io.h"


// ---------------------------------------------------------------------------
// Runbook (identical to distributed_insert_bench)
// ---------------------------------------------------------------------------

struct Operation {
    enum class Type { INSERT, DELETE, SEARCH };
    Type     type     = Type::SEARCH;
    uint32_t step_num = 0;
    uint32_t start    = 0;
    uint32_t end      = 0;
};

struct Runbook {
    std::string            dataset_name;
    size_t                 max_pts = 0;
    std::vector<Operation> ops;
};

static std::string trim(const std::string& s) {
    size_t l = s.find_first_not_of(" \t\r\n");
    size_t r = s.find_last_not_of(" \t\r\n");
    return (l == std::string::npos) ? "" : s.substr(l, r - l + 1);
}
static bool sw(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

static Runbook ParseRunbook(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open runbook: " + path);
    Runbook rb;
    Operation cur;
    bool in_op = false;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] != ' ' && line.back() == ':') {
            rb.dataset_name = line.substr(0, line.size() - 1); continue;
        }
        if (sw(line, "  max_pts:")) {
            std::istringstream ss(line.substr(line.find(':') + 1));
            ss >> rb.max_pts; continue;
        }
        if (line.size() > 2 && line[0]==' ' && line[1]==' ' && line[2]!=' ') {
            size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::string key = trim(line.substr(0, colon));
            if (key.empty() || !std::all_of(key.begin(), key.end(), ::isdigit))
                continue;
            if (in_op) rb.ops.push_back(cur);
            cur = Operation{};
            cur.step_num = static_cast<uint32_t>(std::stoul(key));
            in_op = true; continue;
        }
        if (!in_op) continue;
        if (sw(line, "    operation:")) {
            std::string v = trim(line.substr(line.find(':') + 1));
            v.erase(std::remove(v.begin(), v.end(), '\''), v.end());
            v.erase(std::remove(v.begin(), v.end(), '"'),  v.end());
            if      (v == "insert") cur.type = Operation::Type::INSERT;
            else if (v == "delete") cur.type = Operation::Type::DELETE;
            else if (v == "search") cur.type = Operation::Type::SEARCH;
            else throw std::runtime_error("Unknown op: " + v);
        } else if (sw(line, "    start:")) {
            std::istringstream ss(line.substr(line.find(':') + 1)); ss >> cur.start;
        } else if (sw(line, "    end:")) {
            std::istringstream ss(line.substr(line.find(':') + 1)); ss >> cur.end;
        }
    }
    if (in_op) rb.ops.push_back(cur);
    return rb;
}


// ---------------------------------------------------------------------------
// Seek-based range read for .fbin files
// ---------------------------------------------------------------------------

static PointSet ReadPointsRange(const std::string& path,
                                uint32_t start, uint32_t end) {
    if (!path.ends_with(".fbin"))
        throw std::runtime_error("ReadPointsRange only supports .fbin");
    uint32_t n_file = 0, d = 0;
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open: " + path);
        in.read(reinterpret_cast<char*>(&n_file), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&d),      sizeof(uint32_t));
    }
    end = std::min(end, n_file);
    const size_t count = (end > start) ? end - start : 0;
    PointSet ps; ps.n = count; ps.d = d;
    ps.coordinates.resize(count * d, 0.f);
    if (count > 0) {
        std::ifstream in(path, std::ios::binary);
        in.seekg(static_cast<std::streamoff>(
            2 * sizeof(uint32_t) +
            static_cast<size_t>(start) * d * sizeof(float)));
        in.read(reinterpret_cast<char*>(ps.coordinates.data()),
                static_cast<std::streamsize>(count * d * sizeof(float)));
    }
    return ps;
}


// ---------------------------------------------------------------------------
// Generic Alltoallv helper
// ---------------------------------------------------------------------------

template<typename T>
static void AllToAllV(const std::vector<std::vector<T>>& send_bufs,
                      std::vector<std::vector<T>>&       recv_bufs,
                      MPI_Datatype                       dtype,
                      int comm_size, MPI_Comm comm) {
    std::vector<int> sc(comm_size), sd(comm_size, 0);
    for (int r = 0; r < comm_size; ++r) sc[r] = (int)send_bufs[r].size();
    for (int r = 1; r < comm_size; ++r) sd[r] = sd[r-1] + sc[r-1];
    std::vector<T> sf;
    { size_t tot = 0; for (auto& b : send_bufs) tot += b.size(); sf.reserve(tot); }
    for (int r = 0; r < comm_size; ++r)
        sf.insert(sf.end(), send_bufs[r].begin(), send_bufs[r].end());

    std::vector<int> rc(comm_size, 0), rd(comm_size, 0);
    MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, comm);
    for (int r = 1; r < comm_size; ++r) rd[r] = rd[r-1] + rc[r-1];
    std::vector<T> rf(rd.back() + rc.back());
    MPI_Alltoallv(sf.data(), sc.data(), sd.data(), dtype,
                  rf.data(), rc.data(), rd.data(), dtype, comm);
    recv_bufs.assign(comm_size, {});
    for (int r = 0; r < comm_size; ++r)
        recv_bufs[r].assign(rf.begin() + rd[r], rf.begin() + rd[r] + rc[r]);
}


// ---------------------------------------------------------------------------
// Benchmark class (same as distributed_insert_bench; ProcessSearch extended
// to return per-nprobe results)
// ---------------------------------------------------------------------------

class DistributedInsertBenchmark {
public:
    int rank = 0, comm_size = 1, num_shards = 1, dim = 0;
    int num_neighbors        = 10;
    int num_voting_neighbors = 100;

    // Per-shard size information gathered from all ranks.
    // slots  = cur_element_count  (total graph nodes, including deleted)
    // active = slots - num_deleted_  (live, query-visible elements)
    struct ShardInfo {
        std::vector<size_t> slots;
        std::vector<size_t> active;
    };

    ShardInfo GatherShardInfo() const {
        const size_t local_slots   = local_hnsw ? local_hnsw->cur_element_count.load() : 0;
        const size_t local_deleted = local_hnsw ? local_hnsw->num_deleted_ : 0;
        const size_t local_active  = local_slots - local_deleted;

        const unsigned long long my_slots  = static_cast<unsigned long long>(local_slots);
        const unsigned long long my_active = static_cast<unsigned long long>(local_active);

        std::vector<unsigned long long> all_slots(comm_size, 0);
        std::vector<unsigned long long> all_active(comm_size, 0);
        MPI_Gather(&my_slots,  1, MPI_UNSIGNED_LONG_LONG,
                   all_slots.data(),  1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Gather(&my_active, 1, MPI_UNSIGNED_LONG_LONG,
                   all_active.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

        if (rank != 0) return {};
        ShardInfo info;
        info.slots.resize(comm_size);
        info.active.resize(comm_size);
        for (int r = 0; r < comm_size; ++r) {
            info.slots[r]  = static_cast<size_t>(all_slots[r]);
            info.active[r] = static_cast<size_t>(all_active[r]);
        }
        return info;
    }

    std::unique_ptr<KMeansTreeRouter> kmtr;
    PointSet routing_points; std::vector<int> routing_partition;
    std::unique_ptr<HNSWRouter> router;

#ifdef MIPS_DISTANCE
    std::unique_ptr<hnswlib::InnerProductSpace> space;
#else
    std::unique_ptr<hnswlib::L2Space> space;
#endif
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> local_hnsw;

    std::unordered_map<uint32_t, int32_t> label_to_shard;

    DistributedInsertBenchmark() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        num_shards = comm_size;
    }

    // -----------------------------------------------------------------------
    void BuildInitial(const std::string& point_file,
                      const Clusters& clusters,
                      size_t max_pts,
                      uint32_t batch_start, uint32_t batch_end) {
        if (comm_size != (int)clusters.size()) {
            if (rank == 0)
                std::cerr << "comm_size=" << comm_size
                          << " must equal num_shards=" << clusters.size() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        label_to_shard.clear();
        label_to_shard.reserve(max_pts);
        for (int r = 0; r < comm_size; ++r)
            for (uint32_t id : clusters[r])
                label_to_shard[id] = r;

        PointSet init = ReadPointsRange(point_file, batch_start, batch_end);
        dim = (int)init.d;

        Clusters local_clusters(clusters.size());
        for (size_t b = 0; b < clusters.size(); ++b) {
            local_clusters[b].resize(clusters[b].size());
            for (size_t i = 0; i < clusters[b].size(); ++i)
                local_clusters[b][i] = clusters[b][i] - batch_start;
        }

        KMeansTreeRouterOptions opts{
            .num_centroids=32, .min_cluster_size=200,
            .budget=50000, .search_budget=5000 };
        kmtr = std::make_unique<KMeansTreeRouter>();
        kmtr->Train(init, local_clusters, opts);
        { auto [rp, rpart] = kmtr->ExtractPoints();
          routing_points = std::move(rp); routing_partition = std::move(rpart); }
        router = std::make_unique<HNSWRouter>(
            routing_points, num_shards, routing_partition,
            HNSWParameters{ .M=16, .ef_construction=200, .ef_search=200 });
        router->Train(routing_points);

        space = std::make_unique<
#ifdef MIPS_DISTANCE
            hnswlib::InnerProductSpace
#else
            hnswlib::L2Space
#endif
        >(dim);

        const size_t init_size   = batch_end - batch_start;
        const size_t shard_size  = clusters[rank].size();
        const double ratio       = init_size > 0
            ? (double)shard_size / init_size : 0.0;
        const size_t capacity    = std::max(shard_size + 1,
                                            (size_t)(ratio * max_pts) + 1);

        local_hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space.get(), capacity, 16, 200, 555 + rank);
        local_hnsw->setEf(120);

        const auto& my_cluster = clusters[rank];
        parlay::parallel_for(0, my_cluster.size(), [&](size_t i) {
            uint32_t id = my_cluster[i];
            float* p = init.GetPoint(id - batch_start);
            local_hnsw->addPoint(p, id);
        });

        if (rank == 0)
            std::cout << "  shard_size=" << shard_size
                      << "  capacity=" << capacity << "\n";
    }

    // -----------------------------------------------------------------------
    double ProcessInserts(const std::string& point_file,
                          uint32_t start, uint32_t end) {
        PointSet batch = ReadPointsRange(point_file, start, end);

        // Each rank routes only its disjoint slice (i % comm_size == rank).
        // label_to_shard is then synced via MPI_Allgatherv so every rank has
        // a consistent map — required for correct no-communication deletes.
        std::vector<std::vector<uint32_t>> send_ids(comm_size);
        std::vector<std::vector<float>>    send_vecs(comm_size);
        for (uint32_t i = 0; i < batch.n; ++i) {
            if ((int)(i % comm_size) != rank) continue;
            const uint32_t gid = start + i;
            float* p = batch.GetPoint(i);
            const int t = router->NaiveRoute(p);
            send_ids[t].push_back(gid);
            send_vecs[t].insert(send_vecs[t].end(), p, p + dim);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();

        std::vector<std::vector<uint32_t>> recv_ids;
        std::vector<std::vector<float>>    recv_vecs;
        AllToAllV(send_ids,  recv_ids,  MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
        AllToAllV(send_vecs, recv_vecs, MPI_FLOAT,    comm_size, MPI_COMM_WORLD);

        std::vector<uint32_t> ins_ids;
        std::vector<float*>   ins_ptrs;
        for (int src = 0; src < comm_size; ++src)
            for (size_t j = 0; j < recv_ids[src].size(); ++j) {
                ins_ids.push_back(recv_ids[src][j]);
                ins_ptrs.push_back(recv_vecs[src].data() + j * dim);
            }
        {
            const size_t incoming = ins_ids.size();
            if (local_hnsw->cur_element_count + incoming > local_hnsw->max_elements_) {
                size_t new_max = std::max(
                    local_hnsw->cur_element_count + incoming,
                    local_hnsw->max_elements_ + local_hnsw->max_elements_ / 2);
                local_hnsw->resizeIndex(new_max);
            }
        }
        parlay::parallel_for(0, ins_ids.size(), [&](size_t i) {
            local_hnsw->addPoint(ins_ptrs[i], ins_ids[i]);
        });

        const double elapsed = MPI_Wtime() - t0;
        double max_t = 0.0;
        MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Broadcast the actual label→shard assignments to all ranks.
        // NaiveRoute is non-deterministic (parallel HNSW), so each rank must
        // learn the true destinations decided by other ranks.
        {
            int local_n = (int)ins_ids.size();
            std::vector<int> all_ns(comm_size);
            MPI_Allgather(&local_n, 1, MPI_INT,
                          all_ns.data(), 1, MPI_INT, MPI_COMM_WORLD);
            std::vector<int> disps(comm_size, 0);
            for (int r = 1; r < comm_size; ++r) disps[r] = disps[r-1] + all_ns[r-1];
            const int total = disps[comm_size-1] + all_ns[comm_size-1];
            std::vector<uint32_t> all_labels(total);
            MPI_Allgatherv(ins_ids.data(), local_n, MPI_UINT32_T,
                           all_labels.data(), all_ns.data(), disps.data(),
                           MPI_UINT32_T, MPI_COMM_WORLD);
            for (int r = 0; r < comm_size; ++r)
                for (int j = disps[r]; j < disps[r] + all_ns[r]; ++j)
                    label_to_shard[all_labels[j]] = r;
        }

        return max_t;
    }

    // -----------------------------------------------------------------------
    double ProcessDeletes(uint32_t start, uint32_t end) {
        // label_to_shard is consistent on all ranks (kept in sync by
        // MPI_Allgatherv in ProcessInserts).  Each rank directly marks only
        // the labels it owns — no AlltoAllv needed.
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();

        for (uint32_t id = start; id < end; ++id) {
            auto it = label_to_shard.find(id);
            if (it == label_to_shard.end()) continue;
            if (it->second == rank)
                local_hnsw->markDelete(id);
            label_to_shard.erase(it);
        }

        const double elapsed = MPI_Wtime() - t0;
        double max_t = 0.0;
        MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        return max_t;
    }

    // -----------------------------------------------------------------------
    // Run a search sweep over nprobe = 1 .. num_shards.
    //
    // Routing is computed once per query (finding the sorted shard order).
    // Then for each nprobe value we run a separate Alltoallv fan-out so that
    // timing reflects the actual cost at that nprobe.
    //
    // Returns a vector of (nprobe, elapsed_s, recall) for each nprobe value.
    // recall is meaningful only on rank 0; set to -1.0 if GT file is missing.
    struct SearchResult { int nprobe; double elapsed; double recall; };

    std::vector<SearchResult> ProcessSearchSweep(PointSet& queries,
                                                 const std::string& gt_file) {
        const size_t nq    = queries.n;
        const size_t chunk = (nq + comm_size - 1) / comm_size;
        const size_t qs    = (size_t)rank * chunk;
        const size_t qe    = std::min(nq, qs + chunk);

        // Compute routing order for each query in this rank's chunk.
        // routing_order[q - qs][s] = the s-th shard to probe for query q.
        std::vector<std::vector<int>> routing_order(qe - qs);
        for (size_t q = qs; q < qe; ++q) {
            float* Q = queries.GetPoint(q);
            routing_order[q - qs] =
                router->Query(Q, num_voting_neighbors).RoutingQuery();
        }

        // Load GT and compute kth-distance thresholds (needed for recall).
        bool has_gt = !gt_file.empty() && std::filesystem::exists(gt_file);
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

        std::vector<SearchResult> results;
        results.reserve(num_shards);

        for (int nprobe = 1; nprobe <= num_shards; ++nprobe) {

            // Build per-shard send buffers using only the top-nprobe shards.
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

            // Phase 1: fan queries out to shards.
            std::vector<std::vector<uint32_t>> recv_qids;
            std::vector<std::vector<float>>    recv_qvecs;
            AllToAllV(send_qids,  recv_qids,  MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
            AllToAllV(send_qvecs, recv_qvecs, MPI_FLOAT,    comm_size, MPI_COMM_WORLD);

            // Search received queries in parallel.
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
                auto pq = local_hnsw->searchKnn(Q, num_neighbors);
                auto& res = qresults[ti];
                res.reserve(num_neighbors);
                while (!pq.empty()) { res.push_back(pq.top()); pq.pop(); }
                while ((int)res.size() < num_neighbors)
                    res.push_back({std::numeric_limits<float>::max(), 0});
            });

            // Pack results for return.
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

            // Phase 2: return results to query owners.
            std::vector<std::vector<uint32_t>> rcv_rqids, rcv_rids;
            std::vector<std::vector<float>>    rcv_rdists;
            AllToAllV(snd_rqids,  rcv_rqids,  MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
            AllToAllV(snd_rids,   rcv_rids,   MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
            AllToAllV(snd_rdists, rcv_rdists, MPI_FLOAT,    comm_size, MPI_COMM_WORLD);

            const double elapsed = MPI_Wtime() - t0;
            double max_t = 0.0;
            MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            // Merge results per query.
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
            for (size_t q = qs; q < qe; ++q) {
                auto& nv = neighbors[q];
                std::sort(nv.begin(), nv.end());
                nv.erase(std::unique(nv.begin(), nv.end(),
                    [](const auto& a, const auto& b){ return a.second == b.second; }),
                    nv.end());
                if ((int)nv.size() > num_neighbors) nv.resize(num_neighbors);
            }

            // Compute recall.
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


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, const char* argv[]) {
    MPI_Init(nullptr, nullptr);
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (rank != 0) std::cout.setstate(std::ios_base::failbit);

    if (argc < 7 || argc > 8) {
        if (rank == 0)
            std::cerr <<
                "Usage: mpirun -np <K> ./DistributedInsertBenchSweep"
                " <base.fbin> <queries.fbin> <partition-file>"
                " <gt-prefix> <runbook.yaml>"
                " <num-neighbors> [output-file]\n";
        MPI_Finalize(); return 1;
    }

    const std::string point_file     = argv[1];
    const std::string query_file     = argv[2];
    const std::string partition_file = argv[3];
    const std::string gt_prefix      = argv[4];
    const std::string runbook_path   = argv[5];
    const int         num_neighbors  = std::stoi(argv[6]);
    const std::string output_file    = (argc == 8)
        ? argv[7] : "sweep_results.csv";

    if (rank == 0) std::cout.clear();

    if (rank == 0)
        std::cout << "Parsing runbook " << runbook_path << " …\n";
    Runbook rb = ParseRunbook(runbook_path);
    if (rank == 0)
        std::cout << "Dataset : " << rb.dataset_name << "\n"
                  << "max_pts : " << rb.max_pts << "\n"
                  << "ops     : " << rb.ops.size() << "\n"
                  << "shards  : " << comm_size << "\n"
                  << "output  : " << output_file << "\n\n";

    // Open CSV on rank 0.
    std::ofstream csv;
    if (rank == 0) {
        csv.open(output_file);
        if (!csv) {
            std::cerr << "Cannot open output file: " << output_file << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        csv << "step,operation,nprobe,range_start,range_end,"
            << "time_s,qps_or_throughput,recall@" << num_neighbors;
        for (int r = 0; r < comm_size; ++r)
            csv << ",shard_" << r << "_slots,shard_" << r << "_active";
        csv << "\n";
    }

    if (rank != 0) std::cout.setstate(std::ios_base::failbit);
    PointSet queries = ReadPoints(query_file);
    if (rank == 0) std::cout.clear();

    if (rb.ops.empty() || rb.ops[0].type != Operation::Type::INSERT) {
        if (rank == 0) std::cerr << "Runbook must begin with an insert.\n";
        MPI_Finalize(); return 1;
    }

    DistributedInsertBenchmark bench;
    bench.num_neighbors       = num_neighbors;
    // num_voting_neighbors = 3*num_shards so every shard gets a real distance
    // estimate regardless of which nprobe value we're sweeping.
    bench.num_voting_neighbors = 10*comm_size;

    bool index_built = false;

    for (size_t op_idx = 0; op_idx < rb.ops.size(); ++op_idx) {
        const Operation& op = rb.ops[op_idx];

        // ── INSERT ────────────────────────────────────────────────────────
        if (op.type == Operation::Type::INSERT) {
            if (!index_built) {
                if (rank == 0)
                    std::cout << "[op " << op_idx+1 << "] BUILD"
                              << "  base[" << op.start << "," << op.end << ")\n";
                if (rank != 0) std::cout.setstate(std::ios_base::failbit);
                Clusters clusters = ReadClusters(partition_file);
                if (rank == 0) std::cout.clear();
                if (rb.max_pts == 0) rb.max_pts = op.end + 1;

                MPI_Barrier(MPI_COMM_WORLD);
                const double t0 = MPI_Wtime();
                if (rank != 0) std::cout.setstate(std::ios_base::failbit);
                bench.BuildInitial(point_file, clusters, rb.max_pts,
                                   op.start, op.end);
                if (rank == 0) std::cout.clear();
                MPI_Barrier(MPI_COMM_WORLD);
                const double build_t  = MPI_Wtime() - t0;
                const double build_tp = build_t > 0
                    ? (double)(op.end - op.start) / build_t : 0.0;
                if (rank == 0)
                    std::cout << "  build_time=" << build_t << " s\n";
                index_built = true;

                if (rank == 0) {
                    auto si = bench.GatherShardInfo();
                    csv << op.step_num << ",BUILD,N/A,"
                        << op.start << "," << op.end << ","
                        << build_t << "," << build_tp << ",N/A";
                    for (int r = 0; r < comm_size; ++r)
                        csv << "," << si.slots[r] << "," << si.active[r];
                    csv << "\n"; csv.flush();
                } else { bench.GatherShardInfo(); }

            } else {
                if (rank == 0)
                    std::cout << "[op " << op_idx+1 << "] INSERT"
                              << "  base[" << op.start << "," << op.end << ")"
                              << "  (" << op.end-op.start << " vectors)\n";
                if (rank != 0) std::cout.setstate(std::ios_base::failbit);
                const double t  = bench.ProcessInserts(point_file, op.start, op.end);
                if (rank == 0) std::cout.clear();
                const double tp = t > 0 ? (double)(op.end-op.start)/t : 0.0;
                if (rank == 0)
                    std::cout << "  insert_time=" << t << " s"
                              << "  throughput=" << tp << " vec/s\n";

                if (rank == 0) {
                    auto si = bench.GatherShardInfo();
                    csv << op.step_num << ",INSERT,N/A,"
                        << op.start << "," << op.end << ","
                        << t << "," << tp << ",N/A";
                    for (int r = 0; r < comm_size; ++r)
                        csv << "," << si.slots[r] << "," << si.active[r];
                    csv << "\n"; csv.flush();
                } else { bench.GatherShardInfo(); }
            }
        }

        // ── DELETE ────────────────────────────────────────────────────────
        else if (op.type == Operation::Type::DELETE) {
            if (!index_built) continue;
            if (rank == 0)
                std::cout << "[op " << op_idx+1 << "] DELETE"
                          << "  base[" << op.start << "," << op.end << ")"
                          << "  (" << op.end-op.start << " vectors)\n";
            const double t  = bench.ProcessDeletes(op.start, op.end);
            const double tp = t > 0 ? (double)(op.end-op.start)/t : 0.0;
            if (rank == 0)
                std::cout << "  delete_time=" << t << " s"
                          << "  throughput=" << tp << " vec/s\n";

            if (rank == 0) {
                auto si = bench.GatherShardInfo();
                csv << op.step_num << ",DELETE,N/A,"
                    << op.start << "," << op.end << ","
                    << t << "," << tp << ",N/A";
                for (int r = 0; r < comm_size; ++r)
                    csv << "," << si.slots[r] << "," << si.active[r];
                csv << "\n"; csv.flush();
            } else { bench.GatherShardInfo(); }
        }

        // ── SEARCH (swept over all nprobe values) ─────────────────────────
        else {
            if (!index_built) continue;
            const std::string gt_file =
                gt_prefix + "/step" + std::to_string(op.step_num) + ".gt100";
            if (rank == 0)
                std::cout << "[op " << op.step_num << "] SEARCH"
                          << "  gt=" << gt_file
                          << "  sweeping nprobe 1.." << comm_size << "\n";

            if (rank != 0) std::cout.setstate(std::ios_base::failbit);
            auto sweep = bench.ProcessSearchSweep(queries, gt_file);
            if (rank == 0) std::cout.clear();

            // Shard sizes are the same for all nprobe values; gather once.
            DistributedInsertBenchmark::ShardInfo si;
            if (rank == 0) si = bench.GatherShardInfo();
            else           bench.GatherShardInfo();

            if (rank == 0) {
                for (const auto& r : sweep) {
                    const double qps = r.elapsed > 0
                        ? (double)queries.n / r.elapsed : 0.0;
                    std::cout << "  nprobe=" << r.nprobe
                              << "  time="   << r.elapsed << " s"
                              << "  QPS="    << qps;
                    if (r.recall >= 0.0)
                        std::cout << "  recall@" << num_neighbors
                                  << "=" << r.recall;
                    std::cout << "\n";

                    csv << op.step_num << ",SEARCH," << r.nprobe << ","
                        << 0 << "," << queries.n << ","
                        << r.elapsed << "," << qps << ",";
                    if (r.recall >= 0.0) csv << r.recall; else csv << "N/A";
                    for (int r2 = 0; r2 < comm_size; ++r2)
                        csv << "," << si.slots[r2] << "," << si.active[r2];
                    csv << "\n";
                }
                csv.flush();
            }
        }
    }

    if (rank == 0) {
        std::cout << "\nDone.  Results written to " << output_file << "\n";
        csv.close();
    }

    MPI_Finalize();
    return 0;
}
