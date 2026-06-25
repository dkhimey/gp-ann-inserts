#pragma once

#include "points_io.h"
#include "metis_io.h"
#include "hnsw_router.h"
#include "../external/hnswlib/hnswlib/hnswlib.h"

#include <mpi.h>
#include <parlay/primitives.h>

#include <cstdlib>
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

    PointSet               shard_points;
    std::vector<uint32_t>  shard_point_ids;    // local index → global point ID
    std::vector<int>       partition;               // base-point → shard mapping
    std::vector<int>       routing_index_partition; // centroid → shard mapping (owned, keeps HNSWRouter ref alive)

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

    // Benchmarking rounds.
    // One warm-up pass is run before timing to bring HNSW graph data into
    // cache and let MPI settle.  Then num_bench_rounds timed passes are run
    // and the minimum elapsed time is reported (minimum = least OS noise,
    // standard practice for throughput benchmarks).
    int num_warmup_rounds = 1;
    int num_bench_rounds  = 10;

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
        // ReadClusters adds an empty entry for each trailing newline in the
        // file.  Strip those so num_shards matches the actual shard count.
        while (!clusters.empty() && clusters.back().empty())
            clusters.pop_back();
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

        // Determine the on-disk element size from the file extension.
        // .fbin  → float32  (elem_bytes = 4, no conversion needed)
        // .u8bin → uint8    (elem_bytes = 1, cast to float on read)
        // .i8bin → int8     (elem_bytes = 1, cast to float on read)
        const bool is_float =
            point_set_file.size() >= 5 &&
            point_set_file.substr(point_set_file.size() - 5) == ".fbin";
        const bool is_u8 =
            point_set_file.size() >= 6 &&
            point_set_file.substr(point_set_file.size() - 6) == ".u8bin";
        const bool is_i8 =
            point_set_file.size() >= 6 &&
            point_set_file.substr(point_set_file.size() - 6) == ".i8bin";
        if (!is_float && !is_u8 && !is_i8)
            throw std::runtime_error(
                "LoadShardPointSet: unsupported format for '" + point_set_file +
                "'. Use .fbin, .u8bin, or .i8bin.");
        const size_t elem_bytes = is_float ? sizeof(float) : sizeof(uint8_t);

        uint32_t n = 0, d = 0;
        size_t header_bytes = 0;

        std::ifstream in(point_set_file, std::ios::binary);
        in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
        header_bytes += sizeof(uint32_t);
        in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        header_bytes += sizeof(uint32_t);

        shard_points.n = num_points_in_shard;
        shard_points.d = d;
        shard_points.coordinates.resize(shard_points.n * shard_points.d);
        dim = (int)d;

        // shard_point_ids maps local HNSW label → global point ID so that
        // neighbor IDs returned by HNSW searches match the ground-truth file.
        shard_point_ids.clear();
        shard_point_ids.reserve(num_points_in_shard);

        size_t coords_end = 0;
        for (uint32_t point_id = 0; point_id < n; ++point_id) {
            if (partition[point_id] == rank) {
                // Seek using the on-disk element size, not sizeof(float).
                const size_t file_offset =
                    header_bytes + (size_t)point_id * d * elem_bytes;
                size_t range_length = 0;
                for ( ; point_id < n && partition[point_id] == rank; ++point_id) {
                    shard_point_ids.push_back(point_id);
                    ++range_length;
                }
                in.seekg((std::streamoff)file_offset);

                if (is_float) {
                    // Float32: read directly into the coordinates vector.
                    in.read(reinterpret_cast<char*>(
                                &shard_points.coordinates[coords_end]),
                            (std::streamsize)(range_length * d * sizeof(float)));
                } else if (is_u8) {
                    // Uint8: read into a temporary buffer and cast to float.
                    std::vector<uint8_t> buf(range_length * d);
                    in.read(reinterpret_cast<char*>(buf.data()),
                            (std::streamsize)(range_length * d));
                    for (size_t k = 0; k < buf.size(); ++k)
                        shard_points.coordinates[coords_end + k] =
                            static_cast<float>(buf[k]);
                } else { // is_i8
                    std::vector<int8_t> buf(range_length * d);
                    in.read(reinterpret_cast<char*>(buf.data()),
                            (std::streamsize)(range_length * d));
                    for (size_t k = 0; k < buf.size(); ++k)
                        shard_points.coordinates[coords_end + k] =
                            static_cast<float>(buf[k]);
                }
                coords_end += range_length * d;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Per-shard index cache.
    //
    // Building the in-shard HNSW is the dominant setup cost and is fully
    // deterministic given (shard points, M, ef_construction, seed).  We can
    // therefore persist each rank's index to disk after the first run and load
    // it on subsequent runs, skipping both the point-set read and the build.
    //
    // The cache key encodes the partition file (the dataset/sharding identity)
    // and the HNSW build parameters, so a different partitioning or a changed M
    // / ef_construction yields a different path and is never silently reused.
    //
    // Files are written under ~/extra on each worker (resolved from $HOME, so
    // every rank uses its own local directory).  Only the partition file's
    // basename is used in the name so a fully-qualified partition path does not
    // introduce sub-directories under ~/extra.
    // -----------------------------------------------------------------------
    std::string ShardIndexCacheDir() const {
        const char* home = std::getenv("HOME");
        std::filesystem::path dir =
            (home && *home) ? std::filesystem::path(home) : std::filesystem::path(".");
        dir /= "extra";
        return dir.string();
    }

    std::string ShardIndexCachePath(const std::string& base) const {
        const std::string stem = std::filesystem::path(base).filename().string();
        std::filesystem::path p = ShardIndexCacheDir();
        p /= stem + ".shard" + std::to_string(rank)
             + ".M" + std::to_string(hnsw_parameters.M)
             + ".efc" + std::to_string(hnsw_parameters.ef_construction)
             + ".hnsw";
        return p.string();
    }

    // Read just the (n, dim) header from a points file.  Used on a cache hit so
    // that `dim` is set (needed by the router and searches) without reading the
    // full shard point set.
    void ReadDim(const std::string& point_set_file) {
        std::ifstream in(point_set_file, std::ios::binary);
        if (!in) throw std::runtime_error(
            "ReadDim: cannot open '" + point_set_file + "'");
        uint32_t n = 0, d = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        dim = (int)d;
    }

    // Load a previously serialized shard index.  Returns false if no cache file
    // exists at `cache_path` (caller should then build).  `dim` must be set
    // first (see ReadDim).  shard_point_ids / shard_points are intentionally
    // left empty: HNSW labels are the global point IDs, so nothing else is
    // needed for searching after a load.
    bool LoadInShardIndex(const std::string& cache_path) {
        if (!std::filesystem::exists(cache_path)) return false;
        space = std::make_unique<SpaceT>(dim);
        hnsw  = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space.get(), cache_path);
        hnsw->setEf(hnsw_parameters.ef_search);
        std::cerr << "[rank " << rank << "] HNSW loaded from cache: "
                  << hnsw->cur_element_count << " elements, dim=" << dim
                  << " (" << cache_path << ")\n";
        return true;
    }

    // Build the in-shard HNSW from the loaded shard point set.  When
    // `cache_path` is non-empty the freshly built index is serialized there so
    // future runs can skip the build via LoadInShardIndex.
    void BuildInShardIndex(const std::string& cache_path = "") {
        space = std::make_unique<SpaceT>(dim);
        // Add 1 to capacity: hnswlib can have off-by-one issues when
        // max_elements == cur_element_count during the last insertion.
        hnsw  = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space.get(), shard_points.n + 1,
            hnsw_parameters.M, hnsw_parameters.ef_construction,
            /* random seed = */ 555);
        parlay::parallel_for(0, shard_points.n, [&](size_t i) {
            // Use the global point ID as the HNSW label so that search results
            // can be directly compared against the ground-truth file.
            hnsw->addPoint(shard_points.GetPoint(i), shard_point_ids[i]);
        });
        hnsw->setEf(hnsw_parameters.ef_search);
        std::cerr << "[rank " << rank << "] HNSW built: "
                  << hnsw->cur_element_count << " elements, dim=" << dim << "\n";
        if (!cache_path.empty()) {
            // Make sure the destination directory (e.g. ~/extra) exists.
            std::error_code ec;
            std::filesystem::create_directories(
                std::filesystem::path(cache_path).parent_path(), ec);
            hnsw->saveIndex(cache_path);
            std::cerr << "[rank " << rank << "] HNSW saved to cache: "
                      << cache_path << "\n";
        }
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
        double min_elapsed;       // min across bench rounds (max across ranks each round)
        double recall;            // empirical recall@num_neighbors, or -1.0
        double theoretical_recall; // point-level routing (oracle in-shard) recall, or -1.0
    };

    // -----------------------------------------------------------------------
    // Sweep nprobe from 1 to num_shards.  For each value:
    //   - num_warmup_rounds full pipeline passes are run untimed (cache warm-up,
    //     MPI library warm-up).  Recall is computed from the last warm-up round
    //     so that it is never included in the measured time.
    //   - num_bench_rounds timed passes are run; the minimum elapsed time
    //     (max-across-ranks per round, then min across rounds) is reported.
    //     Using the minimum follows standard throughput-benchmark practice:
    //     it reflects peak hardware performance with the least OS/scheduler noise.
    //
    // Routing is pre-computed once for all rounds (fixed shard ordering).
    // num_voting_neighbors must be >= num_shards so every shard gets a real
    // distance estimate across the full sweep.
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
        // Load GT: build per-query sorted ID sets for recall@k.
        //
        // ID-based recall is used (not distance-based) because some ground
        // truth files (e.g. big-ann-benchmarks .ibin) store only neighbor IDs
        // with no distances.  ReadGroundTruth reads IDs correctly from both
        // formats; distances from ID-only files would be zero/garbage and
        // produce recall = 0 when used as thresholds.
        //
        // gt_ids[q] holds the sorted top-num_neighbors true neighbor IDs for
        // query q.  Sorted so membership can be checked with binary search.
        // ------------------------------------------------------------------
        const bool has_gt = !gt_file.empty() &&
                            std::filesystem::exists(gt_file);
        std::vector<std::vector<uint32_t>> gt_ids(nq);
        if (has_gt) {
            auto gt = ReadGroundTruth(gt_file);
            for (size_t q = 0; q < nq; ++q) {
                auto& ids = gt_ids[q];
                const int k = std::min<int>(num_neighbors, (int)gt[q].size());
                ids.reserve(k);
                for (int j = 0; j < k; ++j)
                    ids.push_back(gt[q][j].second);
                std::sort(ids.begin(), ids.end());
            }
        }

        // ------------------------------------------------------------------
        // Point-level theoretical (routing) recall.
        //
        // This is the recall achievable if in-shard search were perfect: for
        // each query we take the router's ranked shard order and, at nprobe=n,
        // count how many of the query's true k neighbors live in one of the
        // top-n probed shards. It is an upper bound on the empirical recall
        // computed below, which additionally pays for approximate in-shard
        // HNSW search; the gap (theoretical - empirical) is the in-shard loss.
        //
        // Computed once, independently of the timed pipeline, and shares the
        // exact same routing order (router->Query(...).RoutingQuery()) used by
        // run_one_pass so the comparison is apples-to-apples. Each rank handles
        // its own query slice [qs,qe); a neighbor counts as found at nprobe n
        // iff its owning shard partition[neighbor_id] appears within the first
        // n entries of the probe order. We bucket each (query,neighbor) pair by
        // the probe position at which its shard first appears, reduce across
        // ranks, then prefix-sum to get cumulative hits at every nprobe.
        //
        // Denominator is nq * num_neighbors, identical to the empirical recall
        // above, so the two curves are directly comparable.
        // ------------------------------------------------------------------
        std::vector<double> theoretical_recall(num_shards + 1, -1.0);
        if (has_gt) {
            const size_t num_workers = parlay::num_workers();
            // add_at[w][p] = # of (query,neighbor) pairs whose owning shard
            // first appears at probe position p (0-indexed) in the order.
            std::vector<std::vector<uint64_t>> add_at(
                num_workers, std::vector<uint64_t>(num_shards, 0));

            parlay::parallel_for(qs, qe, [&](size_t q) {
                const size_t w = parlay::worker_id();
                float* Q = queries.GetPoint(q);
                auto order = router->Query(Q, num_voting_neighbors).RoutingQuery();

                // pos[s] = first probe position of shard s, or num_shards if absent.
                std::vector<int> pos(num_shards, num_shards);
                const int lim = std::min<int>((int)order.size(), num_shards);
                for (int p = 0; p < lim; ++p) {
                    const int s = order[p];
                    if (s >= 0 && s < num_shards && pos[s] == num_shards) pos[s] = p;
                }

                for (uint32_t gid : gt_ids[q]) {
                    if (gid >= partition.size()) continue;
                    const int s = partition[gid];
                    if (s < 0 || s >= num_shards) continue;   // unpartitioned neighbor
                    const int p = pos[s];
                    if (p < num_shards) add_at[w][p] += 1;    // found once shard s is probed
                }
            });

            // Merge per-worker buckets, then sum across ranks (disjoint slices).
            std::vector<uint64_t> local_add(num_shards, 0);
            for (size_t w = 0; w < num_workers; ++w)
                for (int p = 0; p < num_shards; ++p)
                    local_add[p] += add_at[w][p];

            std::vector<uint64_t> global_add(num_shards, 0);
            MPI_Reduce(local_add.data(), global_add.data(), num_shards,
                       MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                const double denom = (double)nq * num_neighbors;
                uint64_t cum = 0;
                for (int nprobe = 1; nprobe <= num_shards; ++nprobe) {
                    cum += global_add[nprobe - 1];   // neighbors found once nprobe shards probed
                    theoretical_recall[nprobe] = denom > 0 ? (double)cum / denom : 0.0;
                }
            }
        }

        // ------------------------------------------------------------------
        // Helper: run one complete route/fan-out/search/fan-in pass.
        // Routing is performed inside the timed region so that the measured
        // time matches SURGE's shared_static_experiment timing model.
        // neighbors is populated only when want_neighbors=true (skip on
        // timed rounds to save merge cost).
        // ------------------------------------------------------------------
        auto run_one_pass = [&](int nprobe, bool want_neighbors)
            -> std::pair<double, std::vector<NNVec>>
        {
            MPI_Barrier(MPI_COMM_WORLD);
            const double t0 = MPI_Wtime();

            // Routing — inside the timed region, parallelised over queries.
            // Each worker fills its own local buffers to avoid contention on
            // the shared send_qids/send_qvecs, then a sequential merge follows.
            const size_t num_workers = parlay::num_workers();
            std::vector<std::vector<std::vector<uint32_t>>> local_qids(
                num_workers, std::vector<std::vector<uint32_t>>(comm_size));
            std::vector<std::vector<std::vector<float>>> local_qvecs(
                num_workers, std::vector<std::vector<float>>(comm_size));

            parlay::parallel_for(qs, qe, [&](size_t q) {
                const size_t w = parlay::worker_id();
                float* Q = queries.GetPoint(q);
                auto order = router->Query(Q, num_voting_neighbors).RoutingQuery();
                const int np = std::min(nprobe, (int)order.size());
                for (int pi = 0; pi < np; ++pi) {
                    const int s = order[pi];
                    local_qids[w][s].push_back((uint32_t)q);
                    local_qvecs[w][s].insert(local_qvecs[w][s].end(), Q, Q + dim);
                }
            });

            std::vector<std::vector<uint32_t>> send_qids(comm_size);
            std::vector<std::vector<float>>    send_qvecs(comm_size);
            for (size_t w = 0; w < num_workers; ++w) {
                for (int s = 0; s < comm_size; ++s) {
                    send_qids[s].insert(send_qids[s].end(),
                                        local_qids[w][s].begin(), local_qids[w][s].end());
                    send_qvecs[s].insert(send_qvecs[s].end(),
                                         local_qvecs[w][s].begin(), local_qvecs[w][s].end());
                }
            }

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
                // Pad so the fan-in stride is always exactly num_neighbors.
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

            // Phase 2: fan results back to query owners.
            std::vector<std::vector<uint32_t>> rcv_rqids, rcv_rids;
            std::vector<std::vector<float>>    rcv_rdists;
            AllToAllV(snd_rqids,  rcv_rqids,  MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
            AllToAllV(snd_rids,   rcv_rids,   MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
            AllToAllV(snd_rdists, rcv_rdists, MPI_FLOAT,    comm_size, MPI_COMM_WORLD);

            const double elapsed = MPI_Wtime() - t0;
            double max_t = 0.0;
            MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            // Merge results only when the caller needs them (measurement rounds).
            std::vector<NNVec> neighbors;
            if (want_neighbors) {
                neighbors.resize(nq);
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
                for (size_t q = qs; q < qe; ++q) {
                    auto& nv = neighbors[q];
                    std::sort(nv.begin(), nv.end());
                    nv.erase(std::unique(nv.begin(), nv.end(),
                        [](const auto& a, const auto& b) {
                            return a.second == b.second; }),
                        nv.end());
                    if ((int)nv.size() > num_neighbors) nv.resize(num_neighbors);
                }
            }

            return {max_t, std::move(neighbors)};
        };

        // ------------------------------------------------------------------
        // Sweep nprobe = 1 .. num_shards.
        // ------------------------------------------------------------------
        std::vector<SearchResult> results;
        results.reserve(num_shards);

        for (int nprobe = 1; nprobe <= num_shards; ++nprobe) {

            // Warm-up rounds: run the full pipeline untimed.
            // The last warm-up round retrieves neighbors so recall can be
            // computed here — before any timed measurement begins.
            double recall = -1.0;
            for (int r = 0; r < num_warmup_rounds; ++r) {
                const bool last = (r == num_warmup_rounds - 1);
                auto [unused_t, warmup_neighbors] =
                    run_one_pass(nprobe, /*want_neighbors=*/last && has_gt);

                if (last && has_gt) {
                    // Compute recall@num_neighbors from warm-up results.
                    // A returned neighbor counts as a hit if its ID appears in
                    // the sorted ground-truth top-k set (binary search).
                    uint64_t my_hits = 0;
                    for (size_t q = qs; q < qe; ++q) {
                        const auto& ids = gt_ids[q];
                        for (const auto& [dist, id] : warmup_neighbors[q])
                            if (std::binary_search(ids.begin(), ids.end(), id))
                                ++my_hits;
                    }
                    uint64_t total_hits = 0;
                    MPI_Reduce(&my_hits, &total_hits, 1, MPI_UINT64_T,
                               MPI_SUM, 0, MPI_COMM_WORLD);
                    if (rank == 0)
                        recall = (double)total_hits / ((double)nq * num_neighbors);
                }
            }

            // Measurement rounds: timed, no neighbor merge needed.
            double min_t = std::numeric_limits<double>::max();
            for (int r = 0; r < num_bench_rounds; ++r) {
                auto [max_t, unused_n] =
                    run_one_pass(nprobe, /*want_neighbors=*/false);
                min_t = std::min(min_t, max_t);
            }

            results.push_back({nprobe, min_t, recall, theoretical_recall[nprobe]});
        }

        return results;
    }
};
