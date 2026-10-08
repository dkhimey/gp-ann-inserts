#pragma once

#include "defs.h"
#include "points_io.h"
#include "metis_io.h"
#include "hnsw_router.h"
#include "../external/hnswlib/hnswlib/hnswlib.h"

#include <parlay/primitives.h>
#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <numeric>
#include <unordered_set>

// ---------------------------------------------------------------------------
// Generic MPI Alltoallv helper
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
    { size_t tot = 0; for (const auto& b : send_bufs) tot += b.size(); sf.reserve(tot); }
    for (int r = 0; r < comm_size; ++r)
        sf.insert(sf.end(), send_bufs[r].begin(), send_bufs[r].end());

    std::vector<int> rc(comm_size, 0), rd(comm_size, 0);
    MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, comm);
    for (int r = 1; r < comm_size; ++r) rd[r] = rd[r-1] + rc[r-1];

    const size_t total_recv = (size_t)(rd.back() + rc.back());
    std::vector<T> rf(total_recv);
    MPI_Alltoallv(sf.data(), sc.data(), sd.data(), dtype,
                  rf.data(), rc.data(), rd.data(), dtype, comm);

    recv_bufs.assign(comm_size, {});
    for (int r = 0; r < comm_size; ++r)
        recv_bufs[r].assign(rf.begin() + rd[r], rf.begin() + rd[r] + rc[r]);
}

// ---------------------------------------------------------------------------
// DistributedQueryBenchmark
// ---------------------------------------------------------------------------
class DistributedQueryBenchmark {
public:
    int rank      = 0;
    int comm_size = 1;
    int num_shards = 0;
    int dim        = 0;
    int num_neighbors = 10;

    PointSet              shard_points;
    std::vector<uint32_t> shard_point_ids;        // local HNSW index → global point ID
    std::vector<int>      partition;               // global point ID → shard (size = N base points)
    std::vector<int>      routing_index_partition; // routing centroid label → shard

    #ifdef MIPS_DISTANCE
    using SpaceT = hnswlib::InnerProductSpace;
    #else
    using SpaceT = hnswlib::L2Space;
    #endif

    std::unique_ptr<SpaceT>                          space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
    std::unique_ptr<HNSWRouter>                      router;

    HNSWParameters hnsw_parameters;
    int num_voting_neighbors = 250;

    int num_warmup_rounds = 1;
    int num_bench_rounds  = 3;

    DistributedQueryBenchmark() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    }

    // -----------------------------------------------------------------------
    // Load partition from a cluster-format file (output of ./Partition).
    // Each line = one shard; space-separated global point IDs on that line.
    // Trailing empty lines (from trailing newline) are stripped.
    // Builds partition[] : global_point_id → shard_id.
    // -----------------------------------------------------------------------
    void LoadPartition(const std::string& partition_file) {
        Clusters clusters = ReadClusters(partition_file);
        while (!clusters.empty() && clusters.back().empty())
            clusters.pop_back();
        num_shards = (int)clusters.size();

        // Build point → shard map
        uint32_t max_id = 0;
        for (int s = 0; s < num_shards; ++s)
            for (uint32_t id : clusters[s])
                max_id = std::max(max_id, id);

        partition.assign(max_id + 1, -1);
        for (int s = 0; s < num_shards; ++s)
            for (uint32_t id : clusters[s])
                partition[id] = s;

        std::cerr << "[rank " << rank << "] LoadPartition: "
                  << num_shards << " shards, max_point_id=" << max_id << "\n";
    }

    // -----------------------------------------------------------------------
    // Load only the shard-local vectors, converting .u8bin/.i8bin/.fbin
    // to float32. Populates shard_point_ids[local_idx] = global_point_id.
    //
    // Accepts either the full point file (or a sparse copy of it), read by
    // seeking to this rank's points, or a compact shard file written by
    // shard_fanout.py --compact: only this rank's points, in ascending global
    // id order, with header n equal to this rank's point count.
    // -----------------------------------------------------------------------
    void LoadShardPointSet(const std::string& point_set_file) {
        const bool is_float = point_set_file.ends_with(".fbin");
        const bool is_u8    = point_set_file.ends_with(".u8bin");
        const bool is_i8    = point_set_file.ends_with(".i8bin");
        // Default to float for any other extension (e.g. plain .bin)
        const size_t elem_bytes = (is_u8 || is_i8) ? sizeof(uint8_t) : sizeof(float);
        const size_t header_bytes = 2 * sizeof(uint32_t);

        uint32_t n = 0, d = 0;
        std::ifstream in(point_set_file, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open point file: " + point_set_file);
        in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));

        // Reads `count` elements at the current file position into
        // shard_points.coordinates[dst ...], converting to float in bounded chunks.
        auto read_coords = [&](size_t count, size_t dst) {
            constexpr size_t kChunk = size_t(1) << 24;
            std::vector<char> buf;
            for (size_t done = 0; done < count;) {
                const size_t m = std::min(kChunk, count - done);
                float* out = &shard_points.coordinates[dst + done];
                if (is_float) {
                    in.read(reinterpret_cast<char*>(out), m * sizeof(float));
                } else {
                    buf.resize(m);
                    in.read(buf.data(), m);
                    if (is_u8)
                        for (size_t k = 0; k < m; ++k)
                            out[k] = static_cast<float>(static_cast<uint8_t>(buf[k]));
                    else
                        for (size_t k = 0; k < m; ++k)
                            out[k] = static_cast<float>(static_cast<int8_t>(buf[k]));
                }
                if (!in) throw std::runtime_error("Short read from point file: " + point_set_file);
                done += m;
            }
        };

        size_t rank_points = 0;
        for (int s : partition)
            if (s == rank) ++rank_points;

        // With a single shard the full file and a compact file are identical, so
        // treating it as compact is still correct.
        if (n == rank_points) {
            const size_t expected = header_bytes + (size_t)n * d * elem_bytes;
            if (std::filesystem::file_size(point_set_file) != expected)
                throw std::runtime_error("Compact shard file has wrong size: " + point_set_file);

            shard_points.n = n;
            shard_points.d = d;
            shard_points.coordinates.resize((size_t)n * d);
            dim = (int)d;

            shard_point_ids.clear();
            shard_point_ids.reserve(n);
            for (size_t i = 0; i < partition.size(); ++i)
                if (partition[i] == rank) shard_point_ids.push_back((uint32_t)i);

            read_coords((size_t)n * d, 0);
            std::cerr << "[rank " << rank << "] LoadShardPointSet (compact): "
                      << shard_points.n << " points, dim=" << dim << "\n";
            return;
        }

        if ((size_t)n < partition.size())
            throw std::runtime_error(
                "Point file " + point_set_file + " has " + std::to_string(n) +
                " points but the partition covers " + std::to_string(partition.size()) +
                "; a compact shard file here belongs to a different rank or partition");

        // Count how many points belong to this rank
        size_t num_points_in_shard = 0;
        for (uint32_t i = 0; i < n && i < (uint32_t)partition.size(); ++i)
            if (partition[i] == rank) ++num_points_in_shard;

        shard_points.n = num_points_in_shard;
        shard_points.d = d;
        shard_points.coordinates.resize(num_points_in_shard * d);
        dim = (int)d;

        shard_point_ids.clear();
        shard_point_ids.reserve(num_points_in_shard);

        size_t coords_end = 0;
        for (uint32_t point_id = 0; point_id < n; ++point_id) {
            if ((size_t)point_id >= partition.size() || partition[point_id] != rank)
                continue;

            // Scan the contiguous run of points belonging to this rank
            const size_t file_offset =
                header_bytes + (size_t)point_id * d * elem_bytes;
            size_t range_length = 0;
            for (; point_id < n &&
                   (size_t)point_id < partition.size() &&
                   partition[point_id] == rank;
                 ++point_id) {
                shard_point_ids.push_back(point_id);
                ++range_length;
            }
            --point_id; // outer for-loop will increment

            in.seekg((std::streamoff)file_offset);
            read_coords(range_length * d, coords_end);
            coords_end += range_length * d;
        }

        std::cerr << "[rank " << rank << "] LoadShardPointSet: "
                  << shard_points.n << " points, dim=" << dim << "\n";
    }

    // -----------------------------------------------------------------------
    // Per-shard index cache.
    //
    // Building the in-shard HNSW is the dominant setup cost and is fully
    // deterministic given (shard points, M, ef_construction, seed).  We can
    // therefore persist each rank's index to disk after the first run and load
    // it on subsequent runs, skipping both the point-set read and the build.
    //
    // The cache key encodes the partition file basename and HNSW build
    // parameters (M, ef_construction), so a different sharding or changed
    // build parameters yield a different path and are never silently reused.
    //
    // Files are written under ~/extra on each worker node.  Only the partition
    // file's basename is used so fully-qualified paths never create
    // subdirectories under ~/extra.
    // -----------------------------------------------------------------------
    std::string ShardIndexCacheDir() const {
        const char* home = std::getenv("HOME");
        std::filesystem::path dir =
            (home && *home) ? std::filesystem::path(home)
                            : std::filesystem::path(".");
        dir /= "extra";
        return dir.string();
    }

    std::string ShardIndexCachePath(const std::string& base) const {
        const std::string stem =
            std::filesystem::path(base).filename().string();
        std::filesystem::path p = ShardIndexCacheDir();
        p /= stem
             + ".shard" + std::to_string(rank)
             + ".M"     + std::to_string(hnsw_parameters.M)
             + ".efc"   + std::to_string(hnsw_parameters.ef_construction)
             + ".hnsw";
        return p.string();
    }

    // Read just the (n, dim) header from a points file so that `dim` is set
    // without loading all shard points.  Used on a cache hit.
    void ReadDim(const std::string& point_set_file) {
        std::ifstream in(point_set_file, std::ios::binary);
        if (!in) throw std::runtime_error(
            "ReadDim: cannot open '" + point_set_file + "'");
        uint32_t n = 0, d = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        dim = (int)d;
    }

    // Load a previously serialized shard index.  Returns false if the cache
    // file does not exist (caller should then build).  `dim` must be set first
    // (see ReadDim).  shard_point_ids / shard_points are intentionally left
    // empty: HNSW labels are already global point IDs.
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

    // -----------------------------------------------------------------------
    // Build an in-shard HNSW index.
    // IMPORTANT: labels are global point IDs (from shard_point_ids), NOT
    // local sequential indices.  This makes recall checking correct: the IDs
    // returned by searchKnn can be compared directly to ground-truth IDs.
    //
    // If cache_path is non-empty the freshly built index is saved there so
    // future runs can skip the build via LoadInShardIndex.
    // -----------------------------------------------------------------------
    void BuildInShardIndex(const std::string& cache_path = "") {
        space = std::make_unique<SpaceT>(dim);
        // +1: hnswlib reserves element 0 internally; capacity must exceed n
        hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space.get(),
            shard_points.n + 1,
            hnsw_parameters.M,
            hnsw_parameters.ef_construction,
            /* random seed = */ 555);

        parlay::parallel_for(0, shard_points.n, [&](size_t i) {
            hnsw->addPoint(shard_points.GetPoint(i), shard_point_ids[i]);
        });
        hnsw->setEf(hnsw_parameters.ef_search);

        std::cerr << "[rank " << rank << "] HNSW built: "
                  << hnsw->cur_element_count << " elements, dim=" << dim << "\n";

        if (!cache_path.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(
                std::filesystem::path(cache_path).parent_path(), ec);
            hnsw->saveIndex(cache_path);
            std::cerr << "[rank " << rank << "] HNSW saved to cache: "
                      << cache_path << "\n";
        }
        shard_points.Drop();
    }

    // -----------------------------------------------------------------------
    // Load the HNSW router.  The router's HNSW labels are routing-centroid
    // IDs; their shard assignments live in the companion
    // <router_file>.routing_index_partition file, NOT in the base partition.
    // -----------------------------------------------------------------------
    void LoadRouter(const std::string& hnsw_router_file) {
        routing_index_partition =
            ReadMetisPartition(hnsw_router_file + ".routing_index_partition");
        router = std::make_unique<HNSWRouter>(
            hnsw_router_file, dim, routing_index_partition);
    }

    // -----------------------------------------------------------------------
    // One fan-out → search → fan-in pass.
    //
    // send_qids[s]  : query IDs to send to shard s
    // send_qvecs[s] : concatenated query vectors for shard s (len = nids*dim)
    // nq            : total number of queries (for sizing the neighbors array)
    // want_neighbors: if true, return merged, sorted, deduped result list
    //
    // Returns { max_elapsed_across_all_ranks, merged_neighbors }.
    // merged_neighbors is indexed by qid in [0,nq); empty if !want_neighbors.
    // -----------------------------------------------------------------------
    std::pair<double, std::vector<NNVec>>
    RunOnePass(const std::vector<std::vector<uint32_t>>& send_qids,
               const std::vector<std::vector<float>>&    send_qvecs,
               size_t nq,
               bool   want_neighbors)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();

        // ---- Phase 1: fan queries out to shards ----
        std::vector<std::vector<uint32_t>> recv_qids;
        std::vector<std::vector<float>>    recv_qvecs;
        AllToAllV(send_qids,  recv_qids,  MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
        AllToAllV(send_qvecs, recv_qvecs, MPI_FLOAT,    comm_size, MPI_COMM_WORLD);

        // Build a flat task list and search in parallel
        struct QTask { int src; size_t j; uint32_t qid; };
        std::vector<QTask> qtasks;
        for (int src = 0; src < comm_size; ++src)
            for (size_t j = 0; j < recv_qids[src].size(); ++j)
                qtasks.push_back({src, j, recv_qids[src][j]});

        using KNNResult = std::vector<std::pair<float, hnswlib::labeltype>>;
        std::vector<KNNResult> qresults(qtasks.size());
        parlay::parallel_for(0, qtasks.size(), [&](size_t ti) {
            const auto& qt = qtasks[ti];
            float* Q = recv_qvecs[qt.src].data() + qt.j * (size_t)dim;
            auto pq = hnsw->searchKnn(Q, num_neighbors);
            auto& res = qresults[ti];
            res.reserve(num_neighbors);
            while (!pq.empty()) { res.push_back(pq.top()); pq.pop(); }
            // Pad to num_neighbors so every task has the same result count
            while ((int)res.size() < num_neighbors)
                res.push_back({std::numeric_limits<float>::max(),
                               (hnswlib::labeltype)0});
        });

        // Pack results for return to query owners
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

        // ---- Phase 2: return results to query owners ----
        std::vector<std::vector<uint32_t>> rcv_rqids, rcv_rids;
        std::vector<std::vector<float>>    rcv_rdists;
        AllToAllV(snd_rqids,  rcv_rqids,  MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
        AllToAllV(snd_rids,   rcv_rids,   MPI_UINT32_T, comm_size, MPI_COMM_WORLD);
        AllToAllV(snd_rdists, rcv_rdists, MPI_FLOAT,    comm_size, MPI_COMM_WORLD);

        const double elapsed = MPI_Wtime() - t0;
        double max_t = 0.0;
        MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Merge results if requested
        std::vector<NNVec> neighbors;
        if (want_neighbors) {
            neighbors.resize(nq);
            for (int src = 0; src < comm_size; ++src) {
                const size_t nres = rcv_rqids[src].size();
                for (size_t j = 0; j < nres; ++j) {
                    const uint32_t qid = rcv_rqids[src][j];
                    if (qid >= (uint32_t)nq) continue;
                    for (int nn = 0; nn < num_neighbors; ++nn)
                        neighbors[qid].push_back({
                            rcv_rdists[src][j * num_neighbors + nn],
                            rcv_rids[src][j * num_neighbors + nn]});
                }
            }
            // Sort ascending by distance, dedup by ID, trim to k
            for (size_t q = 0; q < nq; ++q) {
                auto& nv = neighbors[q];
                if (nv.empty()) continue;
                std::sort(nv.begin(), nv.end());
                nv.erase(std::unique(nv.begin(), nv.end(),
                    [](const NNVec::value_type& a, const NNVec::value_type& b){
                        return a.second == b.second;
                    }), nv.end());
                if ((int)nv.size() > num_neighbors)
                    nv.resize(num_neighbors);
            }
        }

        return {max_t, std::move(neighbors)};
    }

    // -----------------------------------------------------------------------
    // Sweep nprobe from 1 to num_shards.
    //
    // Routing is computed once per query, outside all timed sections.
    // For each nprobe:
    //   - num_warmup_rounds warmup passes; recall is computed on the last one.
    //   - num_bench_rounds  timed passes with no recall computation.
    // Recall uses ID-based matching (gt IDs from gt_file), so it works
    // correctly for both .bin (distances+IDs) and .ibin (IDs only) GT files.
    //
    // Prints one line per nprobe on rank 0.
    // -----------------------------------------------------------------------
    void ProcessSearchSweep(PointSet& queries, const std::string& gt_file) {
        const size_t nq = queries.n;

        // Partition the query set across ranks
        const size_t chunk = (nq + comm_size - 1) / comm_size;
        const size_t qs    = std::min((size_t)rank * chunk, nq);
        const size_t qe    = std::min(qs + chunk, nq);

        // Pre-compute routing order for every query in this rank's slice.
        // Done once outside every timed window; gives the sorted-shard order
        // that all nprobe values index into.
        std::vector<std::vector<int>> routing_order(qe - qs);
        for (size_t q = qs; q < qe; ++q) {
            routing_order[q - qs] =
                router->Query(queries.GetPoint(q), num_voting_neighbors)
                      .RoutingQuery();
        }

        // Load GT and build per-query data structures (outside timed window).
        // ReadGroundTruth reads IDs correctly from both .bin and .ibin files;
        // distances are ignored to avoid the zeroed-distance .ibin bug.
        bool has_gt = !gt_file.empty() && std::filesystem::exists(gt_file);

        // gt_ids[q] = sorted vector of top-K GT neighbor IDs.
        // Sorted so that std::binary_search works for actual-recall checking.
        // Every rank loads all queries' GT so it can check its own slice [qs,qe).
        std::vector<std::vector<uint32_t>> gt_ids(nq);
        if (has_gt) {
            auto gt = ReadGroundTruth(gt_file);
            const int kk = std::min((int)(gt.empty() ? 0 : gt[0].size()),
                                    num_neighbors);
            for (size_t q = 0; q < nq && q < gt.size(); ++q) {
                gt_ids[q].reserve(kk);
                for (int j = 0; j < kk; ++j)
                    gt_ids[q].push_back((uint32_t)gt[q][j].second);
                std::sort(gt_ids[q].begin(), gt_ids[q].end());
                gt_ids[q].erase(
                    std::unique(gt_ids[q].begin(), gt_ids[q].end()),
                    gt_ids[q].end());
            }
        }

        // Theoretical (routing) recall — computed once from routing_order.
        //
        // For each query in this rank's slice and each of its GT neighbors,
        // find the first probe position at which that neighbor's shard appears
        // in routing_order.  Bucket the hit by that position, then prefix-sum
        // to get cumulative hit counts at every nprobe.  After MPI_Reduce,
        // rank 0 holds theoretical_recall_at[nprobe] = fraction of GT neighbors
        // reachable when probing the first nprobe shards.
        //
        // This is a strict upper bound on empirical recall: if a GT neighbor's
        // shard is not visited, the in-shard HNSW cannot return it.
        // Denominator matches the empirical denominator (nq * num_neighbors).
        std::vector<double> theoretical_recall_at(num_shards + 1, -1.0);
        if (has_gt) {
            std::vector<uint64_t> local_add(num_shards, 0);
            for (size_t q = qs; q < qe; ++q) {
                const auto& order = routing_order[q - qs];
                // pos[s] = first probe index at which shard s appears,
                //          or num_shards if s never appears.
                std::vector<int> pos(num_shards, num_shards);
                for (int p = 0; p < (int)order.size(); ++p) {
                    const int s = order[p];
                    if (s >= 0 && s < num_shards && pos[s] == num_shards)
                        pos[s] = p;
                }
                for (uint32_t gid : gt_ids[q]) {
                    if ((size_t)gid >= partition.size()) continue;
                    const int s = partition[gid];
                    if (s < 0 || s >= num_shards) continue;
                    const int p = pos[s];
                    if (p < num_shards) local_add[p]++;
                }
            }
            std::vector<uint64_t> global_add(num_shards, 0);
            MPI_Reduce(local_add.data(), global_add.data(), num_shards,
                       MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                const double denom = (double)nq * num_neighbors;
                uint64_t cum = 0;
                for (int np = 1; np <= num_shards; ++np) {
                    cum += global_add[np - 1];
                    theoretical_recall_at[np] =
                        denom > 0 ? (double)cum / denom : 0.0;
                }
            }
        }

        if (rank == 0) {
            std::cerr << "[rank 0] ProcessSearchSweep:"
                      << " nq=" << nq
                      << " num_shards=" << num_shards
                      << " HNSW_elements="
                      << (hnsw ? hnsw->cur_element_count.load() : 0)
                      << " has_gt=" << has_gt
                      << " num_voting_neighbors=" << num_voting_neighbors
                      << "\n";
            std::printf("%-8s  %10s  %10s  %10s\n",
                        "nprobe", "QPS", "recall@K", "theo_recall");
            std::printf("%-8s  %10s  %10s  %10s\n",
                        "------", "---", "--------", "-----------");
            std::fflush(stdout);
        }

        for (int nprobe = 1; nprobe <= num_shards; ++nprobe) {

            // Build per-shard send buffers for this nprobe value.
            // These are reused across warmup and benchmark rounds.
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

            // ---- Warmup rounds (recall computed on the last one) ----
            double recall = -1.0;
            for (int r = 0; r < num_warmup_rounds; ++r) {
                const bool last = (r == num_warmup_rounds - 1);
                auto [unused_t, warmup_neighbors] =
                    RunOnePass(send_qids, send_qvecs, nq, last && has_gt);

                if (last && has_gt) {
                    // Each rank counts hits for its own query slice.
                    // gt_ids[q] is sorted, so binary_search is O(log k).
                    uint64_t my_hits = 0;
                    for (size_t q = qs; q < qe; ++q) {
                        for (const auto& [dist, id] : warmup_neighbors[q])
                            if (std::binary_search(gt_ids[q].begin(),
                                                   gt_ids[q].end(), id))
                                ++my_hits;
                    }
                    uint64_t total_hits = 0;
                    MPI_Reduce(&my_hits, &total_hits, 1, MPI_UINT64_T,
                               MPI_SUM, 0, MPI_COMM_WORLD);
                    if (rank == 0)
                        recall = (double)total_hits /
                                 ((double)nq * num_neighbors);
                }
            }

            // ---- Timed benchmark rounds (no recall computation) ----
            double min_t = std::numeric_limits<double>::max();
            for (int r = 0; r < num_bench_rounds; ++r) {
                auto [max_t, unused_n] =
                    RunOnePass(send_qids, send_qvecs, nq, false);
                if (rank == 0)
                    min_t = std::min(min_t, max_t);
            }

            if (rank == 0) {
                const double qps =
                    (min_t > 0.0) ? (double)nq / min_t : -1.0;
                if (has_gt)
                    std::printf(
                        "nprobe=%-4d  QPS=%10.1f  recall@%d=%.4f  theo=%.4f\n",
                        nprobe, qps, num_neighbors,
                        recall, theoretical_recall_at[nprobe]);
                else
                    std::printf("nprobe=%-4d  QPS=%10.1f  (no GT)\n",
                                nprobe, qps);
                std::fflush(stdout);
            }
        }
    }
};
