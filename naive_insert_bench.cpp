// naive_insert_bench.cpp
//
// Drives a GP-ANN index through an operation runbook with three operation types:
//
//   insert [start, end)  — first occurrence: builds the full index from scratch
//                          on vectors base[start, end).
//                          subsequent occurrences: naive insert (route each vector
//                          to the nearest shard via the frozen HNSW router, then
//                          call addPoint on that shard's local graph).
//   delete [start, end)  — shadow delete via hnswlib markDelete; elements are
//                          removed from search results but graph edges are kept.
//   search               — run all queries and report recall against the GT file
//                          for this search step.
//
// Runbook format (YAML, NeurIPS-2023 competition schema):
//
//   <dataset-name>:
//     max_pts: <N>
//     1:
//       operation: 'insert'
//       start: <S>
//       end: <E>          # exclusive
//     2:
//       operation: 'search'
//     3:
//       operation: 'delete'
//       start: <S>
//       end: <E>
//     ...
//
// Ground-truth files are named  <gt-prefix>/step<i>.gt100  where <i> is the
// 0-based index of the search operation in runbook order.  Each GT file must
// reflect the live index state immediately before that search (accounting for
// preceding inserts and deletes).
//
// Usage
// -----
//   ./NaiveInsertBench <base-vectors> <queries> <partition-file>
//                      <gt-prefix> <runbook-yaml>
//                      <num-neighbors> [num-probes-query]
//
//   partition-file    cluster file covering exactly the vectors in the FIRST
//                     insert batch (same IDs, i.e. base[start, end) of op 1)
//   num-probes-query  shards probed per query (default 1)

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <parlay/primitives.h>

#include "defs.h"
#include "hnsw_router.h"
#include "inverted_index_hnsw.h"
#include "kmeans_tree_router.h"
#include "metis_io.h"
#include "points_io.h"
#include "recall.h"


// ---------------------------------------------------------------------------
// Runbook parsing
// ---------------------------------------------------------------------------

struct Operation {
    enum class Type { INSERT, DELETE, SEARCH };
    Type     type     = Type::SEARCH;
    uint32_t start    = 0;
    uint32_t end      = 0;  // exclusive
    uint32_t step_num = 0;  // the numeric YAML key for this op (used in GT filenames)
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

static bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() &&
           s.compare(0, prefix.size(), prefix) == 0;
}

static Runbook ParseRunbook(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open runbook: " + path);

    Runbook rb;
    Operation cur;
    bool in_op = false;

    std::string line;
    while (std::getline(f, line)) {
        // Top-level key (0 leading spaces, ends with ':'): dataset name
        if (!line.empty() && line[0] != ' ' && line.back() == ':') {
            rb.dataset_name = line.substr(0, line.size() - 1);
            continue;
        }

        // "  max_pts: N"
        if (starts_with(line, "  max_pts:")) {
            std::istringstream ss(line.substr(line.find(':') + 1));
            ss >> rb.max_pts;
            continue;
        }

        // Skip non-operation lines at the 2-space indent (gt_url, etc.)
        if (line.size() > 2 && line[0] == ' ' && line[1] == ' ' && line[2] != ' ') {
            size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::string key = trim(line.substr(0, colon));
            bool is_number = !key.empty() &&
                             std::all_of(key.begin(), key.end(), ::isdigit);
            if (!is_number) continue;   // non-numeric key: skip

            // Numeric key → new operation; commit the previous one
            if (in_op) rb.ops.push_back(cur);
            cur          = Operation{};
            cur.step_num = static_cast<uint32_t>(std::stoul(key));
            in_op        = true;
            continue;
        }

        if (!in_op) continue;

        // 4-space indent: operation fields
        if (starts_with(line, "    operation:")) {
            std::string v = trim(line.substr(line.find(':') + 1));
            v.erase(std::remove(v.begin(), v.end(), '\''), v.end());
            v.erase(std::remove(v.begin(), v.end(), '"'),  v.end());
            if      (v == "insert") cur.type = Operation::Type::INSERT;
            else if (v == "delete") cur.type = Operation::Type::DELETE;
            else if (v == "search") cur.type = Operation::Type::SEARCH;
            else throw std::runtime_error("Unknown operation type: " + v);

        } else if (starts_with(line, "    start:")) {
            std::istringstream ss(line.substr(line.find(':') + 1));
            ss >> cur.start;

        } else if (starts_with(line, "    end:")) {
            std::istringstream ss(line.substr(line.find(':') + 1));
            ss >> cur.end;
        }
    }
    if (in_op) rb.ops.push_back(cur);

    return rb;
}


// ---------------------------------------------------------------------------
// Query helper
// ---------------------------------------------------------------------------

static std::vector<NNVec> RunQueries(
        PointSet&          queries,
        InvertedIndexHNSW& ivf,
        HNSWRouter&        router,
        int                num_neighbors,
        int                num_probes,
        int                num_voting_neighbors) {

    std::vector<NNVec> neighbors(queries.n);
    for (size_t q = 0; q < queries.n; ++q) {
        float* Q = queries.GetPoint(q);
        std::vector<int> probes =
            router.Query(Q, num_voting_neighbors).RoutingQuery();
        neighbors[q] = ivf.Query(Q, num_neighbors, probes, num_probes);
    }
    return neighbors;
}


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, const char* argv[]) {
    if (argc < 7 || argc > 8) {
        std::cerr <<
            "Usage: ./NaiveInsertBench"
            " <base-vectors> <queries> <partition-file>"
            " <gt-prefix> <runbook-yaml>"
            " <num-neighbors> [num-probes-query]\n";
        return 1;
    }

    const std::string point_file     = argv[1];
    const std::string query_file     = argv[2];
    const std::string partition_file = argv[3];
    const std::string gt_prefix      = argv[4];
    const std::string runbook_path   = argv[5];
    const int         num_neighbors  = std::stoi(argv[6]);
    const int         num_probes_q   = (argc == 8) ? std::stoi(argv[7]) : 1;

    // Number of nearest routing points to vote over during query time.

    // -----------------------------------------------------------------------
    // Parse runbook
    // -----------------------------------------------------------------------
    std::cout << "Parsing runbook " << runbook_path << " …\n";
    Runbook rb = ParseRunbook(runbook_path);
    std::cout << "Dataset : " << rb.dataset_name << "\n"
              << "max_pts : " << rb.max_pts << "\n"
              << "ops     : " << rb.ops.size() << "\n\n";

    // -----------------------------------------------------------------------
    // Load data
    // -----------------------------------------------------------------------
    std::cout << "Reading base vectors from " << point_file << " …\n";
    PointSet all_points = ReadPoints(point_file);
    PointSet queries    = ReadPoints(query_file);
    std::cout << "Loaded " << all_points.n << " base vectors (dim="
              << all_points.d << "), " << queries.n << " queries.\n\n";

    if (rb.max_pts == 0)
        rb.max_pts = all_points.n;
    if (all_points.n < rb.max_pts) {
        std::cerr << "Warning: base file has " << all_points.n
                  << " vectors but runbook max_pts=" << rb.max_pts
                  << ". Capping to file size.\n";
        rb.max_pts = all_points.n;
    }

    // -----------------------------------------------------------------------
    // Validate: runbook must start with an insert
    // -----------------------------------------------------------------------
    if (rb.ops.empty() || rb.ops[0].type != Operation::Type::INSERT) {
        std::cerr << "Runbook must begin with an insert operation.\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // Index state (populated after the first insert)
    // -----------------------------------------------------------------------
    std::unique_ptr<InvertedIndexHNSW> ivf;
    std::unique_ptr<HNSWRouter>        router;

    // The KMeansTreeRouter and the routing points it produces must outlive the
    // HNSWRouter (which holds a const-ref to routing_partition).
    std::unique_ptr<KMeansTreeRouter> kmtr;
    PointSet         routing_points;
    std::vector<int> routing_partition;

    bool index_built = false;
    int  num_voting_neighbors = 0;  // set on first insert to 3 * num_probes_q
    Timer timer;

    // -----------------------------------------------------------------------
    // Execute runbook
    // -----------------------------------------------------------------------
    for (size_t op_idx = 0; op_idx < rb.ops.size(); ++op_idx) {
        const Operation& op = rb.ops[op_idx];

        // ── INSERT ──────────────────────────────────────────────────────────
        if (op.type == Operation::Type::INSERT) {

            if (!index_built) {
                // First insert: build the full index.
                const uint32_t init_start = op.start;
                const uint32_t init_end   = op.end;
                const size_t   init_size  = init_end - init_start;

                std::cout << "[op " << op_idx + 1 << "] BUILD"
                          << "  base[" << init_start << ", " << init_end << ")"
                          << "  (" << init_size << " vectors)\n";

                // View of just the initial vectors.
                PointSet init_points;
                init_points.n = init_size;
                init_points.d = all_points.d;
                init_points.coordinates.assign(
                    all_points.coordinates.begin() + init_start * all_points.d,
                    all_points.coordinates.begin() + init_end   * all_points.d);

                // Load and validate partition.
                Clusters clusters  = ReadClusters(partition_file);
                const int num_shards = static_cast<int>(clusters.size());
                num_voting_neighbors = 3 * num_probes_q;
                {
                    size_t covered = 0;
                    for (const auto& c : clusters) covered += c.size();
                    if (covered != init_size) {
                        std::cerr << "Partition covers " << covered
                                  << " vectors but first insert batch has "
                                  << init_size << " — they must match.\n";
                        return 1;
                    }
                }

                // The cluster file uses global base-file IDs.  The router
                // training uses init_points which is a local 0-based slice, so
                // remap cluster IDs to local indices for the KMeansTreeRouter.
                Clusters local_clusters(clusters.size());
                for (size_t b = 0; b < clusters.size(); ++b) {
                    local_clusters[b].resize(clusters[b].size());
                    for (size_t i = 0; i < clusters[b].size(); ++i)
                        local_clusters[b][i] = clusters[b][i] - init_start;
                }

                // Train router on the initial vectors.
                KMeansTreeRouterOptions opts{
                    .num_centroids    = 32,
                    .min_cluster_size = 200,
                    .budget           = 50000,
                    .search_budget    = 5000,
                };
                kmtr = std::make_unique<KMeansTreeRouter>();
                timer.Start();
                kmtr->Train(init_points, local_clusters, opts);
                std::cout << "  KMeans-tree router : " << timer.Stop() << " s\n";

                {
                    auto [rp, rpart] = kmtr->ExtractPoints();
                    routing_points    = std::move(rp);
                    routing_partition = std::move(rpart);
                }

                router = std::make_unique<HNSWRouter>(
                    routing_points, num_shards, routing_partition,
                    HNSWParameters{ .M = 16, .ef_construction = 200, .ef_search = 200 });
                timer.Start();
                router->Train(routing_points);
                std::cout << "  HNSW router        : " << timer.Stop() << " s\n";

                // Build per-shard HNSW.  Pre-reserve capacity proportionally to
                // max_pts so future inserts don't need to resize.
                // Pass init_start as id_offset so GetPoint uses local indices
                // while hnswlib labels remain global.
                ivf = std::make_unique<InvertedIndexHNSW>(init_points, rb.max_pts);
                ivf->hnsw_parameters = HNSWParameters{
                    .M = 16, .ef_construction = 200, .ef_search = 120 };

                timer.Start();
                ivf->Build(init_points, clusters, rb.max_pts, init_start);
                std::cout << "  IVF-HNSW build     : " << timer.Stop() << " s"
                          << "  (capacity=" << rb.max_pts << ")\n";

                index_built = true;

            } else {
                // Subsequent inserts: naive route + addPoint.
                const uint32_t ins_start = op.start;
                const uint32_t ins_end   = op.end;
                const size_t   ins_count = ins_end - ins_start;

                std::cout << "[op " << op_idx + 1 << "] INSERT"
                          << "  base[" << ins_start << ", " << ins_end << ")"
                          << "  (" << ins_count << " vectors)\n";

                timer.Start();
                for (uint32_t i = ins_start; i < ins_end; ++i) {
                    float* p  = all_points.GetPoint(i);
                    int shard = router->NaiveRoute(p);
                    ivf->Insert(p, i, shard);
                }
                const double t = timer.Stop();
                std::cout << "  insert_time=" << t << " s"
                          << "  throughput=" << ins_count / t << " vec/s\n";
            }
        }

        // ── DELETE ──────────────────────────────────────────────────────────
        else if (op.type == Operation::Type::DELETE) {
            if (!index_built) {
                std::cerr << "[op " << op_idx + 1
                          << "] DELETE before index built — skipping.\n";
                continue;
            }

            const uint32_t del_start = op.start;
            const uint32_t del_end   = op.end;
            const size_t   del_count = del_end - del_start;

            std::cout << "[op " << op_idx + 1 << "] DELETE"
                      << "  base[" << del_start << ", " << del_end << ")"
                      << "  (" << del_count << " vectors)\n";

            timer.Start();
            for (uint32_t i = del_start; i < del_end; ++i) {
                ivf->Delete(i);
            }
            const double t = timer.Stop();
            std::cout << "  delete_time=" << t << " s"
                      << "  throughput=" << del_count / t << " vec/s\n";
        }

        // ── SEARCH ──────────────────────────────────────────────────────────
        else {  // SEARCH
            if (!index_built) {
                std::cerr << "[op " << op_idx + 1
                          << "] SEARCH before index built — skipping.\n";
                continue;
            }

            const std::string gt_file =
                gt_prefix + "/step" + std::to_string(op.step_num) + ".gt100";

            std::cout << "[op " << op_idx + 1 << "] SEARCH"
                      << "  step=" << op.step_num
                      << "  gt=" << gt_file << "\n";

            if (!std::filesystem::exists(gt_file)) {
                std::cout << "  GT file not found — skipping recall.\n";
                continue;
            }

            auto ground_truth = ReadGroundTruth(gt_file);
            // ConvertGroundTruth… recomputes exact distances using all_points,
            // so labels from any batch (including deleted ones that appear in
            // older GT entries but not newer ones) are resolved correctly.
            auto dist_to_kth  = ConvertGroundTruthToDistanceToKthNeighbor(
                ground_truth, num_neighbors, all_points, queries);

            timer.Start();
            auto neighbors = RunQueries(queries, *ivf, *router,
                                        num_neighbors, num_probes_q,
                                        num_voting_neighbors);
            const double qtime = timer.Stop();
            const double recall = Recall(neighbors, dist_to_kth, num_neighbors);

            std::cout << "  recall@" << num_neighbors << "=" << recall
                      << "  query_time=" << qtime << " s"
                      << "  QPS=" << queries.n / qtime
                      << "  num_probes=" << num_probes_q << "\n";
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
