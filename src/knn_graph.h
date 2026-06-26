#pragma once

#include <iostream>
#include <atomic>
#include <chrono>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <random>
#include <sstream>
#include "defs.h"
#include "dist.h"
#include "spinlock.h"
#include "topn.h"


inline std::vector<int> TopKNeighbors(PointSet& P, uint32_t my_id, int k) {
    TopN top_k(k);
    float* Q = P.GetPoint(my_id);
    for (uint32_t i = 0; i < P.n; ++i) {
        if (i == my_id)
            continue;
        float new_dist = distance(P.GetPoint(i), Q, P.d);
        top_k.Add(std::make_pair(new_dist, i));
    }
    auto x = top_k.Take();
    std::vector<int> y;
    for (const auto& a : x)
        y.push_back(a.second);
    return y;
}

inline AdjGraph BuildExactKNNGraph(PointSet& P, int k) {
    AdjGraph graph(P.n);
    parlay::parallel_for(0, P.n, [&](size_t i) { graph[i] = TopKNeighbors(P, i, k); });
    return graph;
}

struct ApproximateKNNGraphBuilder {
    using Bucket = std::vector<uint32_t>;

    PointSet ExtractPoints(PointSet& points, const Bucket& ids) {
        PointSet bucket_points;
        bucket_points.d = points.d;
        bucket_points.n = ids.size();
        for (auto id : ids) {
            float* P = points.GetPoint(id);
            for (size_t d = 0; d < points.d; ++d) {
                bucket_points.coordinates.push_back(P[d]);
            }
        }
        return bucket_points;
    }

    std::vector<Bucket> RecursivelySketch(PointSet& points, const Bucket& ids, int depth, int fanout) {
        if (ids.size() <= MAX_CLUSTER_SIZE) {
            return { ids };
        }

        if (depth == 0) {
            timer.Start();
        }

        // sample leaders
        size_t num_leaders = depth == 0 ? TOP_LEVEL_NUM_LEADERS : ids.size() * FRACTION_LEADERS;
        num_leaders = std::min<size_t>(num_leaders, MAX_NUM_LEADERS);
        num_leaders = std::max<size_t>(num_leaders, 3);
        Bucket leaders(num_leaders);
        std::mt19937 prng(seed);
        std::sample(ids.begin(), ids.end(), leaders.begin(), leaders.size(), prng);

        PointSet leader_points = ExtractPoints(points, leaders);
        std::vector<Bucket> clusters(leaders.size());

        { // less readable than map + zip + flatten, but at least it's as efficient as possible for fanout = 1
            parlay::sequence<std::pair<uint32_t, uint32_t>> flat(ids.size() * fanout);

            parlay::parallel_for(0, ids.size(), [&](size_t i) {
                uint32_t point_id = ids[i];
                auto cl = ClosestLeaders(points, leader_points, point_id, fanout).Take();
                for (int j = 0; j < fanout; ++j) {
                    flat[i * fanout + j] = std::make_pair(cl[j].second, point_id);
                }
            });

            auto pclusters = parlay::group_by_index(flat, leaders.size());
            // copy clusters from parlay::sequence to std::vector
            parlay::parallel_for(0, pclusters.size(), [&](size_t i) { clusters[i] = Bucket(pclusters[i].begin(), pclusters[i].end()); });
        }

        leaders.clear();
        leaders.shrink_to_fit();
        leader_points.Drop();

        if (depth == 0) {
            double time = timer.Stop();
            if (!quiet)
                std::cout << "Closest leaders on top level took " << time << std::endl;
        }

        std::vector<Bucket> buckets;
        std::sort(clusters.begin(), clusters.end(), [&](const auto& b1, const auto& b2) { return b1.size() > b2.size(); });
        while (!clusters.empty() && clusters.back().size() < MIN_CLUSTER_SIZE) {
            if (buckets.empty() || clusters.back().size() + buckets.back().size() > MAX_MERGED_CLUSTER_SIZE) {
                buckets.emplace_back();
            }
            // merge small clusters together -- and already store them in the return buckets
            // this will add some stupid long range edges. but hopefully this is better than having some isolated nodes.
            // another fix could be to do a brute-force comparison with all points in the bucket one recursion level higher
            // Caveat --> we have to hold the data structures for the graph already
            for (const auto id : clusters.back()) {
                buckets.back().push_back(id);
            }
            clusters.pop_back();
        }

        // recurse on clusters
        if (depth == 0) {
            sketch_clusters_done.store(0, std::memory_order_relaxed);
            sketch_total_clusters = clusters.size();
            sketch_phase_start = std::chrono::steady_clock::now();
            sketch_last_print_ms.store(-120000, std::memory_order_relaxed);
            if (!quiet)
                std::cout << "Recursively sketching " << sketch_total_clusters << " top-level clusters..." << std::endl;
        }
        SpinLock bucket_lock;
        parlay::parallel_for(
                0, clusters.size(),
                [&](size_t cluster_id) {
                    std::vector<Bucket> recursive_buckets;
                    if (depth > MAX_DEPTH || (depth > CONCERNING_DEPTH && clusters[cluster_id].size() > TOO_SMALL_SHRINKAGE_FRACTION * ids.size())) {
                        // Base case for duplicates and near-duplicates. Split the buckets randomly
                        auto ids_copy = clusters[cluster_id];
                        std::mt19937 prng(seed + depth + ids.size());
                        std::shuffle(ids_copy.begin(), ids_copy.end(), prng);
                        for (size_t i = 0; i < ids_copy.size(); i += MAX_CLUSTER_SIZE) {
                            auto& new_bucket = recursive_buckets.emplace_back();
                            for (size_t j = 0; j < MAX_CLUSTER_SIZE; ++j) {
                                new_bucket.push_back(ids_copy[j]);
                            }
                        }
                    } else {
                        // The normal case
                        recursive_buckets = RecursivelySketch(points, clusters[cluster_id], depth + 1, /*fanout=*/1);
                    }

                    bucket_lock.lock();
                    buckets.insert(buckets.end(), recursive_buckets.begin(), recursive_buckets.end());
                    bucket_lock.unlock();

                    if (depth == 0) {
                        size_t c = sketch_clusters_done.fetch_add(1, std::memory_order_relaxed) + 1;
                        int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - sketch_phase_start).count();
                        int64_t prev = sketch_last_print_ms.load(std::memory_order_relaxed);
                        bool time_to_print = (elapsed_ms - prev >= 120000) || (c == sketch_total_clusters);
                        if (!quiet && time_to_print &&
                                sketch_last_print_ms.compare_exchange_strong(prev, elapsed_ms, std::memory_order_relaxed)) {
                            double pct = 100.0 * c / sketch_total_clusters;
                            double elapsed_s = elapsed_ms / 1000.0;
                            double eta_s = (c > 0 && c < sketch_total_clusters)
                                    ? elapsed_s * (sketch_total_clusters - c) / c : 0.0;
                            std::cout << "[" << (int)(elapsed_s / 60) << "m" << ((int)elapsed_s % 60) << "s] "
                                      << "Sketch clusters " << c << " / " << sketch_total_clusters
                                      << " (" << (int)pct << "%)"
                                      << "  ETA: " << (int)(eta_s / 60) << "m" << ((int)eta_s % 60) << "s"
                                      << std::endl;
                        }
                    }
                },
                1);

        return buckets;
    }

    AdjGraph BuildApproximateNearestNeighborGraph(PointSet& points, int num_neighbors) {
        Bucket all_ids(points.n);
        std::iota(all_ids.begin(), all_ids.end(), 0);
        std::vector<Bucket> buckets;
        for (int rep = 0; rep < REPETITIONS; ++rep) {
            if (!quiet)
                std::cout << "Sketching rep " << rep << std::endl;
            Timer timer2;
            timer2.Start();
            std::vector<Bucket> new_buckets = RecursivelySketch(points, all_ids, 0, FANOUT);
            if (!quiet)
                std::cout << "Finished sketching rep. It took " << timer2.Stop() << " seconds." << std::endl;
            buckets.insert(buckets.end(), new_buckets.begin(), new_buckets.end());
        }
        if (!quiet)
            std::cout << "Start bucket brute force" << std::endl;
        return BruteForceBuckets(points, buckets, num_neighbors);
    }


    std::vector<NNVec> CrunchBucket(PointSet& points, const Bucket& bucket, int num_neighbors) {
        PointSet bucket_points = ExtractPoints(points, bucket);

        std::vector<TopN> neighbors(bucket.size(), TopN(num_neighbors));

        for (size_t i = 0; i < bucket.size(); ++i) {
            float* P = bucket_points.GetPoint(i);
            for (size_t j = i + 1; j < bucket.size(); ++j) {
                float* Q = bucket_points.GetPoint(j);
                float dist = distance(P, Q, points.d);
                neighbors[i].Add(std::make_pair(dist, bucket[j]));
                neighbors[j].Add(std::make_pair(dist, bucket[i]));
            }
        }

        std::vector<NNVec> result(bucket.size());
        for (size_t i = 0; i < bucket.size(); ++i) {
            result[i] = neighbors[i].Take();
        }
        return result;
    }

    AdjGraph BruteForceBuckets(PointSet& points, std::vector<Bucket>& buckets, int num_neighbors) {
        std::vector<SpinLock> locks(points.n);
        std::vector<NNVec> top_neighbors(points.n);

        if (!quiet) {
            std::cout << "Number of buckets to crunch " << buckets.size() << std::endl;
            std::vector<size_t> bucket_sizes(buckets.size());
            for (size_t i = 0; i < buckets.size(); ++i)
                bucket_sizes[i] = buckets[i].size();
            double avg_size = 0.0;
            for (const auto& b : bucket_sizes)
                avg_size += b;
            avg_size /= buckets.size();
            std::cout << "avg bucket size : " << avg_size << std::endl;
            std::sort(bucket_sizes.begin(), bucket_sizes.end());
            std::vector<double> quantiles = { 0.0, 0.01, 0.05, 0.1, 0.15, 0.5, 0.85, 0.9, 0.95, 0.99, 1.0 };
            for (double quantile : quantiles) {
                size_t pos = quantile * buckets.size();
                pos = std::clamp<size_t>(pos, 0, buckets.size() - 1);
                std::cout << "quant " << quantile << " size :  " << bucket_sizes[pos] << std::endl;
            }
        }


        timer.Start();

        std::atomic<size_t> buckets_done{0};
        const size_t total_buckets = buckets.size();
        const auto bf_start_time = std::chrono::steady_clock::now();
        std::atomic<int64_t> last_print_ms{-120000}; // forces first print immediately

        parlay::parallel_for(
                0, buckets.size(),
                [&](size_t bucket_id) {
                    auto& bucket = buckets[bucket_id];
                    auto bucket_neighbors = CrunchBucket(points, bucket, num_neighbors);
                    for (size_t j = 0; j < bucket.size(); ++j) {
                        auto point_id = bucket[j];

                        locks[point_id].lock();
                        // insert new neighbors. due to possible duplicate neighbors, we can't insert directly into the top-k data structure, and instead have
                        // to do this
                        auto& n = top_neighbors[point_id];
                        n.insert(n.end(), bucket_neighbors[j].begin(), bucket_neighbors[j].end());
                        // remove duplicates
                        std::sort(n.begin(), n.end(), [](const auto& l, const auto& r) { return std::tie(l.second, l.first) < std::tie(r.second, r.first); });
                        n.erase(std::unique(n.begin(), n.end(), [&](const auto& l, const auto& r) { return l.second == r.second; }), n.end());

                        // keep top k
                        std::sort(n.begin(), n.end());
                        n.resize(std::min<size_t>(n.size(), num_neighbors));

                        locks[point_id].unlock();
                    }
                    bucket.clear();
                    bucket.shrink_to_fit();

                    {
                        size_t c = buckets_done.fetch_add(1, std::memory_order_relaxed) + 1;
                        int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - bf_start_time).count();
                        int64_t prev = last_print_ms.load(std::memory_order_relaxed);
                        bool time_to_print = (elapsed_ms - prev >= 120000) || (c == total_buckets);
                        if (!quiet && time_to_print &&
                                last_print_ms.compare_exchange_strong(prev, elapsed_ms, std::memory_order_relaxed)) {
                            double pct = 100.0 * c / total_buckets;
                            double elapsed_s = elapsed_ms / 1000.0;
                            double eta_s = (c > 0 && c < total_buckets)
                                    ? elapsed_s * (total_buckets - c) / c : 0.0;
                            std::cout << "[" << (int)(elapsed_s / 60) << "m" << ((int)elapsed_s % 60) << "s] "
                                      << "Buckets " << c << " / " << total_buckets
                                      << " (" << (int)pct << "%)"
                                      << "  ETA: " << (int)(eta_s / 60) << "m" << ((int)eta_s % 60) << "s"
                                      << std::endl;
                        }
                    }
                },
                1);

        if (!quiet)
            std::cout << "Brute forcing buckets took " << timer.Stop() << std::endl;

        AdjGraph graph(points.n);
        parlay::parallel_for(0, points.n, [&](size_t i) {
            for (const auto& [dist, point_id] : top_neighbors[i]) {
                graph[i].push_back(point_id);
            }
        });
        return graph;
    }


    ApproximateKNNGraphBuilder() = default;

    ApproximateKNNGraphBuilder(const ApproximateKNNGraphBuilder& o)
        : sketch_clusters_done(0), sketch_total_clusters(o.sketch_total_clusters),
          sketch_phase_start(o.sketch_phase_start), sketch_last_print_ms(-120000),
          seed(o.seed), FRACTION_LEADERS(o.FRACTION_LEADERS),
          TOP_LEVEL_NUM_LEADERS(o.TOP_LEVEL_NUM_LEADERS), MAX_NUM_LEADERS(o.MAX_NUM_LEADERS),
          MAX_CLUSTER_SIZE(o.MAX_CLUSTER_SIZE), MIN_CLUSTER_SIZE(o.MIN_CLUSTER_SIZE),
          MAX_MERGED_CLUSTER_SIZE(o.MAX_MERGED_CLUSTER_SIZE), REPETITIONS(o.REPETITIONS),
          FANOUT(o.FANOUT), MAX_DEPTH(o.MAX_DEPTH), CONCERNING_DEPTH(o.CONCERNING_DEPTH),
          TOO_SMALL_SHRINKAGE_FRACTION(o.TOO_SMALL_SHRINKAGE_FRACTION),
          quiet(o.quiet), timer(o.timer) {}

    ApproximateKNNGraphBuilder& operator=(const ApproximateKNNGraphBuilder& o) {
        if (this == &o) return *this;
        sketch_clusters_done.store(0, std::memory_order_relaxed);
        sketch_total_clusters = o.sketch_total_clusters;
        sketch_phase_start = o.sketch_phase_start;
        sketch_last_print_ms.store(-120000, std::memory_order_relaxed);
        seed = o.seed; FRACTION_LEADERS = o.FRACTION_LEADERS;
        TOP_LEVEL_NUM_LEADERS = o.TOP_LEVEL_NUM_LEADERS; MAX_NUM_LEADERS = o.MAX_NUM_LEADERS;
        MAX_CLUSTER_SIZE = o.MAX_CLUSTER_SIZE; MIN_CLUSTER_SIZE = o.MIN_CLUSTER_SIZE;
        MAX_MERGED_CLUSTER_SIZE = o.MAX_MERGED_CLUSTER_SIZE; REPETITIONS = o.REPETITIONS;
        FANOUT = o.FANOUT; MAX_DEPTH = o.MAX_DEPTH; CONCERNING_DEPTH = o.CONCERNING_DEPTH;
        TOO_SMALL_SHRINKAGE_FRACTION = o.TOO_SMALL_SHRINKAGE_FRACTION;
        quiet = o.quiet; timer = o.timer;
        return *this;
    }

    std::atomic<size_t> sketch_clusters_done{0};
    size_t sketch_total_clusters{0};
    std::chrono::steady_clock::time_point sketch_phase_start;
    std::atomic<int64_t> sketch_last_print_ms{-120000};

    int seed = 555;
    double FRACTION_LEADERS = 0.005;
    size_t TOP_LEVEL_NUM_LEADERS = 950;
    size_t MAX_NUM_LEADERS = 1500;
    size_t MAX_CLUSTER_SIZE = 5000;
    size_t MIN_CLUSTER_SIZE = 50;
    size_t MAX_MERGED_CLUSTER_SIZE = 2500;
    int REPETITIONS = 3;
    int FANOUT = 3;
    int MAX_DEPTH = 14;
    int CONCERNING_DEPTH = 10;
    double TOO_SMALL_SHRINKAGE_FRACTION = 0.8;

    bool quiet = false;

    Timer timer;
};

inline void Symmetrize(AdjGraph& graph) {
    std::vector<size_t> degrees;
    degrees.reserve(graph.size());
    for (const auto& n : graph)
        degrees.push_back(n.size());
    for (size_t u = 0; u < graph.size(); ++u) {
        for (size_t j = 0; j < degrees[u]; ++j) {
            auto v = graph[u][j];
            graph[v].push_back(u);
        }
    }
}
