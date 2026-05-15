#pragma once

#include "defs.h"
#include "topn.h"

#include "../external/hnswlib/hnswlib/hnswlib.h"

#include <parlay/parallel.h>

struct InvertedIndexHNSW {
    HNSWParameters hnsw_parameters;
#ifdef MIPS_DISTANCE
    hnswlib::InnerProductSpace space;
#else
    hnswlib::L2Space space;
#endif

    std::vector<hnswlib::HierarchicalNSW<float> *> bucket_hnsws;

    // Maps every live label (base-file index) to the shard that owns it.
    // Sized to max_pts at construction; entries default to -1 (not yet inserted).
    // Used for O(1) shard lookup during Delete without scanning all shards.
    std::vector<int> label_to_shard;


    InvertedIndexHNSW(PointSet& points) : space(points.d) { }

    // Construct with an explicit label-space size.  Use this when the total
    // number of base vectors (max_pts) is known up front so that label_to_shard
    // never needs to be resized during inserts.
    InvertedIndexHNSW(PointSet& points, size_t max_pts) : space(points.d) {
        label_to_shard.assign(max_pts, -1);
    }

    // Standard build: each shard is allocated exactly as many slots as it has points.
    void Build(PointSet& points, const Clusters& clusters) {
        Build(points, clusters, /* total_capacity = */ 0);
    }

    // Build with extra capacity for future inserts.
    // total_capacity is the total number of elements that will ever live across
    // all shards (initial + all future inserts).  Each shard gets a proportional
    // reservation so that naive inserts rarely trigger an expensive resize.
    // Pass 0 (or use the single-arg overload) for a tight initial allocation.
    //
    // id_offset: subtract from each cluster ID to get the local index into
    // `points`.  Use when the cluster file contains global base-file IDs but
    // `points` is a locally-indexed slice starting at 0.  The global ID is
    // still used as the hnswlib label and stored in label_to_shard.
    void Build(PointSet& points, const Clusters& clusters, size_t total_capacity,
               uint32_t id_offset = 0) {
        size_t num_shards = clusters.size();
        bucket_hnsws.resize(num_shards);

        size_t initial_total = 0;
        for (const auto& c : clusters) initial_total += c.size();

        for (size_t b = 0; b < num_shards; ++b) {
            size_t shard_capacity = clusters[b].size();
            if (total_capacity > 0 && initial_total > 0) {
                double ratio = static_cast<double>(clusters[b].size()) / initial_total;
                shard_capacity = std::max(clusters[b].size(),
                                          static_cast<size_t>(ratio * total_capacity) + 1);
            }
            bucket_hnsws[b] = new hnswlib::HierarchicalNSW<float>(
                &space, shard_capacity,
                hnsw_parameters.M, hnsw_parameters.ef_construction,
                /* random_seed = */ 555 + b);
            bucket_hnsws[b]->setEf(hnsw_parameters.ef_search);
        }

        // Ensure label_to_shard is large enough for all initial labels.
        if (!clusters.empty()) {
            uint32_t max_label = 0;
            for (const auto& c : clusters)
                for (uint32_t id : c)
                    max_label = std::max(max_label, id);
            if (label_to_shard.size() <= max_label)
                label_to_shard.assign(max_label + 1, -1);
        }

        std::cout << "start HNSW insertions" << std::endl;

        // Each label appears in exactly one cluster, so parallel writes to
        // label_to_shard[label] have no conflicts.
        parlay::parallel_for(0, clusters.size(), [&](size_t b) {
            parlay::parallel_for(0, clusters[b].size(), [&](size_t i_local) {
                uint32_t id = clusters[b][i_local];
                float* p = points.GetPoint(id - id_offset);  // local index
                bucket_hnsws[b]->addPoint(p, id);            // global label
                label_to_shard[id] = static_cast<int>(b);
            });
        });
    }

    // Insert a single new vector into a specific shard.
    // global_id is the vector's index in the base file and becomes its hnswlib label.
    // If the shard is at capacity it is grown by 50% before the insert
    // (requires hnswlib >= 0.6 for resizeIndex).
    void Insert(float* point, uint32_t global_id, int shard) {
        auto* h = bucket_hnsws[shard];
        if (h->cur_element_count >= h->max_elements_) {
            size_t new_max = h->max_elements_ + std::max(h->max_elements_ / 2, size_t(1));
            h->resizeIndex(new_max);
        }
        h->addPoint(point, global_id);
        if (global_id < label_to_shard.size())
            label_to_shard[global_id] = shard;
    }

    // Shadow-delete a vector by label.  The element is marked deleted in hnswlib
    // so it will never be returned by future searches, but the graph edges are
    // preserved (no structural repair).  Idempotent: silently ignores labels that
    // were never inserted.
    void Delete(uint32_t global_id) {
        if (global_id >= label_to_shard.size()) return;
        int shard = label_to_shard[global_id];
        if (shard < 0) return;  // never inserted
        bucket_hnsws[shard]->markDelete(global_id);
        label_to_shard[global_id] = -1;  // mark as gone so double-deletes are silent
    }

    ~InvertedIndexHNSW() {
        for (size_t i = 0; i < bucket_hnsws.size(); ++i) {
            delete bucket_hnsws[i];
        }
    }

    NNVec Query(float* Q, int num_neighbors, const std::vector<int>& buckets_to_probe, int num_probes) const {
        TopN top_k(num_neighbors);
        for (int i = 0; i < num_probes; ++i) {
            const int bucket = buckets_to_probe[i];
            auto result = bucket_hnsws[bucket]->searchKnn(Q, num_neighbors);
            while (!result.empty()) {
                const auto [dist, label] = result.top();
                result.pop();
                top_k.Add(std::make_pair(dist, label));
            }
        }
        return top_k.Take();
    }

    NNVec QueryBucket(float* Q, int num_neighbors, int bucket) {
        auto result_pq = bucket_hnsws[bucket]->searchKnn(Q, num_neighbors);
        NNVec result;
        while (!result_pq.empty()) {
            result.emplace_back(result_pq.top());
            result_pq.pop();
        }
        return result;
    }
};
