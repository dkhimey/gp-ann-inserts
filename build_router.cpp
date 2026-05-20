// build_router.cpp
//
// Trains an HNSW router from a GP-ANN partition file and serializes it to
// disk so that DistributedBench (and other tools) can load it later.
//
// The router is a two-level structure:
//   1. KMeansTreeRouter  — trains k-means centroids per shard, producing a
//                          compact set of "routing points" that represent each
//                          shard.
//   2. HNSWRouter        — builds an HNSW index over those routing points so
//                          that a query vector can quickly find its nearest
//                          shards.
//
// Output files (two, always written together):
//   <output-router>                          — hnswlib saved index
//   <output-router>.routing_index_partition  — centroid → shard mapping
//
// Usage
// -----
//   ./BuildRouter <input-points> <partition-file> <output-router>
//
//   <input-points>    : .fbin or .u8bin file of base vectors
//   <partition-file>  : cluster file produced by ./Partition
//                       (e.g. "out.k=40.GP")
//   <output-router>   : path where the router index will be written

#include <iostream>

#include <parlay/primitives.h>

#include "defs.h"
#include "hnsw_router.h"
#include "kmeans_tree_router.h"
#include "metis_io.h"
#include "points_io.h"

int main(int argc, const char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: ./BuildRouter <input-points> <partition-file> <output-router>\n";
        return 1;
    }

    const std::string point_file      = argv[1];
    const std::string partition_file  = argv[2];
    const std::string output_router   = argv[3];

    // ------------------------------------------------------------------
    // Load base points
    // ------------------------------------------------------------------
    std::cout << "Loading points from " << point_file << " ...\n";
    PointSet points;
    if (point_file.ends_with(".fbin")) {
        points = ReadPointsMmap(point_file);
    } else if (point_file.ends_with(".u8bin")) {
        points = ReadU8BinMmap(point_file);
    } else {
        points = ReadPoints(point_file);
    }
    std::cout << "  n=" << points.n << "  d=" << points.d << "\n";

    // ------------------------------------------------------------------
    // Load partition (cluster file produced by ./Partition)
    // ------------------------------------------------------------------
    std::cout << "Loading partition from " << partition_file << " ...\n";
    Clusters clusters = ReadClusters(partition_file);
    const int num_shards = (int)clusters.size();
    std::cout << "  num_shards=" << num_shards << "\n";

    // ------------------------------------------------------------------
    // Train KMeans-tree router to produce a compact set of routing points
    // ------------------------------------------------------------------
    KMeansTreeRouterOptions kmtr_opts{
        .num_centroids   = 32,
        .min_cluster_size = 200,
        .budget          = 50000,
        .search_budget   = 5000,
    };

    Timer timer;
    timer.Start();
    KMeansTreeRouter kmtr;
    kmtr.Train(points, clusters, kmtr_opts);
    std::cout << "KMeansTreeRouter training took " << timer.Stop() << " s\n";

    auto [routing_points, routing_index_partition] = kmtr.ExtractPoints();
    std::cout << "  routing_points.n=" << routing_points.n << "\n";

    // ------------------------------------------------------------------
    // Build HNSW over routing points
    // ------------------------------------------------------------------
    HNSWParameters hnsw_params{ .M = 16, .ef_construction = 200, .ef_search = 200 };

    timer.Start();
    HNSWRouter hnsw_router(routing_points, num_shards, routing_index_partition, hnsw_params);
    hnsw_router.Train(routing_points);
    std::cout << "HNSWRouter training took " << timer.Stop() << " s\n";

    // ------------------------------------------------------------------
    // Serialize: writes two files:
    //   <output-router>
    //   <output-router>.routing_index_partition
    // ------------------------------------------------------------------
    hnsw_router.Serialize(output_router);
    std::cout << "Router written to " << output_router << "\n";
    std::cout << "Partition written to " << output_router << ".routing_index_partition\n";

    return 0;
}
