#include <mpi.h>
#include <filesystem>
#include "distributed_query_benchmark.h"
#include "points_io.h"
#include "metis_io.h"

int main(int argc, const char* argv[]) {
    if (argc != 7) {
        std::cerr <<
            "Usage: mpirun -np <K> ./DistributedBench"
            " <input-points> <queries> <ground-truth>"
            " <num-neighbors> <partition-file> <router-file>\n"
            "\n"
            "  <input-points>   : .fbin / .u8bin / .i8bin base vectors\n"
            "  <queries>        : .fbin / .u8bin / .i8bin query vectors\n"
            "  <ground-truth>   : .bin (IDs+dists) or .ibin (IDs only)\n"
            "                     pass an empty string or nonexistent path\n"
            "                     to skip recall computation\n"
            "  <num-neighbors>  : K for recall@K\n"
            "  <partition-file> : cluster file produced by ./Partition\n"
            "  <router-file>    : HNSW router file produced by ./BuildRouter\n";
        return 1;
    }

    const std::string point_file        = argv[1];
    const std::string query_file        = argv[2];
    const std::string ground_truth_file = argv[3];
    const int         num_neighbors     = std::stoi(argv[4]);
    const std::string partition_file    = argv[5];
    const std::string router_file       = argv[6];

    MPI_Init(nullptr, nullptr);

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Suppress stdout on non-root ranks so the table prints cleanly
    if (rank != 0) std::cout.setstate(std::ios_base::failbit);

    DistributedQueryBenchmark bench;
    bench.num_neighbors        = num_neighbors;
    // Probe enough routing points to rank all shards; 10*comm_size ≈ 3*num_shards
    // when comm_size ≈ num_shards (typical MPI configuration).
    bench.num_voting_neighbors = 10 * comm_size;

    const double t_setup = MPI_Wtime();

    // The partition (point → shard map) is always needed: it drives shard-point
    // selection during a build and the theoretical-recall computation during the
    // sweep.
    std::cerr << "[rank " << rank << "] loading partition\n";
    bench.LoadPartition(partition_file);

    // Per-shard HNSW: load from cache if one exists for this
    // (partition basename, M, ef_construction); otherwise build and save it.
    // On a cache hit the full shard point set is never read — only the 8-byte
    // (n, dim) header is needed to set `dim`.
    const std::string shard_cache = bench.ShardIndexCachePath(partition_file);
    if (std::filesystem::exists(shard_cache)) {
        std::cerr << "[rank " << rank << "] loading cached in-shard HNSW\n";
        bench.ReadDim(point_file);
        bench.LoadInShardIndex(shard_cache);
    } else {
        std::cerr << "[rank " << rank << "] loading shard points\n";
        bench.LoadShardPointSet(point_file);
        std::cerr << "[rank " << rank << "] building in-shard HNSW\n";
        bench.BuildInShardIndex(shard_cache);
    }

    std::cerr << "[rank " << rank << "] loading router\n";
    bench.LoadRouter(router_file);
    std::cerr << "[rank " << rank << "] setup done in "
              << (MPI_Wtime() - t_setup) << " s\n";

    PointSet queries = ReadPoints(query_file);

    MPI_Barrier(MPI_COMM_WORLD);

    bench.ProcessSearchSweep(queries, ground_truth_file);

    MPI_Finalize();
    return 0;
}
