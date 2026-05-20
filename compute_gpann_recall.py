import os
import glob
import argparse
import numpy as np

DEFAULT_PARTITION_FACTORS = [1, 2, 5]

def readclusters(filename):
    clusters = []
    with open(filename, 'r') as f:
        for line in f:
            cluster = list(map(int, line.strip().split()))
            clusters.append(cluster)
    return clusters

def readroutes(filename):
    routes = []
    with open(filename, 'r') as f:
        num_routes = int(f.readline().strip())
        for _ in range(num_routes):
            header = f.readline().strip()
            if header != 'R':
                print(f"routing config doesn't start with marker R. Instead: {header}")
            route = {}
            line = f.readline().strip()
            parts = line.split()
            route['routing_algorithm'] = parts[0]
            route['index_trainer'] = parts[1]
            route['hnsw_num_voting_neighbors'] = int(parts[2])
            route['hnsw_ef_search'] = int(parts[3])
            route['routing_time'] = float(parts[4])
            route['try_increasing_num_shards'] = parts[5].lower() == 'true'
            num_queries = int(parts[6])
            route['routing_index_options'] = {
                'budget': int(parts[7]),
                'num_centroids': int(parts[8]),
                'min_cluster_size': int(parts[9])
            }
            route['buckets_to_probe'] = []
            for _ in range(num_queries):
                line = f.readline().strip()
                buckets = list(map(int, line.split()))
                route['buckets_to_probe'].append(buckets)
            routes.append(route)

    # for i, route in enumerate(routes):
    #     # 
    #     if route['routing_algorithm'] == 'KMeansTree':
    #         print(f"routing algo {i}: {route['routing_algorithm']}")
    #         print(f"  index trainer: {route['index_trainer']}")
    #         print(f"  hnsw num voting neighbors: {route['hnsw_num_voting_neighbors']}")
    #         print(f"  hnsw ef search: {route['hnsw_ef_search']}")
    #         print(f"  routing time: {route['routing_time']:.4f} seconds")
    #         print(f"  routing index options: {route['routing_index_options']}")
    return routes

def read_fbin_ground_truth(filename):
    with open(filename, "rb") as f:
        # Read number of queries and K
        num_queries = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        K = np.frombuffer(f.read(4), dtype=np.uint32)[0]

        total_elements = num_queries * K

        # Read all IDs
        all_ids = np.frombuffer(f.read(total_elements * 4), dtype=np.uint32)

        if all_ids.size != total_elements:
            raise ValueError("Failed to read neighbor IDs")

        # read distances
        all_distances = np.frombuffer(f.read(total_elements * 4), dtype=np.float32)

        # Reshape into list of lists
        gt = all_ids.reshape((num_queries, K)).astype(int).tolist()
        dists = all_distances.reshape((num_queries, K)).tolist()

        return gt, dists
    
def compute_recall_gp(
    partitions_file,
    routes_file,
    gt_file,
    k_neighbors,
    route_idx = 0,
    partition_factors = None,
):

    if partition_factors is None:
        partition_factors = DEFAULT_PARTITION_FACTORS

    # read in partitions (clusters)
    clusters = readclusters(partitions_file)
    # read in routes
    routes = readroutes(routes_file)
    # read in gt
    gt, dists = read_fbin_ground_truth(gt_file)
    num_queries = len(gt)

    recalls = {}
    for num_partitions in partition_factors:
        recalls[num_partitions] = []
    per_partition_counts = {
        num_partitions: np.zeros(len(clusters), dtype=np.int64)
        for num_partitions in partition_factors
    }

    total_vectors = sum(len(cluster) for cluster in clusters)
    max_vector_id = max(max(cluster) for cluster in clusters)
    max_gt_id = max(max(neighbors) for neighbors in gt)
    print(f"number of vectors partitioned: {total_vectors}")
    print(f"maximum vector id: {max_vector_id}")
    print(f"maximum gt id: {max_gt_id}")

    # precompute vector_id -> cluster_id lookup for O(1) membership
    vector_to_cluster = np.full(max_vector_id + 1, -1, dtype=np.int32)
    for cluster_id, cluster in enumerate(clusters):
        vector_to_cluster[np.array(cluster, dtype=np.int64)] = cluster_id

    for query_id in range(num_queries):
        if query_id % 1000 == 0:
            print(f"Processing query {query_id}/{num_queries}")
        # get the partitions for the gt neighbors for this query

        neighbor_ids = np.array(gt[query_id][:k_neighbors], dtype=np.int64)
        mapped_partitions = vector_to_cluster[neighbor_ids]
        gt_partitions = mapped_partitions[mapped_partitions >= 0].tolist()

        if len(gt_partitions) == 0:
            print(f"Warning: No ground truth partitions found for query {query_id}")
            for neighbor_id in gt[query_id][:k_neighbors]:
                print(f"    Neighbor ID: {neighbor_id}")

        # get the partitions visited for this query

        for num_partitions in partition_factors:
            visited_partitions = routes[route_idx]['buckets_to_probe'][query_id][:num_partitions]
            visited_set = set(visited_partitions)
            for partition_id in visited_set:
                if 0 <= partition_id < len(clusters):
                    per_partition_counts[num_partitions][partition_id] += 1

            if len(gt_partitions) > 0:
                hits = sum(1 for part in gt_partitions if part in visited_set)
                recall = hits / len(gt_partitions)
                recalls[num_partitions].append(recall)

    recalls_mean = {
        num_partitions: np.mean(recalls_list) if recalls_list else 0.0
        for num_partitions, recalls_list in recalls.items()
    }
    cof = {}
    for num_partitions, counts in per_partition_counts.items():
        mean = float(np.mean(counts))
        std = float(np.std(counts))
        cof[num_partitions] = (std / mean) if mean > 0 else 0.0

    return recalls_mean, cof

def compute_recall_gp_all_routes(
    partitions_file,
    routes_file,
    gt_file,
    k_neighbors,
    partition_factors,
):
    routes = readroutes(routes_file)
    results = {}
    cofs = {}
    for route_idx in range(len(routes)):
        recalls, cof = compute_recall_gp(
            partitions_file=partitions_file,
            routes_file=routes_file,
            gt_file=gt_file,
            k_neighbors=k_neighbors,
            route_idx=route_idx,
            partition_factors=partition_factors,
        )
        results[str(route_idx)] = recalls
        cofs[str(route_idx)] = cof

    table = {}
    for route_idx, recalls in results.items():
        for num_partitions, value in recalls.items():
            entry = table.setdefault(f"partitions_{num_partitions}", {
                "recall": {},
                "cof": {}
            })
            entry["recall"][route_idx] = value
            entry["cof"][route_idx] = cofs[route_idx].get(num_partitions, 0.0)

    return table

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute GP-ANN recall per step and write results to CSV."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing both partitions and routes files",
    )
    parser.add_argument(
        "--gt-dir",
        default="/dataset/new_subsets",
        help="Directory containing subset_step{step}.gt100 files",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="CSV output path",
    )
    parser.add_argument(
        "--route-idx",
        type=int,
        default=0,
        help="Route index to evaluate (ignored if --all-routes is set)",
    )
    parser.add_argument(
        "--all-routes",
        action="store_true",
        help="Compute recall for all routes",
    )
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument(
        "--partition-factors",
        type=int,
        nargs="+",
        default=DEFAULT_PARTITION_FACTORS,
        help="Partition factors to evaluate (space-separated)",
    )
    parser.add_argument("--step-start", type=int, default=256)
    parser.add_argument("--step-end", type=int, default=946)
    parser.add_argument("--step-stride", type=int, default=2)
    return parser.parse_args()

def read_completed_steps(output_path):
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return set()

    completed_steps = set()
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("step,"):
                continue
            parts = line.split(",")
            if not parts:
                continue
            try:
                completed_steps.add(int(parts[0]))
            except ValueError:
                continue
    return completed_steps

def main():
    args = parse_args()

    partition_factors = list(dict.fromkeys(args.partition_factors))

    if args.all_routes and args.route_idx is not None:
        print("--all-routes is set; ignoring --route-idx")

    file_exists = os.path.exists(args.output_path)
    file_has_content = file_exists and os.path.getsize(args.output_path) > 0
    completed_steps = read_completed_steps(args.output_path)

    partition_glob = os.path.join(args.results_dir, "step*_results.k=*.GP.o=0.0")
    partition_files = glob.glob(partition_glob)
    if not partition_files:
        raise FileNotFoundError(f"No partition files found in {args.results_dir}")
    first_partition_file = os.path.basename(sorted(partition_files)[0])
    try:
        num_partitions = int(first_partition_file.split("k=")[1].split(".GP")[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Could not parse partition count from {first_partition_file}"
        ) from exc

    with open(args.output_path, 'a') as f:
        route_ids = None
        header_written = file_has_content

        for step in range(args.step_start, args.step_end + 1, args.step_stride):
            if step in completed_steps:
                continue
            print(f"Evaluating step {step}")
            partitions_file = f"{args.results_dir}/step{step}_results.k={num_partitions}.GP.o=0.0"
            routes_file = f"{args.results_dir}/step{step}_results_routing.routes.routes"
            gt_file = f"{args.gt_dir}/subset_step{step}.gt100"

            if args.all_routes:
                table = compute_recall_gp_all_routes(
                    partitions_file=partitions_file,
                    routes_file=routes_file,
                    gt_file=gt_file,
                    k_neighbors=args.k_neighbors,
                    partition_factors=partition_factors,
                )
                if route_ids is None:
                    first_key = next(iter(table))
                    route_ids = sorted(int(k) for k in table[first_key]["recall"].keys())
                    header = (
                        "step,partition_factor," +
                        ",".join([f"route_{rid}" for rid in route_ids]) + "," +
                        ",".join([f"cof_route_{rid}" for rid in route_ids]) + "\n"
                    )
                    if not header_written:
                        f.write(header)
                        header_written = True

                for num_partitions in partition_factors:
                    key = f"partitions_{num_partitions}"
                    row_values = [
                        str(table[key]["recall"].get(str(rid), "")) for rid in route_ids
                    ]
                    cof_values = [
                        str(table[key]["cof"].get(str(rid), "")) for rid in route_ids
                    ]
                    row = (
                        f"{step},{num_partitions}," + ",".join(row_values + cof_values) + "\n"
                    )
                    f.write(row)
            else:
                recalls, cof = compute_recall_gp(
                    partitions_file=partitions_file,
                    routes_file=routes_file,
                    gt_file=gt_file,
                    k_neighbors=args.k_neighbors,
                    route_idx=args.route_idx,
                    partition_factors=partition_factors,
                )
                if not header_written:
                    header = "step," + ",".join(
                        [f"partitions_{bf}" for bf in partition_factors] +
                        [f"cof_partitions_{bf}" for bf in partition_factors]
                    ) + "\n"
                    f.write(header)
                    header_written = True
                row = f"{step}," + ",".join(
                    [f"{recalls[bf]}" for bf in partition_factors] +
                    [f"{cof[bf]}" for bf in partition_factors]
                ) + "\n"
                f.write(row)

            f.flush()
            print(f"Step {step} complete")

if __name__ == "__main__":
    main()


# python compute_gpann_recall.py \
#     --results-dir /users/dkhimey/extra/vectorDB/Benchmarks/big-ann/cluster_history_gpann_10 \
#     --output-path /users/dkhimey/extra/vectorDB/Benchmarks/big-ann/cluster_history_gpann_10/recall_results_all_routes.csv \
#     --partition-factors 1 2 3 4 \
#     --all-routes