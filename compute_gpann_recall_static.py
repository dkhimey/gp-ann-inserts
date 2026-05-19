import argparse
import csv
import numpy as np


# ── File readers ──────────────────────────────────────────────────────────────

def read_clusters(filename):
    clusters = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                clusters.append(list(map(int, line.split())))
    return clusters


def read_routes(filename):
    routes = []
    with open(filename) as f:
        num_routes = int(f.readline().strip())
        for _ in range(num_routes):
            header = f.readline().strip()
            if header != "R":
                raise ValueError(f"Expected route marker 'R', got: {header!r}")
            route = {}
            parts = f.readline().strip().split()
            route["routing_algorithm"] = parts[0]
            route["index_trainer"]     = parts[1]
            route["hnsw_num_voting_neighbors"] = int(parts[2])
            route["hnsw_ef_search"]            = int(parts[3])
            route["routing_time"]              = float(parts[4])
            route["try_increasing_num_shards"] = parts[5].lower() == "true"
            num_queries = int(parts[6])
            route["routing_index_options"] = {
                "budget":           int(parts[7]),
                "num_centroids":    int(parts[8]),
                "min_cluster_size": int(parts[9]),
            }
            route["buckets_to_probe"] = [
                list(map(int, f.readline().strip().split()))
                for _ in range(num_queries)
            ]
            routes.append(route)
    return routes


def read_fbin_ground_truth(filename):
    with open(filename, "rb") as f:
        num_queries = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        K           = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        total       = num_queries * K
        ids         = np.frombuffer(f.read(total * 4), dtype=np.uint32).reshape(num_queries, K)
        _dists      = np.frombuffer(f.read(total * 4), dtype=np.float32)   # unused but read
    return ids.astype(int).tolist()


# ── Core recall computation ───────────────────────────────────────────────────

def compute_recall_matrix(clusters, routes, gt, k_neighbors):
    """
    Returns a dict:
        recalls[route_idx][num_shards] = mean recall  (num_shards in 1..total_shards)
    """
    num_shards   = len(clusters)
    num_queries  = len(gt)

    # Build vector_id → cluster_id lookup
    max_vid = max(max(c) for c in clusters if c)
    vector_to_cluster = np.full(max_vid + 1, -1, dtype=np.int32)
    for cid, cluster in enumerate(clusters):
        vector_to_cluster[np.array(cluster, dtype=np.int64)] = cid

    # For each query, determine which clusters contain its k gt neighbors
    gt_cluster_sets = []
    for qid in range(num_queries):
        neighbor_ids = np.array(gt[qid][:k_neighbors], dtype=np.int64)
        valid = neighbor_ids[neighbor_ids <= max_vid]
        mapped = vector_to_cluster[valid]
        gt_cluster_sets.append(set(mapped[mapped >= 0].tolist()))

    # Compute recall for every (route, num_shards) combination
    recalls = {}
    for ridx, route in enumerate(routes):
        per_nshards = {}
        for n in range(1, num_shards + 1):
            hits = []
            for qid in range(num_queries):
                gt_parts = gt_cluster_sets[qid]
                if not gt_parts:
                    continue
                visited = set(route["buckets_to_probe"][qid][:n])
                hits.append(len(gt_parts & visited) / len(gt_parts))
            per_nshards[n] = float(np.mean(hits)) if hits else 0.0
        recalls[ridx] = per_nshards

    return recalls


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute recall for every (num_shards_searched, routing_method) pair."
    )
    p.add_argument("--partitions", required=True, help="Partition/clusters file")
    p.add_argument("--routes",     required=True, help="Routes file")
    p.add_argument("--gt",         required=True, help="Ground-truth .fbin file")
    p.add_argument("--output",     required=True, help="Output CSV path")
    p.add_argument("--k",          type=int, default=10, help="Number of neighbors (default: 10)")
    return p.parse_args()


def main():
    args = parse_args()

    print("Reading partitions …")
    clusters = read_clusters(args.partitions)
    num_shards = len(clusters)
    print(f"  {num_shards} shards, {sum(len(c) for c in clusters):,} total vectors")

    print("Reading routes …")
    routes = read_routes(args.routes)
    print(f"  {len(routes)} routing method(s)")

    print("Reading ground truth …")
    gt = read_fbin_ground_truth(args.gt)
    print(f"  {len(gt)} queries, using top-{args.k} neighbors")

    hnsw_routes = [(i, r) for i, r in enumerate(routes) if "hnsw" in r["routing_algorithm"].lower()]
    if not hnsw_routes:
        raise ValueError("No HNSW routes found in the routes file.")
    print(f"  {len(hnsw_routes)} HNSW route(s) (out of {len(routes)} total)")

    print("Computing recall …")
    hnsw_indices = [i for i, _ in hnsw_routes]
    recalls = compute_recall_matrix(clusters, [r for _, r in hnsw_routes], gt, k_neighbors=args.k)
    # remap recall keys back to original route indices for labelling
    recalls = {orig_i: recalls[new_i] for new_i, orig_i in enumerate(hnsw_indices)}

    # Build route labels from metadata
    route_labels = [
        f"route_{i}_{r['routing_algorithm']}_{r['index_trainer']}"
        for i, r in hnsw_routes
    ]

    print(f"Writing results to {args.output} …")
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_shards_searched"] + route_labels)
        for n in range(1, num_shards + 1):
            row = [n] + [f"{recalls[orig_i][n]:.6f}" for orig_i in hnsw_indices]
            writer.writerow(row)

    print("Done.")


if __name__ == "__main__":
    main()
