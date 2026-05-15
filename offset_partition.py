#!/usr/bin/env python3
"""
offset_partition.py
-------------------
Takes a partition file whose IDs are 0-based (i.e. relative to the first
batch extracted by extract_first_batch.py) and rewrites it with global IDs
by adding the first-insert start offset from the runbook.

The partition file format is one cluster per line, space-separated integer IDs
(as produced by the gp-ann Partition binary).

Usage:
    python3 offset_partition.py <runbook.yaml> <partition_in> <partition_out>
"""

import sys


def parse_first_insert_start(runbook_path):
    """Return the start index of the first insert operation in the runbook."""
    in_first_insert = False
    found_insert = False

    with open(runbook_path) as f:
        for line in f:
            if not line.startswith("    "):
                in_first_insert = False
                continue

            stripped = line.strip()

            if stripped.startswith("operation:"):
                val = stripped.split(":", 1)[1].strip().strip("'\"")
                if val == "insert" and not found_insert:
                    in_first_insert = True
                    found_insert = True
                else:
                    in_first_insert = False

            elif in_first_insert and stripped.startswith("start:"):
                return int(stripped.split(":", 1)[1].strip())

    raise ValueError("No insert operation with a start field found in runbook.")


def offset_partition(part_in, part_out, offset):
    n_clusters = 0
    n_ids = 0
    with open(part_in) as fin, open(part_out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ids = [str(int(x) + offset) for x in line.split()]
            fout.write(" ".join(ids) + "\n")
            n_clusters += 1
            n_ids += len(ids)

    print(f"Offset all IDs by {offset}")
    print(f"Clusters: {n_clusters}  total IDs: {n_ids}")
    print(f"Written to: {part_out}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 offset_partition.py <runbook.yaml> <partition_in> <partition_out>")
        sys.exit(1)

    runbook_path, part_in, part_out = sys.argv[1], sys.argv[2], sys.argv[3]

    offset = parse_first_insert_start(runbook_path)
    print(f"First insert start = {offset}  →  adding this offset to all partition IDs")
    offset_partition(part_in, part_out, offset)
