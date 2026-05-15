#!/usr/bin/env python3
"""
extract_first_batch.py
----------------------
Reads a NeurIPS-2023 runbook YAML, finds the first insert operation,
and extracts that slice of vectors from the base .fbin file into a
new .fbin file.

Usage:
    python3 extract_first_batch.py <runbook.yaml> <base.fbin> <out.fbin>
"""

import sys
import struct
import numpy as np


def parse_first_insert(runbook_path):
    """Return (start, end) from the first insert operation in the runbook."""
    start = end = None
    in_first_insert = False
    found_insert = False

    with open(runbook_path) as f:
        for line in f:
            # Top-level or max_pts lines — skip
            if not line.startswith("    "):
                in_first_insert = False
                continue

            stripped = line.strip()

            # 4-space indent: fields of an operation
            if stripped.startswith("operation:"):
                val = stripped.split(":", 1)[1].strip().strip("'\"")
                if val == "insert" and not found_insert:
                    in_first_insert = True
                    found_insert = True
                else:
                    in_first_insert = False

            elif in_first_insert and stripped.startswith("start:"):
                start = int(stripped.split(":", 1)[1].strip())

            elif in_first_insert and stripped.startswith("end:"):
                end = int(stripped.split(":", 1)[1].strip())

            if start is not None and end is not None:
                return start, end

    raise ValueError("No insert operation with start/end found in runbook.")


def extract_batch(base_path, out_path, start, end):
    count = end - start
    with open(base_path, "rb") as f:
        n, d = struct.unpack("II", f.read(8))
        if end > n:
            raise ValueError(f"end={end} exceeds file size n={n}")
        f.seek(8 + start * d * 4)
        vecs = np.frombuffer(f.read(count * d * 4), dtype=np.float32).copy()

    with open(out_path, "wb") as f:
        f.write(struct.pack("II", count, d))
        f.write(vecs.tobytes())

    print(f"Extracted vectors [{start}, {end})  count={count}  dim={d}")
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 extract_first_batch.py <runbook.yaml> <base.fbin> <out.fbin>")
        sys.exit(1)

    runbook_path, base_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    start, end = parse_first_insert(runbook_path)
    print(f"First insert: [{start}, {end})  ({end - start} vectors)")
    extract_batch(base_path, out_path, start, end)
