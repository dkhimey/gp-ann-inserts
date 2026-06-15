#!/bin/bash

BASE=/dataset/big-ann-benchmarks/data

# --- clustered ---
mpirun -mca btl_tcp_if_include br-flat-lan-1 -np 10 --rankfile rankfile.txt \
    taskset -c 0-31 ./release_l2/DistributedInsertBenchSweep \
    ${BASE}/bigann-clustered/bigann-100M-clustered.u8bin \
    ${BASE}/bigann-clustered/query.public.10K.u8bin \
    ${BASE}/bigann-clustered/gpann_partitions/bigann100Mclustered.k\=10.GP \
    ${BASE}/bigann-clustered/100000000/runbook-bigann-100M.yaml/ \
    ${BASE}/bigann-clustered/runbook-bigann-100M.yaml \
    10 bigann100Mclustered_runbook_results_nprobe_with_theo_sweep.csv


    # --- shift ---
mpirun -mca btl_tcp_if_include br-flat-lan-1 -np 10 --rankfile rankfile.txt \
    taskset -c 0-31 ./release_l2/DistributedInsertBenchSweep \
    ${BASE}/bigann-shift/100M-bigann-shift.u8bin \
    ${BASE}/bigann-shift/query.public.10K.u8bin \
    ${BASE}/bigann-shift/gpann_partitions/bigann100Mshift.k\=10.GP \
    ${BASE}/bigann-shift/100000000/bigann-100M-shift_runbookfinal.yaml/ \
    ${BASE}/bigann-shift/bigann-100M-shift_runbookfinal.yaml \
    10 bigann100Mshift_runbook_results_nprobe_with_theo_sweep.csv

    # --- random ---
mpirun -mca btl_tcp_if_include br-flat-lan-1 -np 10 --rankfile rankfile.txt \
    taskset -c 0-31 ./release_l2/DistributedInsertBenchSweep \
    ${BASE}/bigann-random/bigann-100M-random.u8bin \
    ${BASE}/bigann-random/query.public.10K.u8bin \
    ${BASE}/bigann-random/gpann_partitions/bigann100Mrandom.k\=10.GP \
    ${BASE}/bigann-random/100000000/runbook-bigann-100M-random.yaml/ \
    ${BASE}/bigann-random/runbook-bigann-100M-random.yaml \
    10 bigann100Mrandom_runbook_results_nprobe_with_theo_sweep.csv
