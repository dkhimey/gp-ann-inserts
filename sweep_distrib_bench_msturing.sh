#!/bin/bash

BASE=/dataset/big-ann-benchmarks/data

# --- clustered ---
mpirun -mca btl_tcp_if_include br-flat-lan-1 -np 10 --rankfile rankfile.txt \
    taskset -c 0-31 ./release_l2/DistributedInsertBenchSweep \
    ${BASE}/MSTuring-100M-clustered/100M-msturing-clustered.fbin \
    ${BASE}/MSTuring-100M-clustered/testQuery10K.fbin \
    ${BASE}/MSTuring-100M-clustered/gpann_partitions/msturing100Mclustered.k\=10.GP \
    ${BASE}/MSTuring-100M-clustered/100000000/msturing-100M-clustered_runbookfinal.yaml \
    ${BASE}/MSTuring-100M-clustered/msturing-100M-clustered_runbookfinal.yaml \
    10 msturing100Mclustered_runbook_results_nprobe_with_theo_sweep.csv


    # --- shift ---
mpirun -mca btl_tcp_if_include br-flat-lan-1 -np 10 --rankfile rankfile.txt \
    taskset -c 0-31 ./release_l2/DistributedInsertBenchSweep \
    ${BASE}/MSTuring-100M-shift/100M-msturing-shift.fbin \
    ${BASE}/MSTuring-100M-shift/testQuery10K.fbin \
    ${BASE}/MSTuring-100M-shift/gpann_partitions/msturing100Mshift.k\=10.GP \
    ${BASE}/MSTuring-100M-shift/100000000/msturing-100M-shift_runbookfinal.yaml \
    ${BASE}/MSTuring-100M-shift/msturing-100M-shift_runbookfinal.yaml \
    10 msturing100Mshift_runbook_results_nprobe_with_theo_sweep.csv

    # --- random ---
mpirun -mca btl_tcp_if_include br-flat-lan-1 -np 10 --rankfile rankfile.txt \
    taskset -c 0-31 ./release_l2/DistributedInsertBenchSweep \
    ${BASE}/MSTuring-100M-random/msturing-100M-random.fbin \
    ${BASE}/MSTuring-100M-random/testQuery10K.fbin \
    ${BASE}/MSTuring-100M-random/gpann_partitions/msturing100Mrandom.k\=10.GP \
    ${BASE}/MSTuring-100M-random/100000000/runbook-msturing-100M-random.yaml \
    ${BASE}/MSTuring-100M-random/runbook-msturing-100M-random.yaml \
    10 msturing100Mrandom_runbook_results_nprobe_with_theo_sweep.csv

