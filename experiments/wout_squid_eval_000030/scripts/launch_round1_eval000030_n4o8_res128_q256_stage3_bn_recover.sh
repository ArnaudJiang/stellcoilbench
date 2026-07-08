#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"

POLICY="${POLICY:-policy/squid_eval000030_round1_n4o8_res128_q256_stage3_bn_recover_policy.yaml}"
SURFACE="${SURFACE:-plasma_surfaces/wout_squid_eval_000030.nc}"
RESULTS_DIR="${RESULTS_DIR:-experiments/wout_squid_eval_000030/raw/results/round1_n4o8_res128_q256_scan/stage3_bn_recover}"
SURFACE_RESOLUTION="${SURFACE_RESOLUTION:-128}"
MAX_PARALLEL="${MAX_PARALLEL:-64}"
DATA_TWIN_CAMPAIGN="${DATA_TWIN_CAMPAIGN:?Set DATA_TWIN_CAMPAIGN, or use scripts/optimization_workflow.py prepare/launch.}"

conda run -n stellcoilbench_vmec python scripts/run_simsopt_batch.py \
  --policy "${POLICY}" \
  --surface "${SURFACE}" \
  --results-dir "${RESULTS_DIR}" \
  --backend simsopt \
  --surface-resolution "${SURFACE_RESOLUTION}" \
  --max-parallel-simsopt "${MAX_PARALLEL}" \
  --data-twin-campaign "${DATA_TWIN_CAMPAIGN}" \
  --skip-existing
