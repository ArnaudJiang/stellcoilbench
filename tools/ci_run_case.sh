#!/bin/bash
# Run a single StellCoilBench case for CI parallel execution.
#
# Usage: tools/ci_run_case.sh CASE_FILE WORKSPACE [CONDA_ENV]
#   or:  CASE_FILE=... WORKSPACE=... CONDA_ENV=... tools/ci_run_case.sh
#
# Args (if not in env):
#   $1 - CASE_FILE: path to case YAML
#   $2 - WORKSPACE: repo root (cwd for stellcoilbench)
#   $3 - CONDA_ENV (optional): conda env name (default: use CONDA_ENV from env)
set -euo pipefail

CASE_FILE="${CASE_FILE:-$1}"
WORKSPACE="${WORKSPACE:-$2}"
CONDA_ENV="${CONDA_ENV:-${3:-}}"

if [ -z "$CASE_FILE" ] || [ -z "$WORKSPACE" ]; then
  echo "Usage: tools/ci_run_case.sh CASE_FILE WORKSPACE [CONDA_ENV]"
  exit 1
fi

CASE_NAME=$(basename "$CASE_FILE" .yaml)
LOG_FILE="${LOG_DIR:-/tmp/stellcoilbench_logs}/${CASE_NAME}.log"

log() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] [$CASE_NAME] $1" | tee -a "$LOG_FILE"; }

log "Starting..."

# Source conda (CONDA_ROOT first for self-hosted; then common paths)
CONDA_FOUND=
if [ -n "${CONDA_ROOT:-}" ] && [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  CONDA_FOUND=1
fi
if [ -z "$CONDA_FOUND" ]; then
  for p in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/opt/miniconda3" "$HOME/opt/anaconda3" \
            "/opt/homebrew/Caskroom/miniconda/base" "/opt/homebrew/Caskroom/anaconda/base" \
            "/usr/local/miniconda3" "/usr/local/anaconda3"; do
    if [ -f "$p/etc/profile.d/conda.sh" ]; then
      source "$p/etc/profile.d/conda.sh"
      CONDA_FOUND=1
      break
    fi
  done
fi
[ -z "$CONDA_FOUND" ] && command -v conda >/dev/null 2>&1 && eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV" || { log "ERROR: conda failed"; exit 1; }

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

cd "$WORKSPACE"
log "Running stellcoilbench..."

python -m stellcoilbench.cli submit-case "$CASE_FILE" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
  log "SUCCESS"
else
  log "FAILED (exit code $EXIT_CODE)"
  echo "=== FULL LOG FOR $CASE_FILE ===" && cat "$LOG_FILE"
fi
exit $EXIT_CODE
