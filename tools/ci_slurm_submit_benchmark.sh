#!/usr/bin/env bash
# Submit SLURM job array for benchmark cases.
# Called from update-db-viper workflow run-cases job.
#
# Expects env: CASES_JSON, SLURM_CPUS, SLURM_MEM, SLURM_TIME_CASES
# Writes to GITHUB_OUTPUT: slurm_job_id, log_dir
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
LOG_DIR="$WORKSPACE/slurm_logs"
mkdir -p "$LOG_DIR"

# Write case list (one file per line)
echo "$CASES_JSON" | python3 -c "
import sys, json
for c in json.load(sys.stdin):
    print(c)
" > "$WORKSPACE/slurm_case_list.txt"

NUM_CASES=$(wc -l < "$WORKSPACE/slurm_case_list.txt" | tr -d ' ')
ARRAY_MAX=$((NUM_CASES - 1))

echo "Submitting SLURM array for $NUM_CASES case(s) (array 0-${ARRAY_MAX})"

# Generate the SLURM batch script (quoted heredoc preserves $ for SLURM)
cat > /tmp/stellcoilbench_cases.sh << 'SLURM_SCRIPT'
#!/bin/bash
#SBATCH -J stellcoilbench-cases
#SBATCH -o __LOG_DIR__/case_%A_%a.out
#SBATCH -e __LOG_DIR__/case_%A_%a.err
#SBATCH --array=0-__ARRAY_MAX__
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=__CPUS__
#SBATCH --mem=__MEM__
#SBATCH --time=__TIME__
#SBATCH --mail-type=none

cd __WORKSPACE__
source __WORKSPACE__/tools/ci_slurm_env.sh

mapfile -t FILES < __WORKSPACE__/slurm_case_list.txt
N=${#FILES[@]}

if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then
  echo "Array task $SLURM_ARRAY_TASK_ID >= number of cases ($N), skipping."
  exit 0
fi

case_file="${FILES[$SLURM_ARRAY_TASK_ID]}"
echo "Running case: $case_file"
srun stellcoilbench submit-case "$case_file"
SLURM_SCRIPT

# Substitute placeholders with actual values
CPUS="${SLURM_CPUS:-8}"
MEM="${SLURM_MEM:-10000M}"
TIME="${SLURM_TIME_CASES:-00:30:00}"
# Portable sed in-place (Linux sed -i vs macOS sed -i '')
if sed --version 2>/dev/null | grep -q GNU; then
  sed -i "s|__LOG_DIR__|${LOG_DIR}|g; s|__ARRAY_MAX__|${ARRAY_MAX}|g; s|__CPUS__|${CPUS}|g; s|__MEM__|${MEM}|g; s|__TIME__|${TIME}|g; s|__WORKSPACE__|${WORKSPACE}|g" /tmp/stellcoilbench_cases.sh
else
  sed -i.bak "s|__LOG_DIR__|${LOG_DIR}|g; s|__ARRAY_MAX__|${ARRAY_MAX}|g; s|__CPUS__|${CPUS}|g; s|__MEM__|${MEM}|g; s|__TIME__|${TIME}|g; s|__WORKSPACE__|${WORKSPACE}|g" /tmp/stellcoilbench_cases.sh
  rm -f /tmp/stellcoilbench_cases.sh.bak
fi

echo "=== Generated SLURM script ==="
cat /tmp/stellcoilbench_cases.sh
echo "=============================="

JOB_ID=$(sbatch --parsable /tmp/stellcoilbench_cases.sh)
echo "slurm_job_id=$JOB_ID" >> "${GITHUB_OUTPUT:-/dev/null}"
echo "log_dir=$LOG_DIR" >> "${GITHUB_OUTPUT:-/dev/null}"
echo "Submitted SLURM job array: $JOB_ID"
