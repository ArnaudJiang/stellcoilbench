#!/usr/bin/env bash
# Submit SLURM job for autopilot cases (single node, parallel srun).
# Called from update-db-viper workflow run-autopilot-cases job.
#
# Expects env: SLURM_AUTOPILOT_CPUS_PER_TASK, SLURM_AUTOPILOT_MEM_PER_TASK
# Writes to GITHUB_OUTPUT: slurm_job_id, log_dir, ran_cases
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
LOG_DIR="$WORKSPACE/slurm_logs"
mkdir -p "$LOG_DIR"

# Collect pending cases (cap at 299 per refactor plan)
shopt -s nullglob
PENDING_FILES=(cases/pending/*.json)
COUNT=0
> "$WORKSPACE/slurm_autopilot_list.txt"
AUTOPILOT_CAP="${SLURM_AUTOPILOT_CAP:-299}"
for f in "${PENDING_FILES[@]}"; do
  if [ "$COUNT" -ge "$AUTOPILOT_CAP" ]; then break; fi
  echo "$f" >> "$WORKSPACE/slurm_autopilot_list.txt"
  COUNT=$((COUNT + 1))
done

if [ "$COUNT" -eq 0 ]; then
  echo "No pending autopilot cases"
  [ -n "${GITHUB_OUTPUT:-}" ] && echo "ran_cases=false" >> "$GITHUB_OUTPUT"
  exit 0
fi

# Compute max timeout from per-case resource.timeout_minutes
MAX_TIMEOUT_MIN=60
while IFS= read -r CASE_FILE; do
  T=$(python3 -c "
import json
try:
    d = json.load(open('$CASE_FILE'))
    print(d.get('resource', {}).get('timeout_minutes', 60))
except: print(60)" 2>/dev/null || echo 60)
  if [ "$T" -gt "$MAX_TIMEOUT_MIN" ]; then
    MAX_TIMEOUT_MIN=$T
  fi
done < "$WORKSPACE/slurm_autopilot_list.txt"

# Add 5 min buffer for module loads and teardown
MAX_TIMEOUT_MIN=$((MAX_TIMEOUT_MIN + 5))
SLURM_TIME=$(printf "%02d:%02d:00" $((MAX_TIMEOUT_MIN / 60)) $((MAX_TIMEOUT_MIN % 60)))

# Memory: use SLURM_AUTOPILOT_MEM if set, else MEM_PER_TASK * N
MEM_PER_TASK="${SLURM_AUTOPILOT_MEM_PER_TASK:-30000M}"
if [ -n "${SLURM_AUTOPILOT_MEM:-}" ]; then
  AUTOPILOT_MEM="$SLURM_AUTOPILOT_MEM"
else
  # Parse number from MEM_PER_TASK (e.g. 30000M -> 30000)
  MEM_NUM=$(echo "$MEM_PER_TASK" | sed 's/[^0-9]//g')
  TOTAL_MEM=$((MEM_NUM * COUNT))
  AUTOPILOT_MEM="${TOTAL_MEM}M"
fi

CPUS_PER_TASK="${SLURM_AUTOPILOT_CPUS_PER_TASK:-1}"

echo "Submitting SLURM job for $COUNT autopilot case(s) on 1 node (timeout: ${SLURM_TIME})"

cat > /tmp/stellcoilbench_autopilot.sh << 'SLURM_SCRIPT'
#!/bin/bash
#SBATCH -J stellcoilbench-autopilot
#SBATCH -o __LOG_DIR__/autopilot_%j.out
#SBATCH -e __LOG_DIR__/autopilot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=__AUTOPILOT_NTASKS__
#SBATCH --cpus-per-task=__AUTOPILOT_CPUS_PER_TASK__
#SBATCH --mem=__AUTOPILOT_MEM__
#SBATCH --time=__TIME__
#SBATCH --mail-type=none

cd __WORKSPACE__
export STELLCOILBENCH_CI_VERBOSE=1
source __WORKSPACE__/tools/ci_slurm_env.sh

mapfile -t FILES < __WORKSPACE__/slurm_autopilot_list.txt
N=${#FILES[@]}

echo "Running $N cases on one node (parallel srun)"

for ((i=0; i<N; i++)); do
  (
    case_file="${FILES[$i]}"
    CASE_ID=$(basename "$case_file" .json)
    srun -n 1 -N 1 --exclusive stellcoilbench run-ci-case "$case_file" \
      --output-dir cases/done \
      --policy policy/proposer_policy.yaml \
      > "__LOG_DIR__/autopilot_${SLURM_JOB_ID}_${i}.out" 2> "__LOG_DIR__/autopilot_${SLURM_JOB_ID}_${i}.err"
    EXIT=$?
    rm -f "$case_file"
    echo $EXIT > "__LOG_DIR__/autopilot_${SLURM_JOB_ID}_${i}.exit"
    if [ $EXIT -eq 0 ]; then
      echo "[OK] $CASE_ID"
    else
      echo "[FAIL] $CASE_ID (exit $EXIT)"
    fi
    exit $EXIT
  ) &
done
wait

FAILED=0
for ((i=0; i<N; i++)); do
  if [ -f "__LOG_DIR__/autopilot_${SLURM_JOB_ID}_${i}.exit" ]; then
    code=$(cat "__LOG_DIR__/autopilot_${SLURM_JOB_ID}_${i}.exit")
    [ "$code" != "0" ] && FAILED=$((FAILED + 1))
  fi
done
exit $FAILED
SLURM_SCRIPT

# Substitute placeholders
if sed --version 2>/dev/null | grep -q GNU; then
  sed -i "s|__LOG_DIR__|${LOG_DIR}|g;
          s|__AUTOPILOT_NTASKS__|${COUNT}|g;
          s|__AUTOPILOT_CPUS_PER_TASK__|${CPUS_PER_TASK}|g;
          s|__AUTOPILOT_MEM__|${AUTOPILOT_MEM}|g;
          s|__TIME__|${SLURM_TIME}|g;
          s|__WORKSPACE__|${WORKSPACE}|g" /tmp/stellcoilbench_autopilot.sh
else
  sed -i.bak "s|__LOG_DIR__|${LOG_DIR}|g;
              s|__AUTOPILOT_NTASKS__|${COUNT}|g;
              s|__AUTOPILOT_CPUS_PER_TASK__|${CPUS_PER_TASK}|g;
              s|__AUTOPILOT_MEM__|${AUTOPILOT_MEM}|g;
              s|__TIME__|${SLURM_TIME}|g;
              s|__WORKSPACE__|${WORKSPACE}|g" /tmp/stellcoilbench_autopilot.sh
  rm -f /tmp/stellcoilbench_autopilot.sh.bak
fi

echo "=== Generated SLURM autopilot script ==="
cat /tmp/stellcoilbench_autopilot.sh
echo "=============================="

JOB_ID=$(sbatch --parsable /tmp/stellcoilbench_autopilot.sh)
echo "slurm_job_id=$JOB_ID" >> "${GITHUB_OUTPUT:-/dev/null}"
echo "log_dir=$LOG_DIR" >> "${GITHUB_OUTPUT:-/dev/null}"
echo "ran_cases=true" >> "${GITHUB_OUTPUT:-/dev/null}"
echo "Submitted SLURM autopilot job: $JOB_ID"
