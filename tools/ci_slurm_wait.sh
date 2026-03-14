#!/usr/bin/env bash
# Wait for SLURM job to complete. Polls squeue every 30s.
# Called from update-db-viper workflow after sbatch.
#
# Expects env: JOB_ID (or $1)
# Optional env: LOG_DIR (for future use; ci_slurm_show_logs dumps on failure)
set -euo pipefail

JOB_ID="${JOB_ID:-$1}"
if [ -z "$JOB_ID" ]; then
  echo "ERROR: JOB_ID not set and no argument provided"
  exit 1
fi

echo "Waiting for SLURM job $JOB_ID ..."

while squeue -j "$JOB_ID" -h 2>/dev/null | grep -q .; do
  RUNNING=$(squeue -j "$JOB_ID" -h -t RUNNING 2>/dev/null | wc -l | tr -d ' ')
  PENDING_Q=$(squeue -j "$JOB_ID" -h -t PENDING 2>/dev/null | wc -l | tr -d ' ')
  echo "[$(date '+%H:%M:%S')] SLURM job $JOB_ID: $RUNNING running, $PENDING_Q pending"
  sleep 30
done

echo "All SLURM tasks completed."
echo "=== SLURM job accounting ==="
sacct -j "$JOB_ID" --format=JobID%20,State%12,ExitCode%8,Elapsed%12,MaxRSS%12 -n

# Dump failed task logs (matches original monolithic workflow)
FAILED_COUNT=$(sacct -j "$JOB_ID" --format=State -n -X 2>/dev/null | grep -c -E "FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY" || true)
if [ "$FAILED_COUNT" -gt 0 ] && [ -n "${LOG_DIR:-}" ]; then
  echo "WARNING: $FAILED_COUNT SLURM task(s) failed"
  echo "=== Failed task logs ==="
  # Benchmark array jobs: parse task_id from sacct JobID (e.g. 12345_0 -> 0)
  sacct -j "$JOB_ID" --format=JobID%20,State%12 -n -X 2>/dev/null | while read -r jobid state; do
    state=$(echo "$state" | xargs)
    if echo "$state" | grep -qE "FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY"; then
      task_id=$(echo "$jobid" | sed -n 's/.*_\([0-9]*\)$/\1/p')
      if [ -n "$task_id" ]; then
        echo "--- Task $task_id ($state) stderr ---"
        cat "$LOG_DIR/case_${JOB_ID}_${task_id}.err" 2>/dev/null || echo "(no err file)"
        echo "--- Task $task_id stdout ---"
        cat "$LOG_DIR/case_${JOB_ID}_${task_id}.out" 2>/dev/null || echo "(no out file)"
      fi
    fi
  done
  # Autopilot (single job): dump autopilot_${JOB_ID}_* logs (err full, out tail)
  for f in "$LOG_DIR"/autopilot_${JOB_ID}_*.err; do
    if [ -f "$f" ] && [ -s "$f" ]; then
      echo "--- $(basename "$f") ---"
      cat "$f"
    fi
  done
  for f in "$LOG_DIR"/autopilot_${JOB_ID}_*.out; do
    if [ -f "$f" ] && [ -s "$f" ]; then
      echo "--- $(basename "$f") (last 30 lines) ---"
      tail -30 "$f"
    fi
  done
fi
