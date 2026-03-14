#!/usr/bin/env bash
# Run benchmark cases in parallel (GNU parallel or xargs).
# Called from update-db-self-hosted workflow.
#
# Expects env: CASES_JSON, PARALLEL_JOBS, CONDA_ENV, SIMPLE_EXECUTABLE (optional)
set -euo pipefail

PARALLEL_JOBS="${PARALLEL_JOBS:-10}"

echo "=========================================="
echo "Running cases in parallel (max $PARALLEL_JOBS jobs)"
echo "Cases: $CASES_JSON"
echo "=========================================="

if [ -n "${SIMPLE_EXECUTABLE:-}" ] && [ -f "${SIMPLE_EXECUTABLE}" ]; then
  cp "$SIMPLE_EXECUTABLE" ./simple.x
  echo "Copied simple.x from $SIMPLE_EXECUTABLE"
fi

CASE_LIST=$(echo "$CASES_JSON" | python3 -c "import sys,json; print('\n'.join(json.load(sys.stdin)))")
mkdir -p /tmp/stellcoilbench_logs
REPO_ROOT="$PWD"

# Use GNU parallel if available (better progress and joblog); otherwise fall back to xargs.
RUNNER="CONDA_ENV=${CONDA_ENV:-} tools/ci_run_case.sh"
if command -v parallel &> /dev/null; then
  echo "Using GNU parallel"
  echo "$CASE_LIST" | parallel --will-cite -j "$PARALLEL_JOBS" --halt never \
    --ungroup --joblog /tmp/parallel.log \
    "$RUNNER {} $REPO_ROOT"
  echo "=== Parallel execution summary ===" && cat /tmp/parallel.log || true
else
  echo "GNU parallel not found, using xargs"
  echo "$CASE_LIST" | xargs -P "$PARALLEL_JOBS" -I {} bash -c "$RUNNER '{}' '$REPO_ROOT'"
fi

echo "=== All cases completed ==="
echo "=== LOG FILES ===" && ls -la /tmp/stellcoilbench_logs/ 2>/dev/null || true
