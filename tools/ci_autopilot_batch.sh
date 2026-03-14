#!/usr/bin/env bash
# Run autopilot cases from cases/pending/*.json in parallel with per-case timeouts.
# Called from update-db-self-hosted workflow (run_autopilot_cases job).
#
# Expects: conda env activated, stellcoilbench installed, SIMPLE_EXECUTABLE (optional)
# Writes ran_cases to GITHUB_OUTPUT if GITHUB_OUTPUT is set.
# Respects PAUSE_AUTORUN: if present, exits without running (halts the autopilot loop).
set -euo pipefail

if [ -f "PAUSE_AUTORUN" ]; then
  echo "PAUSE_AUTORUN present; skipping autopilot batch."
  exit 0
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

pip install -e . --no-cache-dir
python -c "import stellcoilbench.reactor_scale; print('stellcoilbench install OK')"

if [ -n "${SIMPLE_EXECUTABLE:-}" ] && [ -f "${SIMPLE_EXECUTABLE}" ]; then
  cp "$SIMPLE_EXECUTABLE" ./simple.x
fi

python -m tools.ci_autopilot_runner

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "ran_cases=true" >> "$GITHUB_OUTPUT"
fi
