#!/usr/bin/env bash
# Run propose_batch with CI guards (PAUSE_AUTORUN, batch barrier).
# Self-hosted: CI_PROPOSE_USE_LLM=1 (default), batch-size 10.
# Viper: CI_PROPOSE_BATCH_SIZE=299, CI_PROPOSE_USE_LLM=0 (GA mode).
# Expects KB_LLM_PROVIDER, KB_LLM_MODEL, ANTHROPIC_API_KEY when USE_LLM=1.
set -euo pipefail

if [ -f PAUSE_AUTORUN ]; then
  echo "PAUSE_AUTORUN file exists — skipping proposal."
  exit 0
fi

shopt -s nullglob
pending=(cases/pending/*.json)
if [ ${#pending[@]} -gt 0 ]; then
  echo "Pending directory not empty (${#pending[@]} cases). Skipping."
  exit 0
fi

BATCH_SIZE="${CI_PROPOSE_BATCH_SIZE:-10}"
USE_LLM="${CI_PROPOSE_USE_LLM:-1}"

LLM_ARGS=()
if [ "$USE_LLM" = "1" ]; then
  pip install anthropic 2>/dev/null || echo "WARNING: failed to install anthropic"
  LLM_ARGS=(--llm)
  # Only add --verify-llm when explicitly requested (CI_PROPOSE_VERIFY_LLM=1).
  # Without it, propose_batch falls back to GA when the LLM API key is missing.
  if [ "${CI_PROPOSE_VERIFY_LLM:-0}" = "1" ]; then
    LLM_ARGS+=(--verify-llm)
  fi
fi

python -m tools.propose_batch \
  --batch-size "$BATCH_SIZE" \
  --submissions-dir submissions \
  --failures-file policy/autopilot_failures.json \
  --pending-dir cases/pending \
  --policy policy/proposer_policy.yaml \
  "${LLM_ARGS[@]}"
