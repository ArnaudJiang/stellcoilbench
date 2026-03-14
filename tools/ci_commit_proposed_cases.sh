#!/usr/bin/env bash
# Commit proposed autopilot cases. Called from update-db-self-hosted propose_autopilot_batch job.
set -euo pipefail

git config --local user.email "action@github.com"
git config --local user.name "ci-autopilot[bot]"
git add cases/pending/ 2>/dev/null || true
git add -f policy/llm_reasoning_history.json 2>/dev/null || true
git add -f policy/autopilot_failures.json 2>/dev/null || true
git add PAUSE_AUTORUN 2>/dev/null || true
if git diff --staged --quiet; then
  echo "No new cases proposed"
else
  git commit -m "ci: propose next autopilot batch"
  bash tools/ci_pull.sh
  git push origin main
  echo "Proposed cases committed"
fi
