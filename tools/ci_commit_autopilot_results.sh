#!/usr/bin/env bash
# Commit autopilot results and update leaderboard.
# Handles merge conflicts by keeping ours for submissions/ and autopilot_failures.json.
# Called from update-db-self-hosted workflow.
set -euo pipefail

# --- Git identity ---
git config --local user.email "action@github.com"
git config --local user.name "ci-autopilot[bot]"

# --- Rebuild leaderboard ---
echo "Rebuilding leaderboard from submissions/ ..."
stellcoilbench update-db 2>&1 || echo "WARNING: update-db failed, leaderboard may be stale"

# --- Stage changes ---
git add -f cases/done/ 2>/dev/null || true
git add -u cases/pending/ 2>/dev/null || true
git add -f submissions/ 2>/dev/null || true
git add -f policy/autopilot_failures.json 2>/dev/null || true
git add docs/leaderboard.rst 2>/dev/null || true

if git diff --staged --quiet; then
  echo "No autopilot results to commit"
  exit 0
fi

# --- Commit and push (with merge conflict resolution) ---
git commit -m "ci: autopilot batch results + leaderboard update"
git fetch origin main
if ! git merge origin/main --no-edit 2>/dev/null; then
  echo "Merge conflicts detected — resolving..."
  git checkout --ours cases/done/ 2>/dev/null || true
  git checkout --ours submissions/ 2>/dev/null || true
  git checkout --ours policy/autopilot_failures.json 2>/dev/null || true
  git checkout --theirs cases/pending/ 2>/dev/null || true
  git checkout --theirs docs/leaderboard.rst 2>/dev/null || true
  git add -f cases/done/ cases/pending/ submissions/ policy/autopilot_failures.json
  stellcoilbench update-db 2>&1 || true
  git add docs/leaderboard.rst
  git commit -m "ci: autopilot batch results + leaderboard update (merge)" --no-edit
fi
git push origin main
echo "Autopilot results and leaderboard committed"
