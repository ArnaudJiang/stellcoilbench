#!/usr/bin/env bash
# Commit leaderboard update. Called from update-db-self-hosted update_leaderboard job.
set -euo pipefail

git config --local user.email "action@github.com"
git config --local user.name "$GITHUB_ACTOR"
git add -f docs/leaderboard.rst \
         docs/leaderboard/metric_definitions.rst \
         docs/leaderboard/surface_specific.rst \
         docs/leaderboard/reactor_scale.rst \
         docs/leaderboard.json 2>/dev/null || true
if git diff --staged --quiet; then
  echo "No leaderboard changes to commit"
else
  git commit -m "chore: update StellCoilBench leaderboard"
  bash tools/ci_pull.sh
  git push origin main
  echo "Leaderboard updated"
fi
