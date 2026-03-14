#!/usr/bin/env bash
# Commit CI-generated submissions. Called from update-db-self-hosted run_cases job.
set -euo pipefail

git config --local user.email "action@github.com"
git config --local user.name "$GITHUB_ACTOR"
git add -f submissions/ 2>/dev/null || true
if git diff --staged --quiet; then
  echo "No new submissions to commit"
else
  COMMIT_MSG="${CI_COMMIT_MSG:-chore: add CI-generated submissions (parallel run)}"
  git commit -m "$COMMIT_MSG"
  bash tools/ci_pull.sh
  git push origin main
  echo "Submissions committed"
fi
