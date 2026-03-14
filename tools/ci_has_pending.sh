#!/usr/bin/env bash
# Check for pending autopilot cases. Writes has_pending to GITHUB_OUTPUT.
# Called from update-db-self-hosted workflow.
set -euo pipefail

shopt -s nullglob
files=(cases/pending/*.json)
if [ ${#files[@]} -eq 0 ]; then
  echo "No pending autopilot cases."
  [ -n "${GITHUB_OUTPUT:-}" ] && echo "has_pending=false" >> "$GITHUB_OUTPUT"
else
  echo "Found ${#files[@]} pending autopilot case(s)."
  [ -n "${GITHUB_OUTPUT:-}" ] && echo "has_pending=true" >> "$GITHUB_OUTPUT"
fi
