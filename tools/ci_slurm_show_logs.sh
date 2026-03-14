#!/usr/bin/env bash
# Dump SLURM log files on workflow failure.
# Called from update-db-viper workflow with if: failure()
#
# Expects env: LOG_DIR
set -euo pipefail

LOG_DIR="${LOG_DIR:-}"
if [ -z "$LOG_DIR" ] || [ ! -d "$LOG_DIR" ]; then
  echo "LOG_DIR not set or not a directory, skipping log dump"
  exit 0
fi

echo "=== ALL SLURM LOG FILES ==="
for f in "$LOG_DIR"/*.out "$LOG_DIR"/*.err; do
  [ -f "$f" ] && [ -s "$f" ] && echo "=== $(basename "$f") ===" && cat "$f"
done
