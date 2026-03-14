#!/usr/bin/env bash
# Print all .log files in the given directory. Used for CI failure diagnostics.
# Usage: bash tools/ci_show_logs.sh /tmp/stellcoilbench_logs
#        bash tools/ci_show_logs.sh /tmp/autopilot_logs
set -euo pipefail

LOG_DIR="${1:-/tmp/stellcoilbench_logs}"
echo "=== LOG FILES ($LOG_DIR) ==="
for f in "$LOG_DIR"/*.log; do
  [ -f "$f" ] && echo "=== $f ===" && cat "$f"
done
