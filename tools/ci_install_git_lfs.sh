#!/usr/bin/env bash
# Install Git LFS via conda and add to PATH for subsequent steps.
# Uses CONDA_ROOT and CONDA_ENV (set by workflow env).
# Append to GITHUB_PATH so checkout step finds git-lfs.
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-}"
CONDA_ENV="${CONDA_ENV:-stellcoilbench_ci}"
if [[ -z "$CONDA_ROOT" ]] || [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
  echo "CONDA_ROOT not set or conda not found; install git-lfs manually"
  exit 1
fi

"$CONDA_ROOT/bin/conda" install -y -n "$CONDA_ENV" -c conda-forge git-lfs
BIN_DIR="$CONDA_ROOT/envs/$CONDA_ENV/bin"
echo "$BIN_DIR" >> "${GITHUB_PATH:-/dev/null}"
"$BIN_DIR/git-lfs" install
echo "Git LFS installed"
