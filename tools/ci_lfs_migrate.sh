#!/usr/bin/env bash
# Migrate existing submissions/* (vtu, vts, png, pdf) to Git LFS.
#
# PREREQUISITES (do before running):
#   1. Create https://github.com/akaptano/stellcoilbench-lfs (empty repo, same org/account)
#   2. Ensure DEPLOY_KEY or CI credentials have write access to stellcoilbench-lfs
#   3. Full backup of submissions/ and repo
#   4. Coordinate with contributors — history rewrite will change all commit SHAs
#
# USAGE:
#   git lfs install
#   bash tools/ci_lfs_migrate.sh
#   git push --force-with-lease origin main
#
# After migration, forks cannot push to upstream LFS (they lack write access to stellcoilbench-lfs).
# Fork owners must add .lfsconfig pointing to their own LFS backend. See docs/forking.md.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "=== Pre-flight ==="
command -v git-lfs >/dev/null 2>&1 || { echo "Install git-lfs: brew install git-lfs && git lfs install"; exit 1; }
git lfs version
git status --short | head -5 || true
echo ""

echo "=== Migrating submissions/*.vtu, *.vts, *.png, *.pdf to LFS ==="
git lfs migrate import \
  --include="submissions/**/*.vtu,submissions/**/*.vts,submissions/**/*.png,submissions/**/*.pdf" \
  --everything

echo ""
echo "=== Migration complete ==="
echo "Run: git push --force-with-lease origin main"
echo "Verify: git lfs ls-files | head"
echo "Fresh clone test: git clone ... && cd ... && git lfs pull"
