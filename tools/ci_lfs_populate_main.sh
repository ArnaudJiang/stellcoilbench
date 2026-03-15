#!/usr/bin/env bash
# Populate main repo's LFS storage with objects from stellcoilbench-lfs.
#
# Use when: reverting from stellcoilbench-lfs back to main. Objects that exist
# only in stellcoilbench-lfs (pushed after the dual-repo config) must be
# copied to main before switching .lfsconfig.
#
# PREREQUISITES:
#   1. Full clone of stellcoilbench (code + LFS pointers)
#   2. Credentials for stellcoilbench-lfs (fetch) and main (push):
#      - LFS_DEPLOY_KEY or GITHUB_TOKEN for fetch from stellcoilbench-lfs
#      - DEPLOY_KEY or GITHUB_TOKEN for push to main
#   3. Write access to main (deploy key or token with repo write)
#
# USAGE:
#   export LFS_DEPLOY_KEY="$(cat ~/.ssh/stellcoilbench_lfs_deploy)"
#   export DEPLOY_KEY="$(cat ~/.ssh/stellcoilbench_deploy)"
#   bash tools/ci_lfs_populate_main.sh
#
# After running, verify: git lfs fsck (with main as LFS source)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "=== Pre-flight ==="
command -v git-lfs >/dev/null 2>&1 || { echo "Install git-lfs: brew install git-lfs && git lfs install"; exit 1; }
git lfs version

# Derive main repo LFS URL from origin
ORIGIN_URL=$(git remote get-url origin 2>/dev/null || true)
if [[ -z "$ORIGIN_URL" ]]; then
  echo "ERROR: No remote 'origin'. Add it with: git remote add origin https://github.com/akaptano/stellcoilbench.git"
  exit 1
fi
if [[ "$ORIGIN_URL" =~ git@github\.com:([^/]+)/([^/.]+) ]]; then
  MAIN_LFS_URL="https://github.com/${BASH_REMATCH[1]}/${BASH_REMATCH[2]}.git/info/lfs"
elif [[ "$ORIGIN_URL" =~ https://github\.com/([^/]+)/([^/.]+) ]]; then
  MAIN_LFS_URL="https://github.com/${BASH_REMATCH[1]}/${BASH_REMATCH[2]}.git/info/lfs"
else
  echo "ERROR: Could not derive LFS URL from origin: $ORIGIN_URL"
  exit 1
fi
echo "Main repo LFS URL (push target): $MAIN_LFS_URL"

LFS_CONFIG=".lfsconfig"
BACKUP="${LFS_CONFIG}.bak.$$"
# Create .lfsconfig if missing (e.g. after it was removed)
if [[ ! -f "$LFS_CONFIG" ]]; then
  touch "$LFS_CONFIG"
fi
cp "$LFS_CONFIG" "$BACKUP"
trap "mv -f '$BACKUP' '$LFS_CONFIG'" EXIT

# Step 1: Point fetch at stellcoilbench-lfs to pull objects
echo ""
echo "=== Step 1: Fetch LFS objects from stellcoilbench-lfs ==="
LFS_REPO_URL="https://github.com/akaptano/stellcoilbench-lfs.git/info/lfs"
cat > "$LFS_CONFIG" << EOF
[lfs]
	url = $LFS_REPO_URL
	skipdownloaderrors = true
EOF

# Fetch uses HTTPS; GITHUB_TOKEN or credential helper provides auth for stellcoilbench-lfs
git lfs fetch origin
git lfs checkout

# Step 2: Point LFS at main for push
echo ""
echo "=== Step 2: Point LFS at main and push all objects ==="
cat > "$LFS_CONFIG" << EOF
[lfs]
	url = $MAIN_LFS_URL
	skipdownloaderrors = true
EOF

# Restore final config (Phase 2 style) before we clear trap
trap - EXIT
rm -f "$BACKUP"

# Push all local LFS objects to main
# DEPLOY_KEY (or GITHUB_TOKEN) should be configured for main repo access
git lfs push origin --all

echo ""
echo "=== Done ==="
echo "Main repo LFS populated. Verify with:"
echo "  git lfs fsck"
echo "  git clone https://github.com/akaptano/stellcoilbench.git && cd stellcoilbench && git lfs pull"
