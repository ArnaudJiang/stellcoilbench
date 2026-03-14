#!/usr/bin/env bash
# Configure SSH so Git LFS can push to stellcoilbench-lfs using LFS_DEPLOY_KEY.
# Uses host alias github-lfs to keep main repo (github.com) on the existing deploy key.
# Call before any git push that may include LFS content.
set -euo pipefail

if [[ -z "${LFS_DEPLOY_KEY:-}" ]]; then
  echo "LFS_DEPLOY_KEY not set; skipping LFS SSH setup (LFS pushes will fail if any)"
  exit 0
fi

TMP_DIR="${TMPDIR:-/tmp}"
KEY_FILE="${TMP_DIR}/lfs_deploy_key"
SSH_DIR="${HOME:-/tmp}/.ssh"
SSH_CONFIG="${SSH_DIR}/config"

mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

echo "$LFS_DEPLOY_KEY" > "$KEY_FILE"
chmod 600 "$KEY_FILE"

# Append github-lfs host alias if not already present
if ! grep -q '^Host github-lfs' "$SSH_CONFIG" 2>/dev/null; then
  {
    echo ""
    echo "Host github-lfs"
    echo "  HostName github.com"
    echo "  IdentityFile $KEY_FILE"
    echo "  IdentitiesOnly yes"
  } >> "$SSH_CONFIG"
  chmod 600 "$SSH_CONFIG"
fi

# Validate connectivity (optional; may fail on first run before key is approved)
if ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new git@github-lfs 2>/dev/null; then
  :  # success banner
else
  :  # still proceed; push may work depending on host key setup
fi

echo "LFS SSH configured (github-lfs -> stellcoilbench-lfs)"
