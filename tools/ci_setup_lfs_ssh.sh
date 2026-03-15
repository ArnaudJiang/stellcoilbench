#!/usr/bin/env bash
# Configure SSH so Git LFS can push to stellcoilbench-lfs using LFS_DEPLOY_KEY.
# When both DEPLOY_KEY and LFS_DEPLOY_KEY are set, creates a combined SSH config
# that overrides core.sshCommand (from actions/checkout) so github.com uses
# DEPLOY_KEY and github-lfs uses LFS_DEPLOY_KEY.
# Call before any git push that may include LFS content.
set -euo pipefail

if [[ -z "${LFS_DEPLOY_KEY:-}" ]]; then
  echo "LFS_DEPLOY_KEY not set; skipping LFS SSH setup (LFS pushes will fail if any)"
  exit 0
fi

TMP_DIR="${TMPDIR:-/tmp}"
SSH_DIR="${HOME:-/tmp}/.ssh"
SSH_CONFIG="${SSH_DIR}/config"

mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

# When both keys are set: create combined config, unset core.sshCommand,
# set GIT_SSH_COMMAND so both git and git-lfs use the right key per host.
if [[ -n "${DEPLOY_KEY:-}" ]] && [[ -n "${LFS_DEPLOY_KEY:-}" ]]; then
  MAIN_KEY_FILE="${TMP_DIR}/main_deploy_key"
  LFS_KEY_FILE="${TMP_DIR}/lfs_deploy_key"
  COMBINED_CONFIG="${TMP_DIR}/ssh_combined_config"

  echo "$DEPLOY_KEY" > "$MAIN_KEY_FILE"
  chmod 600 "$MAIN_KEY_FILE"
  echo "$LFS_DEPLOY_KEY" > "$LFS_KEY_FILE"
  chmod 600 "$LFS_KEY_FILE"

  cat > "$COMBINED_CONFIG" << EOF
Host github.com
  HostName github.com
  IdentityFile $MAIN_KEY_FILE
  IdentitiesOnly yes
Host github-lfs
  HostName github.com
  IdentityFile $LFS_KEY_FILE
  IdentitiesOnly yes
EOF
  chmod 600 "$COMBINED_CONFIG"

  # Unset core.sshCommand so Git falls back to our combined config
  git config --unset core.sshCommand 2>/dev/null || true

  # Persist GIT_SSH_COMMAND for subsequent steps in the same job
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "GIT_SSH_COMMAND=ssh -F $COMBINED_CONFIG -o StrictHostKeyChecking=accept-new" >> "$GITHUB_ENV"
  fi

  echo "LFS SSH configured (combined config: github.com + github-lfs)"
  exit 0
fi

# Fallback: only LFS_DEPLOY_KEY set — append to ~/.ssh/config
# Note: core.sshCommand from actions/checkout may still override; prefer passing both keys.
KEY_FILE="${TMP_DIR}/lfs_deploy_key"
echo "$LFS_DEPLOY_KEY" > "$KEY_FILE"
chmod 600 "$KEY_FILE"

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
