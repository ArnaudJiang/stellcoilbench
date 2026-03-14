"""Version and git information utilities for StellCoilBench.

Provides reproducibility tracking: stellcoilbench commit/branch,
simsopt version/commit/branch/remote.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

GIT_COMMAND_TIMEOUT = 5  # seconds


def _run_git_command(args: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    """Run a git command and return stdout on success.

    Parameters
    ----------
    args : list[str]
        Git command and arguments (e.g., ["git", "rev-parse", "HEAD"]).
    cwd : Path | None, optional
        Working directory for the command.

    Returns
    -------
    str | None
        Stripped stdout if returncode is 0; None otherwise.
    """
    try:
        cmd = ["git"] + args if args[0] != "git" else args
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=GIT_COMMAND_TIMEOUT, cwd=cwd
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("Git command failed: %s", e)
        return None


def _get_version_info() -> dict:
    """Get version information for reproducibility tracking.

    Returns
    -------
    dict[str, str]
        Keys include: stellcoilbench_commit, stellcoilbench_branch, simsopt_version,
        simsopt_commit, simsopt_branch, simsopt_remote (when available).
    """
    info: dict = {}
    commit = _run_git_command(["rev-parse", "HEAD"])
    info["stellcoilbench_commit"] = commit if commit else "unknown"
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        info["stellcoilbench_branch"] = branch

    # Get simsopt version
    try:
        import simsopt

        version = getattr(simsopt, "__version__", "unknown")
        info["simsopt_version"] = version

        # Try to extract git commit from version string (e.g., "0.1.dev5270+gee055f063")
        # The format is: version.devN+gCOMMIT where COMMIT is the abbreviated hash
        if "+g" in version:
            # Extract commit hash after +g
            commit_part = version.split("+g")[-1]
            # Remove any additional suffixes (e.g., .dirty)
            commit_hash = commit_part.split(".")[0]
            info["simsopt_commit"] = commit_hash

        # Try to get simsopt git info if installed from source (editable install)
        simsopt_file = getattr(simsopt, "__file__", None)
        if simsopt_file is None:
            return info
        simsopt_path = Path(simsopt_file).parent
        # Check both parent and grandparent for .git (handles different install layouts)
        for parent in [simsopt_path.parent, simsopt_path.parent.parent]:
            simsopt_git_dir = parent / ".git"
            if simsopt_git_dir.exists():
                commit = _run_git_command(["-C", str(parent), "rev-parse", "HEAD"])
                if commit:
                    info["simsopt_commit"] = commit
                branch = _run_git_command(
                    ["-C", str(parent), "rev-parse", "--abbrev-ref", "HEAD"]
                )
                if branch:
                    info["simsopt_branch"] = branch
                remote = _run_git_command(
                    ["-C", str(parent), "remote", "get-url", "origin"]
                )
                if remote:
                    info["simsopt_remote"] = remote
                break  # Found git info, stop searching
    except ImportError:
        info["simsopt_version"] = "not installed"

    return info
