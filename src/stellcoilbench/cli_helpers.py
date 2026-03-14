"""
Pure-utility helpers extracted from ``cli.py``.

These functions have no dependency on CLI-specific state and are used by both
the CLI entry points and tests.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import typer

from .constants import DEFAULT_USERNAME, ZIP_FILENAME

logger = logging.getLogger(__name__)

GIT_COMMAND_TIMEOUT = 5  # seconds
NETWORK_TIMEOUT = 2  # seconds


def _write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    """Serialize *obj* as JSON to *path* using :class:`NumpyJSONEncoder`.

    Parameters
    ----------
    path : Path
        Destination file path.
    obj : Any
        Object to serialize (dicts, lists, numpy arrays, etc.).
    indent : int, optional
        JSON indentation level (default 2).
    """
    path.write_text(json.dumps(obj, indent=indent, cls=NumpyJSONEncoder))


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and arrays."""

    def default(self, o):
        # Handle numpy integer types
        if isinstance(o, np.integer):
            return int(o)
        # Handle numpy floating point types
        elif isinstance(o, np.floating):
            return float(o)
        # Handle numpy arrays
        elif isinstance(o, np.ndarray):
            return o.tolist()
        # Handle numpy boolean
        elif isinstance(o, np.bool_):
            return bool(o)
        # Handle jax/jaxlib arrays and other array-like objects
        elif hasattr(o, "__array__"):
            try:
                return np.asarray(o).tolist()
            except (TypeError, ValueError) as exc:
                logger.debug("Non-numeric version component: %s", exc)
        # Handle simsopt objects (SurfaceRZFourier, Vmec, etc.) by converting to string
        # These are not JSON serializable but we want to include them in results
        elif hasattr(o, "__module__") and "simsopt" in str(o.__module__):
            return str(o)
        return super().default(o)


def _cli_error(msg: str, *, code: int = 1) -> None:
    """Print an error message to stderr and exit with *code*.

    Centralises the ``typer.echo(…, err=True); raise typer.Exit(…)`` pattern
    used throughout the CLI.

    Parameters
    ----------
    msg : str
        Error message (will be prefixed with ``"Error: "``).
    code : int, optional
        Exit code (default 1).

    Raises
    ------
    typer.Exit
        Always raised to abort the current CLI command.
    """
    typer.echo(f"Error: {msg}", err=True)
    raise typer.Exit(code=code)


def _fmt_scalar(v: Any) -> str:
    """Format a scalar for display in metrics summary (.2e for numbers).

    Parameters
    ----------
    v : Any
        Value to format.

    Returns
    -------
    str
        Formatted string representation.
    """
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        fv = float(v)
        if fv == 0:
            return "0"
        return f"{v:.2e}"
    return str(v)


def _detect_github_username() -> str:
    """Try to detect GitHub username from remote URL or environment variables.

    Extracts the GitHub owner from the ``origin`` remote URL (HTTPS or SSH
    format).  Falls back to the ``GITHUB_ACTOR`` / ``GITHUB_USER``
    environment variables (useful in CI).

    Returns
    -------
    str
        Detected username, or empty string if not found.
    """
    try:
        # Try to get from remote URL first (most reliable for GitHub username)
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=NETWORK_TIMEOUT,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Extract username from common GitHub URL patterns
            if "github.com" in url:
                # Handle https://github.com/user/repo format
                if url.startswith("https://") or url.startswith("http://"):
                    parts = url.replace(".git", "").split("/")
                    # URL format: https://github.com/user/repo
                    # parts: ['https:', '', 'github.com', 'user', 'repo']
                    if len(parts) >= 4 and parts[2] == "github.com":
                        username = parts[3]
                        if username and username != "github.com":
                            return username
                # Handle git@github.com:user/repo format
                elif url.startswith("git@"):
                    # URL format: git@github.com:user/repo
                    # Split on ':' to get the part after github.com:
                    if ":" in url:
                        after_colon = url.split(":", 1)[1]
                        parts = after_colon.replace(".git", "").split("/")
                        if len(parts) >= 1:
                            username = parts[0]
                            if username:
                                return username
    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.SubprocessError,
    ) as e:
        logger.debug("GitHub username detection from git failed: %s", e)

    # Try environment variable (useful in CI)
    import os

    github_user = os.environ.get("GITHUB_ACTOR") or os.environ.get("GITHUB_USER")
    if github_user:
        return github_user

    return ""


def _resolve_github_username() -> str:
    """Detect the GitHub username, falling back to the default if not found.

    Wraps :func:`_detect_github_username` and emits a warning when the
    detection fails.  Returns the username string to use for directory
    structure and contact metadata.

    Returns
    -------
    str
        Detected GitHub username or :data:`DEFAULT_USERNAME`.
    """
    github_username = _detect_github_username()
    if not github_username:
        github_username = DEFAULT_USERNAME
        typer.echo(
            f"Warning: Could not auto-detect GitHub username. Using '{DEFAULT_USERNAME}'."
        )
    return github_username


def _zip_submission_directory(submission_dir: Path) -> Path:
    """Zip the submission files (excluding PDFs and post-processing outputs) into all_files.zip.

    Creates a zip file named ``all_files.zip`` inside the submission directory.
    PDF files and post-processing outputs (QFM surface, Poincaré plots, VMEC plots, etc.)
    are kept in the directory alongside the zip file.

    Parameters
    ----------
    submission_dir : Path
        Directory containing submission files to zip.

    Returns
    -------
    Path
        Path to the created zip file.
    """
    submission_dir = Path(submission_dir)

    if not submission_dir.exists() or not submission_dir.is_dir():
        typer.echo(f"Warning: Submission directory does not exist: {submission_dir}")
        return submission_dir / ZIP_FILENAME

    zip_path = submission_dir / ZIP_FILENAME

    # Find all files in the submission directory
    files_to_zip = []
    pdf_files_to_keep = []
    for file_path in submission_dir.rglob("*"):
        if file_path.is_file():
            # PDF plots stay in the DATE directory and are NOT zipped
            if file_path.suffix.lower() == ".pdf":
                pdf_files_to_keep.append(file_path)
            else:
                files_to_zip.append(file_path)

    if not files_to_zip:
        typer.echo(f"Warning: No files found in {submission_dir} to zip")
        return zip_path

    # Create zip file with remaining files (excluding PDFs)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_zip:
            # Add file to zip with relative path from submission_dir
            arcname = file_path.relative_to(submission_dir)
            zipf.write(file_path, arcname=arcname)

    # Keep post-processing files in addition to PDFs:
    # - PDF files (bn_error plots)
    # - Post-processing outputs: .vts (QFM surface), .png (plots), post_processing_results.json
    # Note: finite_build_coils.vtk NOT kept outside zip
    post_processing_patterns = [
        "qfm_surface",
        "poincare",
        "boozer",
        "iota",
        "quasisymmetry",
        "post_processing_results",
        "simple_loss_fraction",  # SIMPLE fast particle tracing plot
        "simple",  # Also match any file with 'simple' in name
    ]

    # Remove files that should be zipped, but keep PDFs and post-processing files
    for file_path in files_to_zip:
        # Keep if it's a post-processing file (check filename patterns)
        is_post_processing_file = any(
            pattern.lower() in file_path.name.lower()
            for pattern in post_processing_patterns
        ) and file_path.suffix.lower() in {".vts", ".vtk", ".png", ".json"}

        if not is_post_processing_file:
            try:
                file_path.unlink()
                # Try to remove parent directory if it's empty (but not the submission_dir itself)
                parent = file_path.parent
                if (
                    parent != submission_dir
                    and parent.exists()
                    and not any(parent.iterdir())
                ):
                    try:
                        parent.rmdir()
                    except (OSError, FileNotFoundError) as exc:
                        logger.debug("Could not remove file during cleanup: %s", exc)
            except (OSError, FileNotFoundError) as e:
                typer.echo(f"Warning: Failed to remove {file_path}: {e}")

    return zip_path


def _detect_hardware() -> str:
    """Detect hardware information (CPU, GPU, memory).

    Queries the OS for CPU model (via ``lscpu`` on Linux, ``sysctl`` on
    macOS), GPU names (via ``nvidia-smi``), and total RAM (via ``psutil``).

    Returns
    -------
    str
        Formatted string describing the hardware, e.g.
        ``"CPU: Apple M1 | GPU: NVIDIA A100 | RAM: 64.0GB"``.
    """
    parts = []

    # CPU info
    try:
        cpu_info = platform.processor() or platform.machine()
        if cpu_info:
            parts.append(f"CPU: {cpu_info}")
    except (OSError, ValueError) as e:
        logger.debug("CPU info detection failed: %s", e)

    # Try to get more detailed CPU info
    try:
        if platform.system() == "Linux":
            result = subprocess.run(
                ["lscpu"],
                capture_output=True,
                text=True,
                timeout=NETWORK_TIMEOUT,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Model name:" in line:
                        cpu_name = line.split("Model name:")[-1].strip()
                        if cpu_name:
                            parts[0] = f"CPU: {cpu_name}"
                            break
        elif platform.system() == "Darwin":  # macOS
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=NETWORK_TIMEOUT,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts[0] = f"CPU: {result.stdout.strip()}"
    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.SubprocessError,
    ) as e:
        logger.debug("Detailed CPU info detection failed: %s", e)

    # GPU info (NVIDIA)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=NETWORK_TIMEOUT,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_names = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            if gpu_names:
                gpu_str = ", ".join(gpu_names)
                parts.append(f"GPU: {gpu_str}")
    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.SubprocessError,
    ) as e:
        logger.debug("GPU info detection failed: %s", e)

    # Memory info (optional, requires psutil)
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        mem_gb = mem.total / (1024**3)
        parts.append(f"RAM: {mem_gb:.1f}GB")
    except (ImportError, Exception) as e:
        logger.debug("Memory info detection failed: %s", e)

    return " | ".join(parts) if parts else platform.platform()
