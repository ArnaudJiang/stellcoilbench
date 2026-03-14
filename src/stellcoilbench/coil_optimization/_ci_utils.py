"""
CI-related utilities for coil optimization.

Provides helpers for zipping output files, detecting CI environments,
and redirecting verbose output during optimization.
"""

from __future__ import annotations

import os
import sys
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator


def _zip_output_files(out_dir: Path) -> Path:
    """
    Zip all output files in the output directory with a date stamp.

    Finds VTK files (*.vtu, *.vts), zips them, removes the originals,
    and returns the path to the created zip file.

    Parameters
    ----------
    out_dir: Path
        Directory containing output files to zip.

    Returns
    -------
    Path
        Path to the created zip file.
    """
    out_dir = Path(out_dir)

    # Create date-stamped zip filename: YYYY-MM-DD_HH-MM-SS.zip
    now = datetime.now()
    zip_filename = now.strftime("%Y-%m-%d_%H-%M-%S.zip")
    zip_path = out_dir / zip_filename

    # Find all files to zip (VTK files, JSON files, etc.)
    # Only zip VTK files for compression - keep JSON files (coils.json, results.json) unzipped
    files_to_zip = []
    for pattern in ["*.vtu", "*.vts"]:
        files_to_zip.extend(out_dir.glob(pattern))

    # Only create zip if there are files to zip
    if files_to_zip:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files_to_zip:
                # Add file to zip with relative path (just filename)
                zipf.write(file_path, arcname=file_path.name)

        # Remove original VTK files after zipping for compression
        for file_path in files_to_zip:
            file_path.unlink()

    return zip_path


def _is_ci_running() -> bool:
    """
    Check if the code is running in a CI environment.

    Returns
    -------
    bool
        True if running in CI (GitHub Actions, GitLab CI, Jenkins, etc.),
        False otherwise.
    """
    ci_env_vars = [
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "JENKINS_URL",
        "TRAVIS",
        "CIRCLECI",
        "APPVEYOR",
        "BUILDKITE",
    ]
    return any(os.getenv(var) for var in ci_env_vars)


@contextmanager
def _nullcontext() -> Generator[None, None, None]:
    """Null context manager that does nothing (no-op for with statement)."""
    yield


@contextmanager
def _redirect_verbose_to_file(output_file: Path) -> Generator[None, None, None]:
    """
    Context manager to redirect stdout to a file while preserving stderr.

    Parameters
    ----------
    output_file : Path
        Path to the file where stdout should be written.
    """
    original_stdout = sys.stdout
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            sys.stdout = f
            yield
    finally:
        sys.stdout = original_stdout
