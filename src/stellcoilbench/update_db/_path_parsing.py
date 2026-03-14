"""Path parsing utilities for submission directories and zip files."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any, Dict
from yaml import YAMLError

from ..path_utils import get_surface_filename, load_yaml, surface_stem_from_filename

logger = logging.getLogger(__name__)


def load_case_yaml_from_submission(path: Path) -> Dict[str, Any] | None:
    """Load case.yaml from a submission path (zip or directory).

    Path resolution:
    - **Zip files** (``.zip`` suffix): reads ``case.yaml`` from inside the archive.
    - **results.json**: reads ``path.parent / "case.yaml"`` (sibling in submission dir).
    - **Directories**: reads ``path / "case.yaml"`` (case.yaml inside the dir).

    Parameters
    ----------
    path : Path
        Path to a submission: zip file (e.g. ``all_files.zip``), ``results.json``,
        or a submission directory.

    Returns
    -------
    dict | None
        Parsed case data (YAML as dict), or None if not found, invalid YAML,
        or I/O error. Logs a warning on failure.
    """
    if path.suffix == ".zip":
        try:
            with zipfile.ZipFile(path, "r") as zf:
                if "case.yaml" not in zf.namelist():
                    return None
                return load_yaml(content=zf.read("case.yaml"))
        except (zipfile.BadZipFile, YAMLError, OSError, UnicodeDecodeError) as e:
            logger.warning("Failed to load case.yaml from zip %s: %s", path, e)
            return None
    case_yaml_path = (
        path.parent / "case.yaml" if path.name == "results.json" else path / "case.yaml"
    )
    if not case_yaml_path.exists():
        return None
    try:
        return load_yaml(path=case_yaml_path)
    except (YAMLError, OSError, UnicodeDecodeError) as e:
        logger.warning("Failed to load case.yaml from %s: %s", case_yaml_path, e)
        return None


def parse_submission_path(path: Path, submissions_root: Path) -> Dict[str, Any]:
    """
    Parse a submission path into surface, user, timestamp, and version.

    Extracts structured metadata from paths under the submissions directory.
    Supports both directory-based layouts (results.json in timestamp dir) and
    zip files (all_files.zip or timestamp-named zips).

    Parameters
    ----------
    path : Path
        Path to a submission file, e.g. ``submissions/surface/user/timestamp/results.json``
        or ``submissions/surface/user/12-01-2025_01-51.zip``.
    submissions_root : Path
        Root directory containing submissions (e.g. ``repo_root / "submissions"``).

    Returns
    -------
    dict
        Keys: ``surface``, ``user``, ``timestamp``, ``version``, ``parts``.
        Uses ``"unknown"`` for surface/user when path structure is incomplete.
        ``parts`` is the list of path components after stripping ``submissions/``.

    Examples
    --------
    >>> parse_submission_path(Path("submissions/QA/user1/run1/results.json"), Path("submissions"))
    {'surface': 'QA', 'user': 'user1', 'timestamp': 'run1', 'version': 'run1', ...}
    """
    result: Dict[str, Any] = {
        "surface": "unknown",
        "user": "unknown",
        "timestamp": "",
        "version": "",
        "parts": [],
    }
    try:
        rel = path.relative_to(submissions_root)
    except ValueError:
        if "submissions" in path.parts:
            idx = path.parts.index("submissions")
            rel = Path(*path.parts[idx + 1 :])
        else:
            return result
    parts = list(rel.parts)
    if parts and parts[0] == "submissions":
        parts = parts[1:]
    result["parts"] = parts
    if len(parts) >= 3:
        result["surface"] = parts[0]
        result["user"] = parts[1]
        result["timestamp"] = parts[2]
        if path.suffix == ".zip":
            result["version"] = (
                path.parent.name if path.name == "all_files.zip" else path.stem
            )
        else:
            result["version"] = path.parent.name
    elif len(parts) >= 2:
        result["surface"] = parts[0]
        result["user"] = parts[1]
        result["version"] = (
            path.parent.name
            if path.suffix == ".zip"
            else (path.parent.name if len(parts) >= 2 else "")
        )
    elif len(parts) >= 1:
        result["user"] = parts[0]
    return result


def _extract_surface_from_submission_path(
    path_obj: Path, submissions_root: Path
) -> str:
    """
    Extract plasma surface name from a submission path.

    Tries case.yaml first (from zip contents or sibling file), then falls back
    to path structure via :func:`parse_submission_path`. Normalizes surface
    filenames (e.g. ``input.LandremanPaul2021_QA``) to stems using
    :func:`surface_stem_from_filename`.

    Parameters
    ----------
    path_obj : Path
        Path to a submission: either a zip file (e.g. ``all_files.zip``) or a
        directory/results.json file.
    submissions_root : Path
        Root of the submissions directory.

    Returns
    -------
    str
        Surface identifier (e.g. ``"LandremanPaul2021_QA"``), or ``"unknown"``
        if extraction fails (missing case.yaml, invalid path structure, etc.).
    """
    surface_name = "unknown"
    case_data = load_case_yaml_from_submission(path_obj)
    if case_data:
        surface_file = get_surface_filename(case_data)
        if surface_file:
            surface_name = surface_stem_from_filename(surface_file)
    if surface_name == "unknown":
        parsed = parse_submission_path(path_obj, submissions_root)
        if parsed["surface"] != "unknown":
            surface_name = parsed["surface"]
    return surface_name
