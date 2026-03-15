"""Submission loading and I/O helpers for the update_db pipeline."""

from __future__ import annotations

import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from ..path_utils import (
    coils_json_path_from_dir,
    get_surface_filename,
    surface_stem_from_filename,
)

from ._constraints import normalize_submission_metrics
from ._path_parsing import (
    load_case_yaml_from_submission,
    parse_submission_path,
)

logger = logging.getLogger(__name__)

_TIMESTAMP_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})")


def _extract_date_from_path(
    parsed: Dict[str, Any],
    path: Path | None = None,
) -> str | None:
    """Extract an ISO-8601 run_date from a submission path timestamp.

    Parses the ``MM-DD-YYYY_HH-MM`` pattern found in submission directory
    names or zip file stems.  The *parsed* dict comes from
    :func:`parse_submission_path`; *path* is used as a secondary fallback
    for zip files whose stem itself carries the timestamp.

    Parameters
    ----------
    parsed : dict
        Result of :func:`parse_submission_path`, expected to contain a
        ``"timestamp"`` key.
    path : Path | None
        Original file path.  When its suffix is ``.zip`` and the
        timestamp field does not match, the zip file stem is tried as a
        fallback.

    Returns
    -------
    str | None
        ISO-8601 datetime string (``"YYYY-MM-DDTHH:MM:00"``), or *None*
        when no recognisable timestamp pattern is found.
    """
    ts = parsed.get("timestamp", "")
    if ts.endswith(".zip"):
        ts = Path(ts).stem
    match = _TIMESTAMP_RE.match(ts)
    if not match and path is not None and path.suffix == ".zip":
        match = _TIMESTAMP_RE.match(path.stem)
    if match:
        month, day, year, hour, minute = match.groups()
        return f"{year}-{month}-{day}T{hour}:{minute}:00"
    return None


def _extract_coil_params_from_coils_json(
    path: Path,
    case_yaml_data: dict[str, Any] | None,
    repo_root: Path,
    submissions_root: Path,
    existing_params: dict[str, Any],
) -> dict[str, Any]:
    """Extract coil_order and num_coils from coils.json when missing from case.

    Used as a fallback when case.yaml does not specify coil parameters.
    Loads coils via simsopt, reads order from the first coil's curve, and
    computes base coil count from total coils and surface symmetry (nfp,
    stellsym). Surface resolution uses :func:`~stellcoilbench.path_utils.resolve_surface_path`.

    Parameters
    ----------
    path : Path
        Path to results.json or zip; coils.json is sought at ``path.parent / "coils.json"``.
    case_yaml_data : dict | None
        Parsed case.yaml; used to get surface filename for nfp/stellsym lookup.
    repo_root : Path
        Repository root; plasma surfaces searched under ``repo_root / "plasma_surfaces"``.
    submissions_root : Path
        Submissions root; used by :func:`parse_submission_path` when surface
        is inferred from path structure.
    existing_params : dict
        Params already extracted from case (e.g. by :func:`_extract_coil_params_from_case`).
        Only fills in keys not present here.

    Returns
    -------
    dict[str, Any]
        Metric keys: ``coil_order``, ``num_coils``. Empty dict if coils.json
        missing, unreadable, or all needed keys already in existing_params.
    """
    out: dict[str, Any] = {}
    if "coil_order" in existing_params and "num_coils" in existing_params:
        return out
    coils_json_path = coils_json_path_from_dir(path.parent)
    if coils_json_path is None:
        return out
    try:
        from stellcoilbench.post_processing import load_bfield_from_coils_json
        from stellcoilbench.post_processing._coil_io import _get_coils_from_bfield

        bfield = load_bfield_from_coils_json(coils_json_path)
        coils = _get_coils_from_bfield(bfield)
        if not coils:
            return out
        if (
            "coil_order" not in existing_params
            and hasattr(coils[0], "curve")
            and hasattr(coils[0].curve, "order")
        ):
            out["coil_order"] = float(coils[0].curve.order)
        if "num_coils" not in existing_params:
            total_coils = len(coils)
            nfp = 1
            stellsym = True
            surface_file = None
            if case_yaml_data:
                surface_file = get_surface_filename(case_yaml_data)
            if not surface_file:
                parsed = parse_submission_path(path, submissions_root)
                surface_name = parsed.get("surface", "unknown")
                if surface_name != "unknown":
                    for pattern in [
                        f"input.{surface_name}",
                        f"wout.{surface_name}",
                        surface_name,
                    ]:
                        surface_file = pattern
                        break
            if surface_file:
                try:
                    from ..path_utils import (
                        get_surface_search_base_dirs,
                        resolve_surface_path,
                    )

                    base_dirs = get_surface_search_base_dirs(
                        plasma_surfaces_dir=repo_root / "plasma_surfaces"
                    )
                    surface_file_path = resolve_surface_path(surface_file, base_dirs)
                    if surface_file_path:
                        try:
                            from stellcoilbench.post_processing import (
                                load_surface_with_range,
                            )

                            surface = load_surface_with_range(
                                surface_file_path,
                                surface_range="full torus",
                                nphi=8,
                                ntheta=8,
                            )
                        except (ImportError, OSError, RuntimeError, ValueError):
                            surface = None
                        if surface:
                            nfp = surface.nfp
                            stellsym = surface.stellsym
                except (KeyError, TypeError, AttributeError) as exc:
                    logger.debug(
                        "Could not determine symmetry from case config: %s", exc
                    )
            symmetry_factor = nfp * (2 if stellsym else 1)
            base_coils = total_coils // symmetry_factor
            if base_coils > 0:
                out["num_coils"] = float(base_coils)
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as e:
        logger.warning("Failed to extract coil info from %s: %s", coils_json_path, e)
    return out


def _enrich_submission(
    data: Dict[str, Any],
    path: Path,
    submissions_root: Path,
    case_data: Dict[str, Any] | None,
) -> Tuple[str, Dict[str, Any]]:
    """Normalize, backfill metadata, and build the method key for a submission.

    Shared logic used by both the directory and zip code paths in
    :func:`_load_submissions`.

    Parameters
    ----------
    data : dict
        Raw parsed JSON from ``results.json``.
    path : Path
        Path to the ``results.json`` file or the zip archive.
    submissions_root : Path
        Root ``submissions/`` directory.
    case_data : dict or None
        Parsed ``case.yaml`` dict, or None when not available.

    Returns
    -------
    method_key : str
        ``"contact:surface:user:version"``
    data : dict
        Normalized submission data (modified in place).
    """
    data = normalize_submission_metrics(data)
    data.setdefault("metadata", {})
    meta = data["metadata"]

    parsed = parse_submission_path(path, submissions_root)
    if not meta.get("run_date") or meta.get("run_date") == "2025-12-01T00:00:00":
        extracted_date = _extract_date_from_path(parsed, path)
        if extracted_date:
            meta["run_date"] = extracted_date

    contact = meta.get("contact", "UNKNOWN")
    surface = parsed["surface"]
    user = parsed["user"]
    version = meta.get("method_version") or parsed["version"] or path.parent.name

    if case_data is not None:
        surface_file = get_surface_filename(case_data)
        if surface_file:
            surface = surface_stem_from_filename(surface_file)

    method_key = f"{contact}:{surface}:{user}:{version}"
    return method_key, data


def _load_submissions(
    submissions_root: Path,
) -> Iterable[Tuple[str, Path, Dict[str, Any]]]:
    """Iterate over all submission results.json files under submissions_root.

    Handles both regular directories and zip files. Normalizes metrics and
    extracts surface/user/version from path structure or case.yaml.

    Parameters
    ----------
    submissions_root : Path
        Root directory containing submissions (e.g. ``repo_root / "submissions"``).

    Yields
    ------
    method_key : str
        Unique identifier ``"contact:surface:user:version"``.
    path : Path
        Path to results.json or zip file containing it.
    data : dict
        Parsed and normalized submission JSON (metrics, metadata).
    """
    if not submissions_root.exists():
        logger.warning("Submissions directory does not exist: %s", submissions_root)
        return

    found_count = 0

    for path in submissions_root.rglob("*.json"):
        if path.name != "results.json":
            continue
        if "zenodo_" in str(path):
            continue

        try:
            data = json.loads(path.read_text())
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            OSError,
        ) as e:  # pragma: no cover
            logger.warning("Failed to parse JSON from %s: %s", path, e)
            continue

        case_data = load_case_yaml_from_submission(path)
        method_key, data = _enrich_submission(
            data,
            path,
            submissions_root,
            case_data,
        )

        found_count += 1
        yield method_key, path, data

    for zip_path in submissions_root.rglob("*.zip"):
        if "zenodo_" in str(zip_path):
            continue
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if "results.json" not in zf.namelist():
                    continue

                data = json.loads(zf.read("results.json").decode("utf-8"))

                case_data = load_case_yaml_from_submission(zip_path)
                method_key, data = _enrich_submission(
                    data,
                    zip_path,
                    submissions_root,
                    case_data,
                )

                found_count += 1
                yield method_key, zip_path, data
        except (
            zipfile.BadZipFile,
            KeyError,
            json.JSONDecodeError,
            OSError,
            UnicodeDecodeError,
        ) as e:
            logger.warning("Failed to read zip file %s: %s", zip_path, e)
            continue
    if found_count == 0:
        logger.warning("No results.json files found in %s", submissions_root)
    else:
        logger.info("Found %d submission(s) in %s", found_count, submissions_root)
