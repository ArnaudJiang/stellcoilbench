"""Case YAML and coil JSON path resolution utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ResolvedPaths:
    """Canonical result of path resolution for post-processing and coil loading.

    Single entry point for case YAML, surface file, coils JSON, and plasma
    surfaces directory. Use :func:`resolve_all` to populate.
    """

    case_yaml: Path | None
    surface_file: Path | None
    coils_json: Path | None
    plasma_surfaces_dir: Path | None


@dataclass(frozen=True)
class CaseResolutionResult:
    """Result of case and surface path resolution.

    Use :func:`resolve_case_and_surface` as the canonical resolver.
    Supports tuple unpacking: ``case_path, surface_path, case_data = result``.
    """

    case_yaml_path: Path | None
    surface_path: Path | None
    case_data: dict[str, Any]

    def __iter__(self) -> Any:
        """Allow tuple unpacking for backward compatibility."""
        return iter((self.case_yaml_path, self.surface_path, self.case_data))


from ._path_search import find_dir_up, find_file_up, find_plasma_surfaces_dir
from ._surface_extraction import get_surface_filename
from ._surface_resolution import (
    get_surface_search_base_dirs,
    resolve_surface_file_path,
    resolve_surface_path,
    surface_stem_from_filename,
)
from ._yaml import load_yaml

_SURFACE_HINT_NAMES: tuple[str, ...] = (
    "Landreman",
    "HSX",
    "CFQS",
    "MUSE",
    "NCSX",
    "W7X",
    "tokamak",
    "ellipse",
)


def resolve_case_yaml_path(
    out_dir: Path,
    case_path_hint: Path | str | None = None,
    surface_filename: str | None = None,
) -> Path | None:
    """
    Resolve case.yaml path from output directory, hints, and surface filename.

    Search order: (1) case_path_hint if valid file/dir, (2) ``out_dir/case.yaml``,
    (3) ``out_dir.parent/case.yaml``, (4) surface-based paths (``surface_dir``,
    ``cases/stem``),
    (5) scan cases/*.yaml for surface match.

    Parameters
    ----------
    out_dir : Path
        Output directory (e.g., from optimization).
    case_path_hint : Path | str | None
        Known case path (file or directory containing case.yaml).
    surface_filename : str | None
        Surface filename (e.g., from s.filename) for cases/stem/case.yaml search.

    Returns
    -------
    Path | None
        Path to case.yaml if found; None otherwise.
    """
    # 1. From hint
    if case_path_hint is not None:
        p = Path(case_path_hint) if isinstance(case_path_hint, str) else case_path_hint
        if p.exists():
            if p.is_file():
                return p
            if p.is_dir():
                cp = p / "case.yaml"
                if cp.exists():
                    return cp

    # 2. Out dir
    out = Path(out_dir)
    for candidate in [out / "case.yaml", out.parent / "case.yaml"]:
        if candidate.exists():
            return candidate

    # 3. Surface-based
    if surface_filename:
        surface_dir = Path(surface_filename).parent
        stem = surface_stem_from_filename(surface_filename)
        for path in [
            surface_dir / "case.yaml",
            surface_dir.parent / "case.yaml",
            Path("cases") / stem / "case.yaml",
        ]:
            if path.exists():
                return path

    # 4. Find cases dir and scan for surface match
    cases_dir = find_dir_up(out, "cases", max_levels=10) or Path("cases")
    if cases_dir.exists() and surface_filename:
        surf_name = Path(surface_filename).name
        for yaml_file in cases_dir.glob("*.yaml"):
            try:
                data = load_yaml(yaml_file)
                if data and isinstance(data, dict):
                    surf_in_case = get_surface_filename(data)
                    if surf_name and surf_name in surf_in_case:
                        return yaml_file.resolve()
                    if surf_in_case in surf_name:
                        return yaml_file.resolve()
            except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
                continue

    return None


def coils_json_path_from_dir(out_dir: Path) -> Path | None:
    """Return path to coils JSON in directory, preferring biot_savart_optimized.json.

    Checks for ``biot_savart_optimized.json`` first (simsopt optimization output),
    then ``coils.json`` (generic fallback). Returns the first that exists.

    Parameters
    ----------
    out_dir : Path
        Directory containing coil output files.

    Returns
    -------
    Path | None
        Path to coils JSON file if found; None otherwise.
    """
    out = Path(out_dir)
    for name in ("biot_savart_optimized.json", "coils.json"):
        candidate = out / name
        if candidate.exists():
            return candidate
    return None


def _search_cases_dir_for_yaml(
    coils_json_path: Path,
    potential_case_paths: list[Path],
) -> Path | None:
    """Scan cases/ directory for a YAML whose surface matches the coils path."""
    cases_dir = find_dir_up(coils_json_path.parent, "cases", max_levels=7)
    if cases_dir is None:
        return None

    surface_hint: str | None = None
    for part in coils_json_path.parts:
        if any(name in part for name in _SURFACE_HINT_NAMES):
            surface_hint = part
            break

    if surface_hint is None:
        return None

    for yaml_file in cases_dir.glob("*.yaml"):
        try:
            case_data = load_yaml(path=yaml_file)
            if not case_data or not isinstance(case_data, dict):
                continue
            surface_in_case = get_surface_filename(case_data)
            normalised_hint = surface_hint.replace("_", "")
            normalised_surface = surface_in_case.replace("_", "")
            if normalised_hint in normalised_surface:
                potential_case_paths.append(yaml_file)
                return yaml_file
            hint_fuzzy = surface_hint.replace("2021", "").replace("_", "").lower()
            surface_fuzzy = surface_in_case.replace("2021", "").replace("_", "").lower()
            if hint_fuzzy in surface_fuzzy:
                potential_case_paths.append(yaml_file)
                return yaml_file
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
            continue
    return None


def resolve_case_and_surface(
    case_hint: Path | str | None,
    coils_path: Path | None = None,
    plasma_dir: Path | None = None,
) -> CaseResolutionResult:
    """Resolve case YAML and plasma surface paths from hints.

    Single source of truth for case/surface resolution. When case_hint is
    None, searches from coils_path: walk up for case.yaml, guess cases/<stem>/,
    or scan cases/ for matching surface.

    Parameters
    ----------
    case_hint : Path or str or None
        Explicit path to case.yaml or case directory. If None, resolved from
        coils_path when provided.
    coils_path : Path, optional
        Path to coils JSON; used as search base when case_hint is None.
    plasma_dir : Path, optional
        Directory containing plasma surface files.

    Returns
    -------
    CaseResolutionResult
        (case_yaml_path, surface_path, raw_case_data). Any may be None/empty
        if not found. Supports tuple unpacking for backward compatibility.
    """
    case_yaml_path: Path | None = None
    if case_hint is not None:
        p = Path(case_hint)
        if p.exists():
            if p.is_file():
                case_yaml_path = p
            elif p.is_dir() and (p / "case.yaml").exists():
                case_yaml_path = p / "case.yaml"

    potential_case_paths: list[Path] = []

    if case_yaml_path is None or not case_yaml_path.exists():
        if coils_path is not None:
            coils_path = Path(coils_path)
            case_yaml_from_up = find_file_up(coils_path.parent, "case.yaml")
            if case_yaml_from_up:
                potential_case_paths.append(case_yaml_from_up)

            potential_case_paths.append(
                Path("cases")
                / coils_path.stem.replace("coils", "")
                .replace("biot_savart", "")
                .replace("_optimized", "")
                .replace(".json", "")
                / "case.yaml"
            )

            for path in potential_case_paths:
                if path.exists():
                    case_yaml_path = path
                    break

            if case_yaml_path is None or not (
                case_yaml_path and case_yaml_path.exists()
            ):
                case_yaml_path = _search_cases_dir_for_yaml(
                    coils_path, potential_case_paths
                )

    if case_yaml_path is None or not case_yaml_path.exists():
        return CaseResolutionResult(None, None, {})

    case_data = load_yaml(path=case_yaml_path)
    surface_file = get_surface_filename(case_data)
    if not surface_file:
        return CaseResolutionResult(case_yaml_path, None, case_data)

    if Path(surface_file).is_absolute():
        surface_path = Path(surface_file)
        if surface_path.exists():
            return CaseResolutionResult(case_yaml_path, surface_path, case_data)
        return CaseResolutionResult(case_yaml_path, None, case_data)

    base_dirs = get_surface_search_base_dirs(
        case_path=case_yaml_path.parent if case_yaml_path else None,
        plasma_surfaces_dir=Path(plasma_dir) if plasma_dir else None,
        coils_json_path=coils_path,
    )
    surface_path = resolve_surface_path(surface_file, base_dirs)
    return CaseResolutionResult(case_yaml_path, surface_path, case_data)


def find_case_and_surface_path(
    coils_json_path: Path,
    case_yaml_path: Path | None,
    plasma_surfaces_dir: Path | None,
) -> tuple[Path, Path]:
    """Locate case YAML and plasma surface path; raise if not found.

    Wrapper around :func:`resolve_case_and_surface` that raises
    FileNotFoundError or ValueError when case/surface cannot be resolved.

    Parameters
    ----------
    coils_json_path : Path
        Path to coils JSON file.
    case_yaml_path : Path | None
        Explicit case YAML path, or None to search from coils_json_path.
    plasma_surfaces_dir : Path | None
        Optional plasma surfaces directory.

    Returns
    -------
    tuple of (Path, Path)
        (resolved_case_yaml_path, surface_path).

    Raises
    ------
    FileNotFoundError
        If case YAML or surface file cannot be found.
    ValueError
        If case YAML does not specify a surface.
    """
    result = resolve_case_and_surface(
        case_hint=case_yaml_path,
        coils_path=coils_json_path,
        plasma_dir=plasma_surfaces_dir,
    )
    case_path, surface_path, case_data = result
    if case_path is None or not case_path.exists():
        raise FileNotFoundError(f"Could not find case YAML near {coils_json_path}")
    if not get_surface_filename(case_data):
        raise ValueError("No surface file specified in case.yaml")
    if surface_path is None or not surface_path.exists():
        raise FileNotFoundError(
            f"Surface file not found. Case: {case_path}, "
            f"surface_params.surface: {get_surface_filename(case_data)}"
        )
    return (case_path, surface_path)


def resolve_all(
    out_dir: Path,
    case_hint: Path | str | None = None,
    surface_filename: str | None = None,
    coils_hint: Path | None = None,
) -> ResolvedPaths:
    """Resolve case YAML, surface file, coils JSON, and plasma_surfaces dir.

    Canonical path resolution entry point. Combines resolve_case_yaml_path,
    resolve_surface_file_path, coils_json_path_from_dir, and
    find_plasma_surfaces_dir into a single call.

    Parameters
    ----------
    out_dir : Path
        Output directory (e.g., optimization output dir) used as search base.
    case_hint : Path | str | None, optional
        Explicit case path (file or directory containing case.yaml).
    surface_filename : str | None, optional
        Surface filename for case.yaml search (e.g., from s.filename).
    coils_hint : Path | None, optional
        Explicit coils JSON path. When None, probes out_dir for coils.

    Returns
    -------
    ResolvedPaths
        case_yaml, surface_file, coils_json, plasma_surfaces_dir (any may be None).
    """
    out = Path(out_dir).resolve()
    case_yaml = resolve_case_yaml_path(
        out, case_path_hint=case_hint, surface_filename=surface_filename
    )
    coils_json = coils_hint if coils_hint is not None else coils_json_path_from_dir(out)
    plasma_surfaces_dir = find_plasma_surfaces_dir(out)
    surface_file = resolve_surface_file_path(
        case_yaml_path=case_yaml,
        surface_filename=surface_filename,
        plasma_surfaces_dir=plasma_surfaces_dir,
        coils_json_path=coils_json,
    )
    return ResolvedPaths(
        case_yaml=case_yaml,
        surface_file=surface_file,
        coils_json=coils_json,
        plasma_surfaces_dir=plasma_surfaces_dir,
    )
