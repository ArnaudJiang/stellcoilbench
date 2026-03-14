"""Shared surface path resolution for post-processing and coil loading.

Consolidates resolution logic from VMEC, Poincaré, coil I/O, and external eval.
Uses path_utils for case/surface lookup; adds walk-up-from-coils fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..path_utils import resolve_surface_file_path

if TYPE_CHECKING:
    from simsopt.geo import SurfaceRZFourier


def _resolve_surface_path_from_hints(
    surface_filename: str | None,
    case_yaml_path: Path | None,
    plasma_dir: Path | None,
    coils_path: Path | None,
) -> Path | None:
    """Resolve plasma surface file path from hints.

    Shared by _resolve_surface_from_hints, _setup_surface_for_eval, and
    callers that have surface filename (or surface.filename) plus case/plasma/coils hints.
    Tries direct path first, then resolve_surface_file_path, then walk-up from coils.

    Parameters
    ----------
    surface_filename : str or None
        Surface filename (e.g. from surface.filename or explicit surface_file).
    case_yaml_path : Path or None
        Path to case YAML.
    plasma_dir : Path or None
        Plasma surfaces directory.
    coils_path : Path or None
        Path to coils JSON for walk-up search.

    Returns
    -------
    Path or None
        Resolved path to surface/VMEC input file.
    """
    if surface_filename:
        potential = Path(surface_filename)
        if potential.exists():
            return potential

    resolved = resolve_surface_file_path(
        case_yaml_path=case_yaml_path,
        surface_filename=surface_filename,
        plasma_surfaces_dir=plasma_dir,
        coils_json_path=coils_path,
    )
    if resolved is not None and resolved.exists():
        return resolved

    if surface_filename and coils_path is not None:
        potential_path = Path(surface_filename)
        coils_json_dir = coils_path.parent
        for _ in range(5):
            for subdir in ("", "plasma_surfaces"):
                candidate = (
                    coils_json_dir / subdir / potential_path.name
                    if subdir
                    else coils_json_dir / potential_path.name
                )
                if candidate.exists():
                    return candidate
            if coils_json_dir.parent == coils_json_dir:
                break
            coils_json_dir = coils_json_dir.parent

    return None


def _resolve_surface_from_hints(
    surface: "SurfaceRZFourier",
    case_yaml_path: Path | None,
    plasma_dir: Path | None,
    coils_path: Path | None,
) -> Path | None:
    """Resolve plasma surface file path from surface object and hints.

    Shared by VMEC and Poincaré surface resolution. Extracts surface.filename
    and delegates to _resolve_surface_path_from_hints.

    Parameters
    ----------
    surface : SurfaceRZFourier
        Surface with optional filename attribute.
    case_yaml_path : Path or None
        Path to case YAML.
    plasma_dir : Path or None
        Plasma surfaces directory.
    coils_path : Path or None
        Path to coils JSON for walk-up search.

    Returns
    -------
    Path or None
        Resolved path to surface/VMEC input file.
    """
    surface_filename = getattr(surface, "filename", None) or None
    return _resolve_surface_path_from_hints(
        surface_filename, case_yaml_path, plasma_dir, coils_path
    )
