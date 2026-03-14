"""Coil loading utilities for post-processing.

Loads BiotSavart or MagneticFieldSum from coils.json / biot_savart_optimized.json,
optionally strips simsopt-version-incompatible keys for cross-branch compatibility,
and provides helpers to resolve case/surface paths when only a coils file is given.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

from simsopt import load
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier

from .._optional_imports import optional_import

MagneticFieldSum = optional_import(
    "simsopt.field.magneticfield", "MagneticFieldSum", fallback=None
)

from ..path_utils import find_case_and_surface_path, load_yaml, load_surface_with_range


def load_coils_from_json(
    path: Path,
    strip_compat_keys: bool = True,
) -> Any:
    """
    Load BiotSavart or MagneticFieldSum from coils JSON file.

    Optionally strips simsopt version-incompatible keys (nfp, stellsym,
    regularization) for auglag_coils vs main branch compatibility.

    Parameters
    ----------
    path : Path
        Path to coils.json or biot_savart_optimized.json.
    strip_compat_keys : bool, default=True
        If True, strip nfp/stellsym/regularization from serialized objects
        before loading (for cross-version compatibility).

    Returns
    -------
    BiotSavart | MagneticFieldSum
        Loaded magnetic field object.
    """
    if not path.exists():
        raise FileNotFoundError(f"Coils JSON not found: {path}")
    data = json.loads(path.read_text())
    if strip_compat_keys:
        for obj in data.get("simsopt_objs", {}).values():
            if isinstance(obj, dict):
                if obj.get("@class") == "CurvePlanarFourier":
                    obj.pop("nfp", None)
                    obj.pop("stellsym", None)
                if obj.get("@class") == "Coil":
                    obj.pop("regularization", None)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(data, tf, indent=2)
            tmp_path = Path(tf.name)
        try:
            return load(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
    return load(str(path))


def load_bfield_from_coils_json(
    path: Path,
    strip_compat_keys: bool = True,
) -> Any:
    """Load coils JSON and return normalized magnetic field (BiotSavart or MagneticFieldSum).

    Loads via :func:`load_coils_from_json` then normalizes: if the loaded object
    is a raw list of coils, wraps it in BiotSavart. Used by :func:`load_coils_and_surface`
    and external coil evaluation to avoid duplicating load+normalize logic.

    Parameters
    ----------
    path : Path
        Path to coils.json or biot_savart_optimized.json.
    strip_compat_keys : bool, default=True
        If True, strip nfp/stellsym/regularization before loading for
        cross-version compatibility between simsopt branches.

    Returns
    -------
    BiotSavart | MagneticFieldSum
        Magnetic field object ready for evaluation. BiotSavart for modular
        coils; MagneticFieldSum when multiple BiotSavart fields are combined.

    Raises
    ------
    FileNotFoundError
        If the coils JSON file does not exist at *path*.
    """
    if not path.exists():
        raise FileNotFoundError(f"Coils JSON file not found: {path}")

    bfield = load_coils_from_json(path, strip_compat_keys=strip_compat_keys)

    if isinstance(bfield, BiotSavart):
        return bfield
    if MagneticFieldSum is not None and isinstance(bfield, MagneticFieldSum):
        return bfield
    coils = bfield if isinstance(bfield, list) else [bfield]
    return BiotSavart(coils)


def _get_coils_from_bfield(bfield: BiotSavart) -> list:
    """Extract list of coils from BiotSavart or MagneticFieldSum.

    At runtime *bfield* may also be a :class:`MagneticFieldSum` (which
    wraps multiple ``BiotSavart`` instances); the ``BiotSavart``
    annotation is used because ``MagneticFieldSum`` may not be
    available at import time.

    Parameters
    ----------
    bfield : BiotSavart
        Magnetic field object containing coils.  Accepts
        ``MagneticFieldSum`` at runtime.

    Returns
    -------
    list
        Flat list of coil objects (simsopt Coil instances).
    """
    if isinstance(bfield, BiotSavart):
        return list(bfield.coils)
    if MagneticFieldSum is not None and isinstance(bfield, MagneticFieldSum):
        coils = []
        for bf in bfield.Bfields:
            coils.extend(_get_coils_from_bfield(bf))
        return coils
    return []


def get_unique_coils(
    coils: list,
    nfp: int = 1,
    stellsym: bool = False,
) -> list:
    """Extract the unique (base) coil subset using stellarator symmetry.

    Stellarator coils are often generated via symmetry (nfp field periods,
    optionally stellarator symmetry). This function returns one representative
    coil per base type, avoiding redundant FEM/mesh work on symmetric copies.

    Parameters
    ----------
    coils : list
        Full list of simsopt Coil objects (including symmetric copies).
    nfp : int, default=1
        Number of field periods. Use 1 when unknown (e.g. tokamak).
    stellsym : bool, default=False
        Whether stellarator symmetry is used. With nfp=1, use False to treat
        all coils as unique.

    Returns
    -------
    list
        Subset of coils representing unique base coils (one per symmetry).
    """
    if not coils:
        return []
    symmetry_factor = nfp * (2 if stellsym else 1)
    ncoils = max(1, len(coils) // symmetry_factor)
    # Simsopt coils_via_symmetries returns base-first order: [B0, B1, ..., B_n, copies]
    return coils[:ncoils]


def _setup_surface_for_eval(
    coils_json_path: Path,
    case_yaml_path: Optional[Path] = None,
    plasma_surfaces_dir: Optional[Path] = None,
    surface_file: Optional[str] = None,
    surface_range: str = "half period",
    nphi: int = 256,
    ntheta: int = 256,
) -> Tuple[Path, str, SurfaceRZFourier]:
    """Resolve and load plasma surface for coil evaluation.

    Tries find_case_and_surface_path first; on failure uses shared
    _resolve_surface_path_from_hints (case/plasma/coils + walk-up) when surface_file
    is provided. Used by load_coils_and_surface and external eval.

    Parameters
    ----------
    coils_json_path : Path
        Path to coils JSON (for walk-up case/surface resolution).
    case_yaml_path : Path, optional
        Explicit case YAML path.
    plasma_surfaces_dir : Path, optional
        Directory containing plasma surface files.
    surface_file : str, optional
        Explicit surface filename (used when no case found).
    surface_range : str
        Default surface range when not in case YAML.
    nphi, ntheta : int
        Quadrature resolution for loaded surface.

    Returns
    -------
    tuple
        (surface_path, surface_range, surface).

    Raises
    ------
    FileNotFoundError
        If surface cannot be resolved.
    """
    plasma_surfaces_dir = plasma_surfaces_dir or Path("plasma_surfaces")
    surface_path: Path | None = None
    effective_range = surface_range

    try:
        case_path, surface_path = find_case_and_surface_path(
            coils_json_path, case_yaml_path, plasma_surfaces_dir
        )
        case_data = load_yaml(path=case_path)
        if case_data:
            effective_range = case_data.get("surface_params", {}).get(
                "range", surface_range
            )
    except (FileNotFoundError, ValueError) as e:
        if surface_file:
            from ._surface_resolution import _resolve_surface_path_from_hints

            surface_path = _resolve_surface_path_from_hints(
                surface_filename=surface_file,
                case_yaml_path=case_yaml_path,
                plasma_dir=plasma_surfaces_dir,
                coils_path=coils_json_path,
            )
            if surface_path is None or not surface_path.exists():
                raise FileNotFoundError(
                    f"Surface not found (case/surface_file={surface_file})"
                ) from e
        else:
            raise

    surface = load_surface_with_range(
        surface_path,
        surface_range=effective_range,
        nphi=nphi,
        ntheta=ntheta,
    )
    surface.filename = str(surface_path.resolve())

    return surface_path, effective_range, surface


def load_coils_and_surface(
    coils_json_path: Path,
    case_yaml_path: Optional[Path] = None,
    plasma_surfaces_dir: Optional[Path] = None,
    nphi: int = 256,
    ntheta: int = 256,
) -> Tuple[Any, SurfaceRZFourier]:
    """
    Load coils from JSON and plasma surface from case.yaml or VMEC input file.

    Parameters
    ----------
    coils_json_path : Path
        Path to coils JSON file (e.g., biot_savart_optimized.json or coils.json).
    case_yaml_path : Path, optional
        Path to case.yaml file. If None, tries to find it relative to coils_json_path.
    plasma_surfaces_dir : Path, optional
        Directory containing plasma surface files. Defaults to "plasma_surfaces".
    nphi : int, default=256
        Number of phi quadrature points for the loaded surface.
    ntheta : int, default=256
        Number of theta quadrature points for the loaded surface.

    Returns
    -------
    Tuple[BiotSavart | MagneticFieldSum, SurfaceRZFourier]
        Magnetic field object and plasma surface.

    Raises
    ------
    FileNotFoundError
        If coils JSON or surface file cannot be found.
    ValueError
        If surface file type is not recognized.
    """
    bfield = load_bfield_from_coils_json(coils_json_path)
    _path, _range, surface = _setup_surface_for_eval(
        coils_json_path,
        case_yaml_path=case_yaml_path,
        plasma_surfaces_dir=plasma_surfaces_dir,
        nphi=nphi,
        ntheta=ntheta,
    )
    return bfield, surface
