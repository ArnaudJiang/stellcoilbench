"""Poincaré plot and fieldline tracing coordination for post-processing.

Dispatches fieldline integration and Poincaré section plotting, using
simsopt or dedicated fieldline tracers. Provides functions to generate
Poincaré plots from coil field and surface, including surface resolution
for full-torus fieldline tracing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier

from ..mpi_utils import comm_world, proc0_print, proc0_try
from ..utils import timed_section

from ._fieldlines import trace_fieldlines
from ..path_utils import load_surface_with_range


def _run_poincare_analysis(
    bfield: BiotSavart,
    surface: SurfaceRZFourier,
    output_dir: Path,
    case_yaml_path: Optional[Path],
    plasma_surfaces_dir: Optional[Path],
    nfieldlines: int,
    is_mpi_parallel: bool,
) -> Dict[str, Any]:
    r"""Generate Poincaré section plot from coil field and surface.

    Traces magnetic field lines and records their intersection with a
    Poincaré section (plane or surface). Field line equation
    :math:`d\mathbf{x}/dt = \mathbf{B}(\mathbf{x})`.

    Parameters
    ----------
    bfield : BiotSavart or MagneticFieldSum
        Magnetic field from coils.
    surface : SurfaceRZFourier
        Plasma boundary surface.
    output_dir : Path
        Output directory for PNG file.
    case_yaml_path : Path or None
        Path to case YAML for surface lookup.
    plasma_surfaces_dir : Path or None
        Directory containing plasma surface files.
    nfieldlines : int
        Number of fieldlines to trace.
    is_mpi_parallel : bool
        Whether MPI parallelism is active.

    Returns
    -------
    Dict[str, Any]
        Poincare results dict, or empty dict on failure.
    """
    result: Dict[str, Any] = {}
    with proc0_try(
        "Poincaré plot generation failed: {e}",
        OSError,
        RuntimeError,
        ValueError,
        ImportError,
        TypeError,
        default={},
        on_catch=lambda: proc0_print("Skipping Poincaré plot."),
    ):
        proc0_print("Generating Poincaré plot...")
        with timed_section("poincare_total"):
            with timed_section("poincare_load_surface"):
                poincare_surface = _resolve_poincare_surface(
                    surface, case_yaml_path, plasma_surfaces_dir
                )

            _poincare_comm = comm_world if is_mpi_parallel else None
            poincare_results = trace_fieldlines(
                bfield,
                poincare_surface,
                output_dir / "poincare_plot.png",
                nfieldlines=nfieldlines,
                comm=_poincare_comm,
            )
        result = {"poincare_results": poincare_results}
    return result


def _resolve_poincare_surface(
    surface: SurfaceRZFourier,
    case_yaml_path: Path | None,
    plasma_surfaces_dir: Path | None,
    coils_json_path: Path | None = None,
) -> SurfaceRZFourier:
    """Load a full-torus surface for Poincaré section plotting.

    Fieldline tracing requires a surface spanning the full torus. Tries
    the surface filename first, then falls back to case YAML/coils resolution,
    then to the original surface object.

    Parameters
    ----------
    surface : SurfaceRZFourier
        Original optimization surface.
    case_yaml_path : Path or None
        Path to case YAML for surface file lookup.
    plasma_surfaces_dir : Path or None
        Directory containing plasma surface files.
    coils_json_path : Path or None, optional
        Path to coils JSON for walk-up search.

    Returns
    -------
    SurfaceRZFourier
        Full-torus surface for Poincaré plotting.
    """
    from ._surface_resolution import _resolve_surface_from_hints

    resolved = _resolve_surface_from_hints(
        surface, case_yaml_path, plasma_surfaces_dir, coils_json_path
    )
    if resolved is not None:
        with proc0_try(
            f"Failed to load resolved surface from {resolved}: {{e}}",
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
        ):
            proc0_print("Loading surface with 'full torus' range for Poincaré plot...")
            return load_surface_with_range(resolved, surface_range="full torus")

    proc0_print(
        "Warning: Could not find surface file for full torus loading, using original surface"
    )
    return surface
