"""Structural analysis and shape gradient runners for post-processing.

Coordinates FEM structural analysis (Lorentz force, stress) and shape
gradient computation for coil curves when requested by post-process
or submit-case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier

from ..mpi_utils import proc0_print, proc0_try, proc0_warning
from ..utils import timed_section

from ._coil_io import _get_coils_from_bfield, get_unique_coils


def _run_structural(
    bfield: BiotSavart,
    surface: SurfaceRZFourier,
    output_dir: Path,
    results: Dict[str, Any],
    finite_build_width: Optional[float],
    finite_build_height: Optional[float],
    structural_E: Optional[float],
    structural_nu: Optional[float],
    export_full_coil_set: bool = False,
) -> None:
    r"""Run FEM structural analysis on finite-build coil mesh (rank 0 only).

    Evaluates Lorentz body-force :math:`\mathbf{f} = \mathbf{J}\times\mathbf{B}`
    on the winding pack, solves linear elasticity, and exports displacement
    and Von Mises stress. Uses only unique (base) coils for the mesh; the
    full bfield is used for J × B so the magnetic field includes all coils.

    The mesh resolution is always ``DEFAULT_STRUCTURAL_MESH_RESOLUTION_M``;
    it is not read from the case config. The structural objective in the
    optimization loop uses ``structural_mesh_resolution_coarse`` /
    ``structural_mesh_resolution_fine`` from ``coil_objective_terms`` instead.

    Parameters
    ----------
    bfield : BiotSavart
        Magnetic field containing coil data.
    surface : SurfaceRZFourier
        Plasma surface (used for nfp/stellsym to extract unique coils).
    output_dir : Path
        Directory for output files.
    results : dict
        Accumulated results dict; ``structural_metrics`` is added in-place.
    finite_build_width, finite_build_height : float or None
        Winding-pack cross-section dimensions. Width defaults to 35 cm
        (reactor scale, a0-scaled with 5 cm floor); height defaults to width.
    structural_E : float or None
        Young's modulus override.
    structural_nu : float or None
        Poisson ratio override.
    """
    try:
        from stellcoilbench.structural_analysis import run_structural_analysis
        from stellcoilbench.structural_analysis._pipeline import (
            DEFAULT_STRUCTURAL_MESH_RESOLUTION_M,
        )

        all_coils = _get_coils_from_bfield(bfield)
        coils_for_sa = get_unique_coils(
            all_coils,
            nfp=int(surface.nfp) if hasattr(surface, "nfp") else 1,
            stellsym=bool(getattr(surface, "stellsym", False)),
        )
        if coils_for_sa:
            fb_w = finite_build_width if finite_build_width is not None else 0.35
            fb_h = finite_build_height if finite_build_height is not None else fb_w
            fb_vtk = results.get("finite_build_vtk_path")
            nfp = int(surface.nfp) if hasattr(surface, "nfp") else 1
            stellsym = bool(getattr(surface, "stellsym", False))
            with timed_section("structural_analysis"):
                sa_results = run_structural_analysis(
                    coils=coils_for_sa,
                    bs=bfield,
                    output_dir=output_dir,
                    vtk_path=Path(fb_vtk) if fb_vtk else None,
                    width=fb_w,
                    height=fb_h,
                    E=structural_E,
                    nu=structural_nu,
                    structural_mesh_resolution_coarse=DEFAULT_STRUCTURAL_MESH_RESOLUTION_M,
                    export_full_coil_set=export_full_coil_set,
                    nfp=nfp,
                    stellsym=stellsym,
                )
            results["structural_metrics"] = sa_results
    except (OSError, RuntimeError, ImportError, ValueError) as e:
        proc0_print(
            f"\n{'=' * 60}\n"
            f"WARNING: Structural analysis SKIPPED\n"
            f"  Reason: {type(e).__name__}: {e}\n"
            f"{'=' * 60}\n"
        )


def _run_shape_gradient_analysis(
    bfield: BiotSavart,
    surface: SurfaceRZFourier,
    output_dir: Path,
    results: Dict[str, Any],
) -> None:
    r"""Compute per-coil shape gradients and save in ``coils_optimized.vtu``.

    Uses ``SquaredFlux`` on the given surface/field to obtain the objective
    gradient :math:`dJ/d\mathbf{p}`, then solves for the pointwise shape
    gradient :math:`\nabla_s J` on every coil (Fourier-weighted Gram system)
    and writes the result as extra point data into ``coils_optimized.vtu``.

    Parameters
    ----------
    bfield : BiotSavart
        Magnetic field containing coils.
    surface : SurfaceRZFourier
        Plasma boundary surface.
    output_dir : Path
        Directory for output files.
    results : dict
        Accumulated results dict; ``shape_gradient_vtk_path`` is added
        in-place on success.
    """
    with proc0_try(
        "Shape gradient computation failed: {e}",
        OSError,
        RuntimeError,
        ImportError,
        ValueError,
        TypeError,
    ):
        from simsopt.field import coils_to_vtk

        from ._shape_gradient import (
            compute_shape_gradients,
            shape_gradient_to_vtk_data,
        )

        coils = _get_coils_from_bfield(bfield)
        if not coils:
            proc0_warning("Shape gradient: no coils found in bfield.")
            return

        proc0_print("Computing per-coil shape gradients...")
        with timed_section("shape_gradient"):
            sg = compute_shape_gradients(coils, bfield, surface)
            vtk_extra = shape_gradient_to_vtk_data(sg, coils, close=True)
            vtk_path = output_dir / "coils_optimized"
            coils_to_vtk(coils, vtk_path, close=True, extra_data=vtk_extra)

        results["shape_gradient_vtk_path"] = str(vtk_path)
        n_valid = sum(1 for x in sg if x is not None)
        proc0_print(
            f"  Shape gradients computed for {n_valid}/{len(coils)} coils, "
            f"saved to {vtk_path}.vtu"
        )
