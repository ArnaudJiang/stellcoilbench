"""Finite-build coil VTK generation runner for post-processing.

Orchestrates finite-build coil geometry (extrusion along winding pack)
and VTK export for visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier

from ..structural_analysis._pipeline import DEFAULT_STRUCTURAL_MESH_RESOLUTION_M
from ..mpi_utils import proc0_print, proc0_try
from ..utils import timed_section

from ._coil_io import _get_coils_from_bfield, get_unique_coils


def _run_finite_build_vtk(
    bfield: BiotSavart,
    surface: SurfaceRZFourier,
    output_dir: Path,
    width: Optional[float],
    height: Optional[float],
    results: Dict[str, Any],
) -> None:
    """Generate finite-build coil VTK output (rank 0 only).

    Uses a rotation-minimizing frame sweep to produce a surface mesh VTK.

    Parameters
    ----------
    bfield : BiotSavart
        Magnetic field containing coil data.
    surface : SurfaceRZFourier
        Plasma surface (used for nfp/stellsym to extract unique coils).
    output_dir : Path
        Directory for output files.
    width, height : float or None
        Winding-pack cross-section dimensions.
    results : dict
        Accumulated results dict; ``finite_build_vtk_path`` is added in-place.
    """

    def _on_finite_build_catch(exc: BaseException) -> None:
        import traceback

        proc0_print(
            f"[finite-build diagnostic] Exception type={type(exc).__name__},"
            f" repr={exc!r}"
        )
        proc0_print(f"[finite-build diagnostic] Traceback:\n{traceback.format_exc()}")

    with proc0_try(
        "Finite-build VTK generation failed: {e}",
        Exception,
        on_catch=_on_finite_build_catch,
    ):
        from stellcoilbench.finite_build import finite_build_coils_to_vtk

        all_coils = _get_coils_from_bfield(bfield)
        coils_for_fb = get_unique_coils(
            all_coils,
            nfp=int(surface.nfp) if hasattr(surface, "nfp") else 1,
            stellsym=bool(getattr(surface, "stellsym", False)),
        )
        proc0_print(
            f"[finite-build diagnostic] all_coils={len(all_coils)},"
            f" unique coils_for_fb={len(coils_for_fb)}"
            f" (nfp={int(surface.nfp) if hasattr(surface, 'nfp') else 1},"
            f" stellsym={bool(getattr(surface, 'stellsym', False))})"
        )
        for i, c in enumerate(coils_for_fb):
            try:
                g = c.curve.gamma()
                n = len(g) if hasattr(g, "__len__") else getattr(g, "shape", (0,))[0]
            except Exception as inner_e:
                n = f"ERROR({type(inner_e).__name__})"
            proc0_print(
                f"[finite-build diagnostic]   coil {i}: curve.gamma() -> {n} points"
            )
        if coils_for_fb:
            # Post-processing: always use DEFAULT_STRUCTURAL_MESH_RESOLUTION_M
            # for finite-build mesh so structural solve uses consistent resolution
            min_mesh_size = DEFAULT_STRUCTURAL_MESH_RESOLUTION_M
            max_mesh_size = DEFAULT_STRUCTURAL_MESH_RESOLUTION_M
            with timed_section("finite_build_vtk"):
                fb_path = finite_build_coils_to_vtk(
                    coils_for_fb,
                    output_dir / "finite_build_coils",
                    width=width,
                    height=height,
                    min_mesh_size=min_mesh_size,
                    max_mesh_size=max_mesh_size,
                )
            results["finite_build_vtk_path"] = str(fb_path)
