"""
FEM-based structural analysis for finite-build stellarator coils.

Solves 3-D linear elasticity on a tetrahedral mesh of the winding pack,
using electromagnetic Lorentz-force body loads (J × B) and outputs the
displacement field, Cauchy stress tensor, and Von Mises stress as
XDMF / VTK files suitable for ParaView.

The primary solver backend is **DOLFINx** (FEniCSx).  A lightweight
**scikit-fem** fallback is provided for environments where DOLFINx is
not installed (no MPI parallelism, but pip-installable).

Both backends are optional imports; a clear error is raised when neither
is available and the user requests a structural solve.
"""

from __future__ import annotations

from typing import Any

from ._pipeline import (
    DEFAULT_STRUCTURAL_MESH_RESOLUTION_M,
    _DOLFINX_AVAILABLE,
    _SKFEM_AVAILABLE,
    _require_backend,
    compute_lorentz_body_force,
    compute_stress_field,
    export_results,
    load_coil_mesh,
    solve_linear_elasticity,
    write_structural_vtk,
)
from ._runner import run_structural_analysis

__all__ = [
    "DEFAULT_STRUCTURAL_MESH_RESOLUTION_M",
    "_DOLFINX_AVAILABLE",
    "_SKFEM_AVAILABLE",
    "_require_backend",
    "compute_lorentz_body_force",
    "compute_stress_field",
    "export_results",
    "load_coil_mesh",
    "run_structural_analysis",
    "solve_linear_elasticity",
    "write_structural_vtk",
]


# Lazy re-exports: import backend-specific helpers only when accessed.
def __getattr__(name: str) -> Any:
    """Lazy loader for backend-specific functions (DOLFINx or scikit-fem).

    Avoids importing heavy FEM backends until explicitly requested.
    Raises AttributeError for unknown names.
    """
    _dolfinx_names = {
        "_create_mesh_from_points_cells",
        "_load_mesh_dolfinx",
        "_compute_body_force_dolfinx",
        "_solve_elasticity_dolfinx",
        "_compute_stress_dolfinx",
        "_export_dolfinx",
    }
    _skfem_names = {
        "_load_mesh_skfem",
        "_solve_elasticity_skfem",
        "_compute_von_mises_skfem",
        "_export_skfem",
    }

    if name in _dolfinx_names:
        from . import _dolfinx

        return getattr(_dolfinx, name)

    if name in _skfem_names:
        from . import _skfem

        return getattr(_skfem, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
