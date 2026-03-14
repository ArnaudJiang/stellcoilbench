"""
Finite-build coil geometry generation and VTK export.

Generates 3D coil geometry by sweeping a rectangular cross-section along
the coil centerline (filament) using a rotation-minimizing frame. Output
is a surface mesh VTK file suitable for visualization. Uses ParaStell
when available for tetrahedral mesh; falls back to built-in sweep otherwise.
"""

from __future__ import annotations

from stellcoilbench.constants import MIN_POINTS_ALONG_CURVE

from ._core import (
    DEFAULT_CROSS_SECTION_M,
    _compute_cross_section_frame,
    finite_build_coils_to_msh,
    finite_build_coils_to_vtk,
    sweep_rectangular_cross_section,
)
from ._parastell import (
    _finite_build_coils_to_msh_parastell,
    _finite_build_coils_to_vtk_parastell,
    _last_parastell_error,
)
from ._vtk import _write_vtk_unstructured

__all__ = [
    "_compute_cross_section_frame",
    "sweep_rectangular_cross_section",
    "finite_build_coils_to_vtk",
    "finite_build_coils_to_msh",
    "_finite_build_coils_to_vtk_parastell",
    "_finite_build_coils_to_msh_parastell",
    "_last_parastell_error",
    # Also export for internal use / backwards compatibility
    "_write_vtk_unstructured",
    "STELLARIS_TURN_SIDE_M",
    "DEFAULT_CROSS_SECTION_M",
]

# Stellaris turn cross-section: 20 mm × 20 mm (Lion et al., FED 2025, Table 7)
STELLARIS_TURN_SIDE_M = 0.020  # 20 mm


# Constants namespace for public API
class _ConstantsNamespace:
    """Namespace for finite-build constants."""

    MIN_POINTS_ALONG_CURVE = MIN_POINTS_ALONG_CURVE
    STELLARIS_TURN_SIDE_M = STELLARIS_TURN_SIDE_M
    DEFAULT_CROSS_SECTION_M = DEFAULT_CROSS_SECTION_M


constants = _ConstantsNamespace()
