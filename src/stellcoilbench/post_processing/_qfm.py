"""QFM (quasi-flux surface) computation utilities for post-processing.

Computes the QFM surface from a plasma boundary and magnetic field
using simsopt's make_qfm, used for flux-surface quality metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simsopt.field import BiotSavart
    from simsopt.geo import SurfaceRZFourier

from .._optional_imports import optional_import

make_qfm = optional_import(
    "simsopt.util.permanent_magnet_helper_functions", "make_qfm", fallback=None
)
if make_qfm is None:
    make_qfm = optional_import("simsopt.util", "make_qfm", fallback=None)
if make_qfm is None:
    raise ImportError(
        "make_qfm not found. Please ensure simsopt is installed with "
        "permanent magnet utilities."
    )
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier


def compute_qfm_surface(
    surface: SurfaceRZFourier,
    bfield: BiotSavart,
    n_iters: int = 20,
) -> SurfaceRZFourier:
    r"""Compute QFM (quasi-flux surface) from plasma boundary and magnetic field.

    The QFM surface is a flux surface constructed by iteratively moving
    points along field lines; it approximates a surface where :math:`\mathbf{B}
    \cdot\mathbf{n} \approx 0` (minimal normal field component).

    Parameters
    ----------
    surface : SurfaceRZFourier
        Plasma boundary surface.
    bfield : BiotSavart
        Magnetic field from coils.
    n_iters : int, default=20
        Number of iterations for QFM convergence.

    Returns
    -------
    SurfaceRZFourier
        QFM surface.
    """
    qfm_surf = make_qfm(surface, bfield, n_iters=n_iters)
    return qfm_surf.surface
