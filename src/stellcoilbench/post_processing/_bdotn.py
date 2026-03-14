"""
Shared B·n (magnetic field normal component) computation utilities.

Provides functions to evaluate the magnetic field produced by coils on
plasma surface quadrature points and compute B·n error metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict
import numpy as np

if TYPE_CHECKING:
    from simsopt.field import BiotSavart
    from simsopt.geo import SurfaceRZFourier


def compute_bdotn_on_surface(
    bfield: "BiotSavart",
    surface: "SurfaceRZFourier",
) -> Dict[str, float]:
    r"""Compute B·n error metrics on the plasma surface.

    Evaluates the magnetic field produced by *bfield* on the quadrature
    points of *surface*. The normal component is:

    .. math::
        B\cdot n = \mathbf{B}\cdot\mathbf{n}

    Returns the mean absolute value and the normalised error:

    .. math::
        \frac{\langle |B\cdot n| \rangle}{\langle |\mathbf{B}| \rangle}

    Parameters
    ----------
    bfield : BiotSavart | MagneticFieldSum
        Magnetic field object whose ``set_points`` / ``B`` / ``AbsB``
        methods will be called.
    surface : SurfaceRZFourier
        Plasma surface with ``gamma``, ``unitnormal``, ``quadpoints_phi``,
        and ``quadpoints_theta`` attributes.

    Returns
    -------
    Dict[str, float]
        ``{"BdotN": <value>, "BdotN_over_B": <value>}``
    """
    bfield.set_points(surface.gamma().reshape((-1, 3)))
    B = bfield.B()
    n = surface.unitnormal()
    nphi = surface.quadpoints_phi.size
    ntheta = surface.quadpoints_theta.size
    B_reshaped = B.reshape((nphi, ntheta, 3))
    n_reshaped = n.reshape((nphi, ntheta, 3))
    BdotN = np.mean(np.abs(np.sum(B_reshaped * n_reshaped, axis=2)))
    BdotN_over_B = BdotN / np.mean(bfield.AbsB())
    return {"BdotN": float(BdotN), "BdotN_over_B": float(BdotN_over_B)}


def compute_bdotn_point_data(
    bfield: "BiotSavart",
    surface: "SurfaceRZFourier",
) -> Dict[str, np.ndarray]:
    r"""Compute per-point B·n arrays on the surface.

    Returns :math:`B\cdot n = \mathbf{B}\cdot\mathbf{n}` and
    :math:`|\mathbf{B}|` at each quadrature point, shaped ``(nphi, ntheta)``.
    Used by VTK export and B·n error plotting.

    Parameters
    ----------
    bfield : BiotSavart | MagneticFieldSum
        Magnetic field object.
    surface : SurfaceRZFourier
        Plasma surface with ``gamma``, ``unitnormal``, ``quadpoints_phi``,
        and ``quadpoints_theta`` attributes.

    Returns
    -------
    Dict[str, np.ndarray]
        ``{"BdotN": (nphi, ntheta), "absB": (nphi, ntheta)}``
    """
    bfield.set_points(surface.gamma().reshape((-1, 3)))
    B = bfield.B()
    n = surface.unitnormal()
    nphi = surface.quadpoints_phi.size
    ntheta = surface.quadpoints_theta.size
    B_reshaped = B.reshape((nphi, ntheta, 3))
    n_reshaped = n.reshape((nphi, ntheta, 3))
    BdotN = np.sum(B_reshaped * n_reshaped, axis=2)
    absB = bfield.AbsB().reshape((nphi, ntheta))
    return {"BdotN": BdotN, "absB": absB}
