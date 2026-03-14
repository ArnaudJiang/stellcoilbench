"""Surface loading utilities (VMEC input, wout, FOCUS format).

Loads plasma surfaces as SurfaceRZFourier with configurable toroidal range.
Moved from post_processing to path_utils to resolve circular imports
(coil_optimization -> post_processing).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from simsopt.geo import SurfaceRZFourier


def load_surface_with_range(
    surface_path: Path | str,
    surface_range: str = "full torus",
    nphi: int = 256,
    ntheta: int = 256,
    *,
    endpoints: bool = True,
) -> SurfaceRZFourier:
    """
    Load a surface from file with a specified range and resolution.

    Detects file type from path (input/wout/focus) and uses the appropriate
    SurfaceRZFourier loader.

    Parameters
    ----------
    surface_path : Path | str
        Path to surface file.
    surface_range : str, default="full torus"
        Range for surface loading ("full torus" or "half period").
    nphi : int, default=256
        Number of phi quadrature points.
    ntheta : int, default=256
        Number of theta quadrature points.
    endpoints : bool, default=True
        If True, quadpoints include 0 and 1 so the surface closes at the
        boundaries. Use for VTK visualization to avoid holes in ParaView.

    Returns
    -------
    SurfaceRZFourier
        Loaded surface with specified range.
    """
    path_str = str(surface_path)
    surface_file_lower = path_str.lower()
    if "input" in surface_file_lower:
        surface = SurfaceRZFourier.from_vmec_input(
            path_str, range=surface_range, nphi=nphi, ntheta=ntheta
        )
    elif "wout" in surface_file_lower:
        surface = SurfaceRZFourier.from_wout(
            path_str, range=surface_range, nphi=nphi, ntheta=ntheta
        )
    elif "focus" in surface_file_lower:
        surface = SurfaceRZFourier.from_focus(
            path_str, range=surface_range, nphi=nphi, ntheta=ntheta
        )
    else:
        raise ValueError(f"Unknown surface type: {surface_path}")

    # Capture reference radii from raw load (stable, independent of quadpoint rebuild).
    # Used throughout for threshold scaling, constraint scaling, and reactor metrics.
    minor_radius_raw = float(surface.minor_radius())
    major_radius_raw = float(surface.major_radius())

    if not endpoints:
        surface._minor_radius_raw = minor_radius_raw  # type: ignore[attr-defined]
        surface._major_radius_raw = major_radius_raw  # type: ignore[attr-defined]
        return surface

    # Rebuild with endpoint-inclusive quadpoints so VTK surfaces close in ParaView
    if surface_range == "full torus":
        quadpoints_phi = np.linspace(0.0, 1.0, nphi, endpoint=True)
        quadpoints_theta = np.linspace(0.0, 1.0, ntheta, endpoint=True)
    else:
        # half period / field period: match simsopt's extent
        try:
            nfp = int(surface.nfp)
            if surface_range == "half period":
                phi_end = 0.5 / nfp
            else:
                phi_end = 1.0 / nfp
            quadpoints_phi = np.linspace(0.0, phi_end, nphi, endpoint=True)
        except (TypeError, ValueError, AttributeError, ZeroDivisionError):
            quadpoints_phi = np.linspace(0.0, 1.0, nphi, endpoint=True)
        quadpoints_theta = np.linspace(0.0, 1.0, ntheta, endpoint=True)

    try:
        nfp_val = int(surface.nfp)
        stellsym_val = surface.stellsym
        mpol_val = surface.mpol
        ntor_val = surface.ntor
    except (TypeError, ValueError, AttributeError):
        surface._minor_radius_raw = minor_radius_raw  # type: ignore[attr-defined]
        surface._major_radius_raw = major_radius_raw  # type: ignore[attr-defined]
        return surface  # Cannot rebuild; return original (e.g. Mock in tests)

    s_plot = SurfaceRZFourier(
        nfp=nfp_val,
        stellsym=stellsym_val,
        mpol=mpol_val,
        ntor=ntor_val,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    for m in range(mpol_val + 1):
        for n in range(-ntor_val, ntor_val + 1):
            try:
                rc_val = surface.get_rc(m, n)
                zs_val = surface.get_zs(m, n)
            except (TypeError, AttributeError):
                continue
            if rc_val != 0:
                s_plot.set_rc(m, n, rc_val)
            if zs_val != 0:
                s_plot.set_zs(m, n, zs_val)
    s_plot._minor_radius_raw = minor_radius_raw  # type: ignore[attr-defined]
    s_plot._major_radius_raw = major_radius_raw  # type: ignore[attr-defined]
    return s_plot


def get_reference_radii(surface: SurfaceRZFourier) -> tuple[float, float]:
    """
    Return (major_radius, minor_radius) from raw file when available, else from surface.

    Surfaces loaded via load_surface_with_range have _major_radius_raw and _minor_radius_raw
    from the raw simsopt load (before quadpoint rebuild). These are stable across resolution
    and ensure consistent a0 and threshold scaling.

    Parameters
    ----------
    surface : SurfaceRZFourier
        Plasma boundary surface.

    Returns
    -------
    tuple[float, float]
        (major_radius [m], minor_radius [m]).
    """
    mjr = getattr(surface, "_major_radius_raw", None)
    mnr = getattr(surface, "_minor_radius_raw", None)
    if mjr is not None and mnr is not None:
        return float(mjr), float(mnr)
    return surface.major_radius(), surface.minor_radius()
