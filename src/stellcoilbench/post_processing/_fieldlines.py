"""Fieldline tracing and Poincaré plot generation.

This submodule contains the ``trace_fieldlines`` function extracted from
the main ``post_processing`` package so that the package stays manageable
in size.
"""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier

from ..mpi_utils import proc0_print, proc0_try
from ..path_utils import get_reference_radii
from ..utils import suppress_output, timed_section

if TYPE_CHECKING:
    from mpi4py.MPI import Comm as MPI_Comm

from .._optional_imports import optional_import

compute_fieldlines = optional_import(
    "simsopt.field.tracing", "compute_fieldlines", fallback=None
)
plot_poincare_data = optional_import(
    "simsopt.field.tracing", "plot_poincare_data", fallback=None
)
LevelsetStoppingCriterion = optional_import(
    "simsopt.field.tracing", "LevelsetStoppingCriterion", fallback=None
)
InterpolatedField = optional_import(
    "simsopt.field.magneticfieldclasses", "InterpolatedField", fallback=None
)
SurfaceClassifier = optional_import("simsopt.geo", "SurfaceClassifier", fallback=None)
TRACING_AVAILABLE = all(
    x is not None
    for x in (
        compute_fieldlines,
        plot_poincare_data,
        LevelsetStoppingCriterion,
        InterpolatedField,
        SurfaceClassifier,
    )
)


def trace_fieldlines(
    bfield: BiotSavart,
    surface: SurfaceRZFourier,
    output_path: Path,
    nfieldlines: int = 20,
    tmax: float = 20000,
    tol: float = 1e-7,
    n_phi_slices: int = 4,
    use_interpolated_field: bool = True,
    markersize: int = 1,
    comm: MPI_Comm | None = None,
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    Trace fieldlines and generate Poincaré plots.

    This function creates Poincaré plots by tracing magnetic field lines
    starting from points on the magnetic axis outward toward the plasma boundary.

    Parameters
    ----------
    bfield : BiotSavart
        Magnetic field from coils.
    surface : SurfaceRZFourier
        Plasma boundary surface.
    output_path : Path
        Where to save the Poincaré plot.
    nfieldlines : int, default=10
        Number of fieldlines to trace.
    tmax : float, default=10000
        Maximum integration time for fieldline tracing.
    tol : float, default=1e-8
        Tolerance for fieldline integration.
    n_phi_slices : int, default=4
        Number of toroidal angles at which to record Poincaré sections.
    use_interpolated_field : bool, default=True
        Whether to use InterpolatedField for faster tracing (recommended).
    comm : Any, optional
        MPI communicator for parallel tracing.
    dpi : int, default=300
        Resolution for saved figure.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'poincare_plot_path': Path to the generated Poincaré plot
        - 'nfieldlines': Number of fieldlines traced
        - 'tmax': Maximum integration time used
    """
    if not TRACING_AVAILABLE:
        raise ImportError(
            "Fieldline tracing requires simsopt.field.tracing. "
            "Please ensure simsopt is installed with tracing capabilities."
        )

    # Set up initial fieldline starting points
    # Sample R0 between innermost and outermost point along phi = 0, Z = 0 line
    gamma = surface.gamma()  # Shape: (nphi, ntheta, 3)

    # Find phi index closest to 0
    # Surface uses normalized phi in [0, 1] representing [0, 2*pi/nfp]
    phi_normalized_0 = 0.0
    phi_normalized_values = surface.quadpoints_phi
    phi_idx = np.argmin(np.abs(phi_normalized_values - phi_normalized_0))

    # Get all points at phi = 0 (or closest to it)
    points_at_phi0 = gamma[phi_idx, :, :]  # Shape: (ntheta, 3)

    # Find points where Z ≈ 0 (within tolerance)
    z_tolerance = 0.01  # 1 cm tolerance
    z_near_zero_mask = np.abs(points_at_phi0[:, 2]) < z_tolerance

    if np.any(z_near_zero_mask):
        # Compute R = sqrt(X^2 + Y^2) for points where Z ≈ 0
        points_z0 = points_at_phi0[z_near_zero_mask]
        R_values = np.sqrt(points_z0[:, 0] ** 2 + points_z0[:, 1] ** 2)

        R_min = np.min(R_values)
        R_max = np.max(R_values)
    else:
        # Fallback: find point closest to Z = 0
        z_abs = np.abs(points_at_phi0[:, 2])
        closest_idx = np.argmin(z_abs)
        closest_point = points_at_phi0[closest_idx]
        R_closest = np.sqrt(closest_point[0] ** 2 + closest_point[1] ** 2)

        # Use a range around this point
        major_radius, minor_radius_component = get_reference_radii(surface)
        R_min = max(R_closest - minor_radius_component * 0.5, major_radius * 0.5)
        R_max = R_closest + minor_radius_component * 0.5

    # Sample R0 between innermost and outermost points
    # Stay slightly inside boundary to avoid starting on surface
    R_start = R_min * 1.01  # Slightly inside innermost point
    R_end = R_max * 0.99  # Slightly inside outermost point

    # Ensure R_start < R_end (safety check)
    if R_start >= R_end:
        # If they're too close, use a small range around the midpoint
        R_mid = (R_min + R_max) / 2.0
        R_range = max(R_max - R_min, R_min * 0.1)  # At least 10% of R_min
        R_start = R_mid - R_range * 0.4
        R_end = R_mid + R_range * 0.4

    R0 = np.linspace(R_start, R_end, nfieldlines)
    if comm is None or (comm is not None and comm.rank == 0):
        proc0_print(f"R0 values: {R0}")
    Z0 = np.zeros(nfieldlines)

    # Toroidal angles for Poincaré sections
    phis = [(i / n_phi_slices) * (2 * np.pi / surface.nfp) for i in range(n_phi_slices)]

    # Create surface classifier for stopping criteria
    # Following simsopt example: examples/1_Simple/tracing_fieldlines_QA.py
    with timed_section("surface_classifier_setup"):
        sc_fieldline = SurfaceClassifier(surface, h=0.04 * surface.major_radius(), p=2)

    # Use interpolated field for faster tracing if requested
    if use_interpolated_field:
        proc0_print("Initializing InterpolatedField")

        with timed_section("interpolated_field_setup"):
            # Bounds for interpolation chosen so surface is entirely contained
            n = 20
            gamma = surface.gamma()
            rs = np.linalg.norm(gamma[:, :, 0:2], axis=2)
            zs = gamma[:, :, 2]
            rrange = (np.min(rs), np.max(rs), n)
            phirange = (0, 2 * np.pi / surface.nfp, n * 2)
            # Exploit stellarator symmetry and only consider positive z values if applicable
            zrange = (
                (0, np.max(zs), n // 2)
                if surface.stellsym
                else (np.min(zs), np.max(zs), n // 2)
            )

            # Skip function to avoid evaluating outside domain
            def skip(rs, phis, zs):
                rphiz = np.asarray([rs, phis, zs]).T.copy()
                dists = sc_fieldline.evaluate_rphiz(rphiz)
                skip_mask = list((dists < -0.05).flatten())
                proc0_print(
                    "Skip", sum(skip_mask), "cells out of", len(skip_mask), flush=True
                )
                return skip_mask

            # Create interpolated field - matching simsopt example signature exactly
            bfield_interp = InterpolatedField(
                bfield,
                2,
                rrange,
                phirange,
                zrange,
                True,
                nfp=surface.nfp,
                stellsym=surface.stellsym,
                skip=skip,
            )
        proc0_print("Done initializing InterpolatedField.")

        bfield_interp.set_points(surface.gamma().reshape((-1, 3)))
        bfield.set_points(surface.gamma().reshape((-1, 3)))
        proc0_print("Mean(|B|) on plasma surface =", np.mean(bfield.AbsB()), flush=True)

        field_to_trace = bfield_interp
    else:
        field_to_trace = bfield

    proc0_print(
        f"Beginning fieldline tracing ({nfieldlines} fieldlines)...", flush=True
    )
    t1 = _time.time()
    with proc0_try(
        "compute_fieldlines failed: {e}",
        Exception,
        reraise=True,
    ):
        with suppress_output():
            fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
                field_to_trace,
                R0,
                Z0,
                tmax=tmax,
                tol=tol,
                comm=comm,
                phis=phis,
                stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)],
            )
    t2 = _time.time()
    proc0_print(
        f"Time for fieldline tracing={t2 - t1:.3f}s. Num steps={sum([len(ll) for ll in fieldlines_tys]) // nfieldlines}",
        flush=True,
    )

    # Generate Poincaré plot (only on rank 0 for MPI runs)
    if comm is None or comm.rank == 0:
        proc0_print("Generating Poincaré plot...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with timed_section("plot_poincare_data"):
            plot_poincare_data(
                fieldlines_phi_hits,
                phis,
                str(output_path),
                dpi=dpi,
                s=markersize,
                surf=surface,
                aspect="equal",
            )

    # Return only metadata, not the raw trajectory data (which can be huge)
    return {
        "poincare_plot_path": str(output_path),
        "nfieldlines": nfieldlines,
        "tmax": tmax,
    }
