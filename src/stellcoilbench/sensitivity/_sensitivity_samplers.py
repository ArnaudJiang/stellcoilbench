"""Gaussian-process perturbation samplers for coil sensitivity analysis.

Provides functions to build and validate GaussianSampler instances used for
stochastic coil perturbations. The perturbation model uses a squared-exponential
covariance kernel in arclength along each coil. Samplers are created with
sigma=1 and scaled at evaluation time for reuse across bisection iterations.

Key functions:
- _coil_arc_length: Physical arc length of a coil curve [m].
- _build_sampler_L_via_eigendecomposition: Robust Cholesky factor when LDLT fails.
- _repair_sampler_L: Validates and repairs numerically defective samplers.
- _make_full_torus_surface: Full-torus plotting surface from half-period input.
- _build_unit_samplers: Builds one sigma=1 GaussianSampler per coil.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from simsopt.geo import CurveXYZFourier, SurfaceRZFourier

logger = logging.getLogger(__name__)


def _coil_arc_length(curve: CurveXYZFourier) -> float:
    """Return the physical arc length of a simsopt Curve (metres)."""
    from simsopt.geo import CurveLength

    return float(CurveLength(curve).J())


def _build_sampler_L_via_eigendecomposition(sampler: Any) -> None:
    """Build sampler.L via eigendecomposition (robust to ill-conditioned covariance).

    Recomputes the covariance kernel from the sampler's parameters, regularises
    via eigendecomposition (clip negative eigenvalues, add jitter), and
    replaces ``sampler.L`` with a Cholesky factor.  Use this when LDLT in
    simsopt's GaussianSampler produces numerically defective results.

    Parameters
    ----------
    sampler : GaussianSampler
        Sampler whose ``.L`` will be replaced (in-place).
    """
    from scipy.linalg import cholesky
    from sympy import Symbol, exp, lambdify

    xs = sampler.points
    ls = sampler.length_scale
    sig = sampler.sigma
    n_derivs = sampler.n_derivs
    n = len(xs)

    x_sym, y_sym = Symbol("x"), Symbol("y")
    kernel = sum(
        sig**2 * exp(-((x_sym - y_sym + i) ** 2) / ls**2) for i in range(-5, 6)
    )

    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    dim = n * (n_derivs + 1)
    cov_mat = np.zeros((dim, dim))
    for ii in range(n_derivs + 1):
        for jj in range(n_derivs + 1):
            derivs = ii * [x_sym] + jj * [y_sym]
            expr = kernel if (ii + jj == 0) else kernel.diff(*derivs)
            lam = lambdify((x_sym, y_sym), expr, "numpy")
            cov_mat[ii * n : (ii + 1) * n, jj * n : (jj + 1) * n] = lam(XX, YY)

    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    eigvals = np.maximum(eigvals, 0.0)
    eps = 1e-10 * np.max(eigvals)
    eigvals += eps
    cov_repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
    cov_repaired = 0.5 * (cov_repaired + cov_repaired.T)

    sampler.L = cholesky(cov_repaired, lower=True)


def _repair_sampler_L(
    sampler: Any,
    sigma: float = 1.0,
    tol: float = 2.0,
    *,
    coil_index: int | None = None,
    arc_len_m: float | None = None,
    correlation_length_m: float | None = None,
    normalised_ls: float | None = None,
) -> bool:
    """Validate and repair the ``GaussianSampler`` factorisation matrix.

    The simsopt ``GaussianSampler`` computes a matrix square-root of the
    covariance kernel via LDLT decomposition.  For certain coil geometries
    the covariance matrix is numerically near-singular (many eigenvalues
    at machine-epsilon level become slightly negative), causing the LDLT
    factorisation to produce an ``L`` matrix whose samples have wildly
    incorrect variance -- sometimes 100--1000x larger than the requested
    ``sigma``.

    This function checks the expected position-block RMS of
    ``sampler.L @ z`` (where ``z ~ N(0,1)``) against the target
    ``sigma``.  If the ratio deviates by more than *tol*, the covariance
    kernel is **recomputed from scratch** using the sampler's parameters,
    regularised via eigendecomposition (negative eigenvalues clipped, small
    diagonal jitter added), and ``sampler.L`` is replaced with a Cholesky
    factor of the repaired matrix.

    Parameters
    ----------
    sampler : GaussianSampler
        A sampler whose ``.L`` attribute may be numerically defective.
    sigma : float
        The ``sigma`` that was used to construct the sampler (typically 1.0
        for unit-sigma samplers).
    tol : float
        Maximum acceptable ratio between the observed position-block RMS
        and the expected value ``sigma``.  Samplers within ``[1/tol, tol]``
        of the expected RMS are left untouched.
    coil_index : int, optional
        Coil index (for diagnostic logging when repair triggers).
    arc_len_m : float, optional
        Physical arc length [m] (for diagnostic logging).
    correlation_length_m : float, optional
        Correlation length [m] (for diagnostic logging).
    normalised_ls : float, optional
        length_scale in [0,1] domain (for diagnostic logging).

    Returns
    -------
    bool
        *True* if the sampler was repaired, *False* if it was already fine.
    """
    L = sampler.L
    n = len(sampler.points)

    LLT = L @ L.T
    pos_trace = np.trace(LLT[:n, :n])
    expected_trace = n * sigma**2
    rms_ratio = np.sqrt(pos_trace / max(expected_trace, 1e-30))

    if 1.0 / tol <= rms_ratio <= tol:
        return False

    logger.warning(
        "GaussianSampler L-matrix numerically defective "
        "(position RMS ratio = %.2f, expected ~1.0); repairing via "
        "eigendecomposition.",
        rms_ratio,
    )
    if coil_index is not None or arc_len_m is not None or normalised_ls is not None:
        diag_parts = []
        if coil_index is not None:
            diag_parts.append(f"coil_index={coil_index}")
        if arc_len_m is not None:
            diag_parts.append(f"arc_length={arc_len_m:.4f} m")
        if correlation_length_m is not None:
            diag_parts.append(f"correlation_length_m={correlation_length_m:.4f}")
        if normalised_ls is not None:
            diag_parts.append(f"normalised_ls={normalised_ls:.4f}")
        diag_parts.append(f"n_quadpoints={n}")
        logger.info("  Diagnostic: %s", ", ".join(diag_parts))

    _build_sampler_L_via_eigendecomposition(sampler)

    pos_trace_new = np.trace((sampler.L @ sampler.L.T)[:n, :n])
    logger.info(
        "Repaired sampler: position RMS ratio %.4f -> %.4f",
        rms_ratio,
        np.sqrt(pos_trace_new / max(expected_trace, 1e-30)),
    )
    return True


def _make_full_torus_surface(
    surface: SurfaceRZFourier,
    nphi: int = 128,
    ntheta: int = 128,
) -> SurfaceRZFourier:
    """Create a full-torus plotting surface from an optimisation surface.

    Copies Fourier coefficients from *surface* into a new
    ``SurfaceRZFourier`` with ``quadpoints_phi/theta`` spanning [0, 1]
    (i.e. the full torus).  If the source surface has a ``filename``
    attribute, the surface is loaded directly from the file with
    ``range="full torus"`` for maximum fidelity.

    Parameters
    ----------
    surface : simsopt.geo.SurfaceRZFourier
        Source surface (may cover only a half-period).
    nphi : int
        Number of toroidal quadrature points on the output surface.
    ntheta : int
        Number of poloidal quadrature points on the output surface.

    Returns
    -------
    simsopt.geo.SurfaceRZFourier
        Full-torus surface with the same shape as the input.
    """
    from simsopt.geo import SurfaceRZFourier

    from ..post_processing import load_surface_with_range

    s_plot = None
    if hasattr(surface, "filename") and surface.filename is not None:
        try:
            s_plot = load_surface_with_range(
                surface.filename,
                surface_range="full torus",
                nphi=nphi,
                ntheta=ntheta,
            )
        except (UnicodeDecodeError, ValueError, OSError):
            s_plot = None

    if s_plot is None:
        quadpoints_phi = np.linspace(0, 1, nphi, endpoint=True)
        quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
        s_plot = SurfaceRZFourier(
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            mpol=surface.mpol,
            ntor=surface.ntor,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
    for m in range(surface.mpol + 1):
        for n in range(-surface.ntor, surface.ntor + 1):
            if surface.get_rc(m, n) != 0:
                s_plot.set_rc(m, n, surface.get_rc(m, n))
            if surface.get_zs(m, n) != 0:
                s_plot.set_zs(m, n, surface.get_zs(m, n))
    return s_plot


def _build_unit_samplers(
    coils: list,
    correlation_length_m: float,
    n_derivs: int = 1,
) -> list:
    """Build ``GaussianSampler`` objects with ``sigma=1`` for each coil.

    Creating a ``GaussianSampler`` is expensive (SymPy symbolic
    differentiation + LDLT factorization).  Because the covariance
    matrix scales as ``sigma**2 * K(length_scale)``, samples drawn from
    a unit-sigma sampler can be rescaled by any desired sigma later.
    This allows the samplers to be created once and reused across many
    different sigma evaluations (e.g. bisection iterations).

    Parameters
    ----------
    coils : list
        List of ``simsopt.field.Coil`` objects.
    correlation_length_m : float
        Correlation length in metres.
    n_derivs : int
        Number of derivatives to sample (1 for ``Coil``, 2+ for
        ``RegularizedCoil``).

    Returns
    -------
    list[GaussianSampler]
        One sampler per coil, with ``sigma=1.0``.
    """
    from simsopt.geo import GaussianSampler

    _NORMALISED_LS_LOW = 0.1
    _NORMALISED_LS_HIGH = 1.5

    samplers: list[GaussianSampler] = []
    for idx, coil in enumerate(coils):
        arc_len = _coil_arc_length(coil.curve)
        normalised_ls = correlation_length_m / max(arc_len, 1e-12)
        sampler = GaussianSampler(
            coil.curve.quadpoints,
            1.0,
            normalised_ls,
            n_derivs=n_derivs,
        )
        if normalised_ls < _NORMALISED_LS_LOW or normalised_ls > _NORMALISED_LS_HIGH:
            _build_sampler_L_via_eigendecomposition(sampler)
            logger.debug(
                "Used robust L-build for coil %d (normalised_ls=%.4f)",
                idx,
                normalised_ls,
            )
        elif _repair_sampler_L(
            sampler,
            coil_index=idx,
            arc_len_m=arc_len,
            correlation_length_m=correlation_length_m,
            normalised_ls=normalised_ls,
        ):
            logger.info("Repaired GaussianSampler for coil %d", idx)
        samplers.append(sampler)
    return samplers
