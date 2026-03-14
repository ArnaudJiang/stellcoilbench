"""Physics computations for structural analysis.

Implements the Landreman et al. (2025) regularized internal magnetic field
(Breg + B0 + Bkappa + Bb) for coil self-field at points inside conductors,
Lamé parameters for elasticity, and J×B Lorentz force density.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import constants

if TYPE_CHECKING:
    from simsopt.field import BiotSavart, Coil

MU0 = constants.mu_0


def _lame_parameters(E: float, nu: float) -> tuple[float, float]:
    """Compute Lamé parameters from Young's modulus and Poisson ratio.

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson ratio (dimensionless).

    Returns
    -------
    lam : float
        First Lamé parameter (λ).
    mu : float
        Shear modulus (second Lamé parameter, μ).
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def _G_helper(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Landreman et al. (2025) Eq 17-18 auxiliary function.

    G(x, y) = y * arctan(x/y) + (x/2) * ln(1 + y²/x²)

    Uses arctan(x/y) NOT arctan2, per the paper. Handles x=0 and y=0 limits.

    Parameters
    ----------
    x : np.ndarray
        First argument (may be scalar or array).
    y : np.ndarray
        Second argument (same shape as x).

    Returns
    -------
    np.ndarray
        G(x, y) values, same shape as x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(x)

    # x=0, y=0: limit is 0
    both_zero = (np.abs(x) < 1e-20) & (np.abs(y) < 1e-20)
    out[both_zero] = 0.0
    mask = ~both_zero

    # y=0, x≠0: arctan(x/0) → sign(x)*pi/2, ln(1+0) = 0
    # G(x,0) = 0 + (x/2)*0 = 0 (limit from Taylor expansion)
    y_zero = mask & (np.abs(y) < 1e-20)
    out[y_zero] = 0.0
    mask = mask & ~y_zero

    # x=0, y≠0: arctan(0)=0, ln(1+y²/0) diverges - use limit x→0: (x/2)*ln(1+y²/x²) → 0
    x_zero = mask & (np.abs(x) < 1e-20)
    out[x_zero] = 0.0
    mask = mask & ~x_zero

    # General case: G(x,y) = y*arctan(x/y) + (x/2)*ln(1+y²/x²)
    xm, ym = x[mask], y[mask]
    out[mask] = ym * np.arctan(xm / ym) + (xm / 2) * np.log(1.0 + (ym**2) / (xm**2))
    return out


def _compute_coil_frame(coil: "Coil") -> dict[str, Any]:
    """Compute the centroid frame {t, p, q} and curvature components for a coil.

    Uses centroid frame (Eq 40): C = mean(gamma), w = gamma - C,
    p = (w - (w·t)t) / |...|, q = t × p.
    Curvature: kappa*n = d(t)/d(arclength), kappa1 = kappa*n·p, kappa2 = kappa*n·q,
    kappa_b = kappa1*q - kappa2*p (Eq 8).

    Parameters
    ----------
    coil : Coil
        simsopt Coil (curve + current).

    Returns
    -------
    dict
        Keys: gamma, t, p, q, kappa1, kappa2, kappa_b, current, n_pts.
    """
    gamma = np.asarray(coil.curve.gamma(), dtype=float).reshape(-1, 3)
    gammadash = np.asarray(coil.curve.gammadash(), dtype=float).reshape(-1, 3)
    gammadashdash = np.asarray(coil.curve.gammadashdash(), dtype=float).reshape(-1, 3)

    # t = tangent (simsopt uses param in [0,1], so gammadash = d/d(phi/2pi) * 2pi)
    t_norm = np.linalg.norm(gammadash, axis=1, keepdims=True)
    t_norm = np.where(t_norm < 1e-14, 1.0, t_norm)
    t = gammadash / t_norm

    # Centroid frame (Eq 40)
    C = np.mean(gamma, axis=0)
    w = gamma - C
    wt = np.sum(w * t, axis=1, keepdims=True)
    w_perp = w - wt * t
    w_perp_norm = np.linalg.norm(w_perp, axis=1, keepdims=True)
    # Avoid division by zero (straight wire: w_perp can be ~0)
    w_perp_norm = np.where(w_perp_norm < 1e-14, 1.0, w_perp_norm)
    p = w_perp / w_perp_norm
    q = np.cross(t, p, axis=1)

    # Curvature: kappa*n = dt/ds, ds = |gammadash|*d(phi/2pi)
    # For simsopt: phi in [0, 2pi], dgamma/dphi = gammadash (already 2pi scaled)
    # So ds/dphi = |gammadash|, dt/dphi = (gammadashdash/|gammadash|) - t*(t·gammadashdash/|gammadash|^2)
    ds_dphi = np.squeeze(t_norm)
    denom = np.where(ds_dphi < 1e-14, 1.0, ds_dphi)[:, np.newaxis]
    dt_dphi = (gammadashdash / t_norm) - t * np.sum(
        t * gammadashdash, axis=1, keepdims=True
    ) / (t_norm**2)
    dt_ds = dt_dphi / denom
    kappa_n = dt_ds

    kappa_mag = np.linalg.norm(kappa_n, axis=1, keepdims=True)
    kappa_mag = np.where(kappa_mag < 1e-14, 1.0, kappa_mag)
    n_hat = kappa_n / kappa_mag
    kappa = np.squeeze(kappa_mag)

    kappa1 = np.sum(n_hat * p, axis=1) * kappa
    kappa2 = np.sum(n_hat * q, axis=1) * kappa
    kappa_b = kappa1[:, np.newaxis] * q - kappa2[:, np.newaxis] * p

    current = float(coil.current.get_value())
    return {
        "gamma": gamma,
        "t": t,
        "p": p,
        "q": q,
        "kappa1": kappa1,
        "kappa2": kappa2,
        "kappa_b": kappa_b,
        "current": current,
        "n_pts": len(gamma),
    }


def _compute_B0(
    u: np.ndarray,
    v: np.ndarray,
    a: float,
    b: float,
    current: float,
    p: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Compute B0 (infinite straight wire field) at (u, v). Landreman Eqs 17-18.

    B0 = (mu0*I/(4*pi*a*b)) * sum_{su,sv in {-1,1}} su*sv * [G(b(v-sv), a(u-su))*q - G(a(u-su), b(v-sv))*p]

    Parameters
    ----------
    u, v : np.ndarray
        Shape (n,), cross-section coords in [-1, 1].
    a, b : float
        Cross-section full side dimensions [m] (paper Eq 6).
    current : float
        Coil current [A].
    p, q : np.ndarray
        Frame vectors at each point, shape (n, 3).

    Returns
    -------
    np.ndarray
        B0 field, shape (n, 3).
    """
    prefac = (MU0 * current) / (4 * np.pi * a * b)
    # Vectorized over (su, sv) in {(1,1), (1,-1), (-1,1), (-1,-1)}; coeff su*sv = 1, -1, -1, 1
    c1_11 = _G_helper(b * (v - 1), a * (u - 1))
    c2_11 = _G_helper(a * (u - 1), b * (v - 1))
    c1_1m1 = _G_helper(b * (v + 1), a * (u - 1))
    c2_1m1 = _G_helper(a * (u - 1), b * (v + 1))
    c1_m11 = _G_helper(b * (v - 1), a * (u + 1))
    c2_m11 = _G_helper(a * (u + 1), b * (v - 1))
    c1_m1m1 = _G_helper(b * (v + 1), a * (u + 1))
    c2_m1m1 = _G_helper(a * (u + 1), b * (v + 1))
    B0 = prefac * (
        (c1_11[:, np.newaxis] * q - c2_11[:, np.newaxis] * p)
        - (c1_1m1[:, np.newaxis] * q - c2_1m1[:, np.newaxis] * p)
        - (c1_m11[:, np.newaxis] * q - c2_m11[:, np.newaxis] * p)
        + (c1_m1m1[:, np.newaxis] * q - c2_m1m1[:, np.newaxis] * p)
    )
    return B0


def _K_curvature_vector(
    U: np.ndarray,
    V: np.ndarray,
    a: float,
    b: float,
    kappa1: np.ndarray,
    kappa2: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Landreman Eq 20: vector K(U,V) for curvature correction Bkappa.

    K(U,V) = -2UV*(kappa1*q - kappa2*p)*ln(aU²/b + bV²/a)
             + (kappa2*q - kappa1*p)*(aU²/b + bV²/a)*ln(aU²/b + bV²/a)
             + 4aU²*kappa2*p/b * arctan(bV/(aU))
             - 4bV²*kappa1*q/a * arctan(aU/(bV))

    Handles U=0 and V=0 limits: arctan(bV/(aU)) -> sign(V)*pi/2 when U->0,
    arctan(aU/(bV)) -> sign(U)*pi/2 when V->0. Uses arg*ln(arg) -> 0 as arg->0.

    Parameters
    ----------
    U, V : np.ndarray
        Shape (n,), dimensionless cross-section offsets.
    a, b : float
        Full side dimensions [m].
    kappa1, kappa2 : np.ndarray
        Curvature components, shape (n,).
    p, q : np.ndarray
        Frame vectors, shape (n, 3).

    Returns
    -------
    np.ndarray
        K(U,V) vector field, shape (n, 3).
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    arg = (a * U**2) / b + (b * V**2) / a
    arg_safe = np.maximum(arg, 1e-300)

    ln_arg = np.log(arg_safe)
    arg_ln_arg = np.where(arg < 1e-300, 0.0, arg * ln_arg)

    k1q_m_k2p = kappa1[:, np.newaxis] * q - kappa2[:, np.newaxis] * p
    k2q_m_k1p = kappa2[:, np.newaxis] * q - kappa1[:, np.newaxis] * p

    eps = 1e-14
    with np.errstate(divide="ignore", invalid="ignore"):
        atan_va = np.where(
            np.abs(U) < eps,
            np.sign(V) * (np.pi / 2),
            np.arctan((b * V) / (a * U)),
        )
        atan_ub = np.where(
            np.abs(V) < eps,
            np.sign(U) * (np.pi / 2),
            np.arctan((a * U) / (b * V)),
        )

    term1 = -2 * U[:, np.newaxis] * V[:, np.newaxis] * k1q_m_k2p * ln_arg[:, np.newaxis]
    term2 = k2q_m_k1p * arg_ln_arg[:, np.newaxis]
    term3 = (
        (4 * a * U**2 / b)[:, np.newaxis]
        * (kappa2[:, np.newaxis] * p)
        * atan_va[:, np.newaxis]
    )
    term4 = (
        -(4 * b * V**2 / a)[:, np.newaxis]
        * (kappa1[:, np.newaxis] * q)
        * atan_ub[:, np.newaxis]
    )

    return term1 + term2 + term3 + term4


def _compute_Bkappa(
    u: np.ndarray,
    v: np.ndarray,
    a: float,
    b: float,
    current: float,
    p: np.ndarray,
    q: np.ndarray,
    kappa1: np.ndarray,
    kappa2: np.ndarray,
) -> np.ndarray:
    r"""Compute Bκ (curvature correction to internal field). Landreman et al. (2025) Eqs 19–20.

    .. math::
        \mathbf{B}_\kappa = \frac{\mu_0 I}{64\pi}
        \sum_{s_u,s_v \in \{\pm 1\}} s_u s_v \,
        \mathbf{K}(u - s_u, v - s_v)

    where :math:`\mathbf{K}(U,V)` is the curvature vector from Eq. 20.

    Parameters
    ----------
    u, v : np.ndarray
        Dimensionless cross-section coordinates in :math:`[-1, 1]`.
    a, b : float
        Cross-section full side dimensions [m].
    current : float
        Coil current [A].
    p, q : np.ndarray
        Frame vectors at each point, shape ``(n, 3)``.
    kappa1, kappa2 : np.ndarray
        Curvature components, shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Bκ field, shape ``(n, 3)``.
    """
    prefac = (MU0 * current) / (64 * np.pi)
    result = np.zeros((len(u), 3))
    for su in (1, -1):
        for sv in (1, -1):
            U = u - su
            V = v - sv
            result += (
                prefac * su * sv * _K_curvature_vector(U, V, a, b, kappa1, kappa2, p, q)
            )
    return result


def _compute_Bb(current: float, kappa_b: np.ndarray, delta: float) -> np.ndarray:
    r"""Compute Bb (binormal curvature term). Landreman et al. (2025) Eq. 21.

    .. math::
        \mathbf{B}_b = \frac{\mu_0 I}{8\pi}
        \left( 4 + 2\ln 2 + \ln\delta \right) \boldsymbol{\kappa}_b

    where :math:`\delta` is the regularisation parameter and
    :math:`\boldsymbol{\kappa}_b` is the binormal curvature vector.

    Parameters
    ----------
    current : float
        Coil current [A].
    kappa_b : np.ndarray
        Binormal curvature vector at each centerline point, shape ``(n, 3)``.
    delta : float
        Regularisation parameter (typically :math:`\delta = \mathtt{reg}/(ab)`).

    Returns
    -------
    np.ndarray
        Bb field, shape ``(n, 3)``.
    """
    prefac = (MU0 * current) / (8 * np.pi) * (4 + 2 * np.log(2) + np.log(delta))
    return prefac * kappa_b


def _compute_Breg_for_coil(coil: "Coil", width: float, height: float) -> np.ndarray:
    """Compute Breg (regularized centerline field) at all centerline points.

    Uses simsopt ``B_regularized_pure`` with ``regularization_rect(width, height)``
    to obtain the finite, regularized field along the coil centerline. This
    avoids the filamentary singularity and is combined with B0, Bκ, Bb for
    the full internal field model.

    Parameters
    ----------
    coil : Coil
        simsopt coil (curve + current).
    width, height : float
        Full cross-section dimensions [m] passed to ``regularization_rect``.

    Returns
    -------
    np.ndarray
        Breg at centerline points, shape ``(n_pts, 3)``.
    """
    from simsopt.field.selffield import B_regularized_pure, regularization_rect

    gamma = np.asarray(coil.curve.gamma(), dtype=float).reshape(-1, 3)
    gammadash = np.asarray(coil.curve.gammadash(), dtype=float).reshape(-1, 3)
    gammadashdash = np.asarray(coil.curve.gammadashdash(), dtype=float).reshape(-1, 3)
    n = len(gamma)
    quadpoints = getattr(coil.curve, "quadpoints", None)
    if quadpoints is None:
        quadpoints = np.linspace(0, 1, n, endpoint=False)
    else:
        quadpoints = np.asarray(coil.curve.quadpoints)
    reg = regularization_rect(width, height)
    if hasattr(reg, "__array__"):
        reg = float(np.asarray(reg))
    else:
        reg = float(reg)
    current = float(coil.current.get_value())
    Breg = B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, reg)
    return np.asarray(Breg, dtype=float).reshape(-1, 3)


def _compute_B_internal(
    coil_frame: dict[str, Any],
    Breg: np.ndarray,
    eval_coords: np.ndarray,
    nearest_idx: np.ndarray,
    width: float,
    height: float,
    delta: float,
) -> np.ndarray:
    r"""Compute full internal magnetic field at points inside the conductor.

    .. math::
        \mathbf{B} = \mathbf{B}_{\mathrm{reg}} + \mathbf{B}_0
        + \mathbf{B}_\kappa + \mathbf{B}_b

    For each evaluation point, local cross-section coordinates
    :math:`(u,v) \in [-1,1]` are computed via
    :math:`\boldsymbol{\delta} = \mathbf{x} - \boldsymbol{\gamma}_i`,
    :math:`u = (2/\mathtt{width})\,\boldsymbol{\delta}\cdot\mathbf{p}`,
    :math:`v = (2/\mathtt{height})\,\boldsymbol{\delta}\cdot\mathbf{q}`.

    Parameters
    ----------
    coil_frame : dict
        Output of :func:`_compute_coil_frame`.
    Breg : np.ndarray
        Regularized field from :func:`_compute_Breg_for_coil`.
    eval_coords : np.ndarray
        Evaluation points, shape ``(n, 3)`` (must lie inside conductor).
    nearest_idx : np.ndarray
        Centerline index for each eval point.
    width, height : float
        Cross-section dimensions [m].
    delta : float
        Regularisation parameter.

    Returns
    -------
    np.ndarray
        Total internal field, shape ``(n, 3)``.
    """
    gamma = coil_frame["gamma"]
    p = coil_frame["p"]
    q = coil_frame["q"]
    current = coil_frame["current"]
    kappa1 = coil_frame["kappa1"]
    kappa2 = coil_frame["kappa2"]
    kappa_b = coil_frame["kappa_b"]

    a = width
    b = height

    # Local (u,v) from displacement. Callers must pass only points with |u|<=1, |v|<=1.
    delta_vec = eval_coords - gamma[nearest_idx]
    u = (2 / width) * np.sum(delta_vec * p[nearest_idx], axis=1)
    v = (2 / height) * np.sum(delta_vec * q[nearest_idx], axis=1)

    p_ln = p[nearest_idx]
    q_ln = q[nearest_idx]
    kappa1_ln = kappa1[nearest_idx]
    kappa2_ln = kappa2[nearest_idx]
    kappa_b_ln = kappa_b[nearest_idx]

    B0 = _compute_B0(u, v, a, b, current, p_ln, q_ln)
    Bkappa = _compute_Bkappa(u, v, a, b, current, p_ln, q_ln, kappa1_ln, kappa2_ln)
    Bb = _compute_Bb(current, kappa_b_ln, delta)

    return Breg[nearest_idx] + B0 + Bkappa + Bb


def _build_coil_centerline_data(
    coils: list["Coil"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract and concatenate coil centerline geometry for nearest-neighbour lookup.

    Parameters
    ----------
    coils : list
        simsopt ``Coil`` objects.

    Returns
    -------
    all_gamma : np.ndarray
        Concatenated centerline sample points, shape ``(N, 3)``.
    all_tangents : np.ndarray
        Unit tangent vectors at each sample point, shape ``(N, 3)``.
    all_currents : np.ndarray
        Current magnitude at each sample point, shape ``(N,)``.
    coil_boundaries : np.ndarray
        Cumulative indices: coil k uses global indices in [coil_boundaries[k], coil_boundaries[k+1]).
        Length len(coils)+1.
    """
    gammas: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    currents: list[np.ndarray] = []
    boundaries = [0]
    for coil in coils:
        gamma = np.asarray(coil.curve.gamma(), dtype=float).reshape(-1, 3)
        gd = np.asarray(coil.curve.gammadash(), dtype=float).reshape(-1, 3)
        norms = np.linalg.norm(gd, axis=1, keepdims=True)
        norms = np.where(norms < 1e-14, 1.0, norms)
        gammas.append(gamma)
        tangents.append(gd / norms)
        currents.append(np.full(len(gamma), float(coil.current.get_value())))
        boundaries.append(boundaries[-1] + len(gamma))

    return (
        np.vstack(gammas),
        np.vstack(tangents),
        np.concatenate(currents),
        np.array(boundaries, dtype=np.intp),
    )


def _compute_jcross_b(
    coords: np.ndarray,
    coils: list["Coil"],
    bs: "BiotSavart",
    cross_section_area: float,
    *,
    width: float = 0.05,
    height: float = 0.05,
    use_regularized: bool = True,
    bs_mutual_list: list | None = None,
    cached_coil_frames: list[dict[str, Any]] | None = None,
    cached_Breg_list: list[np.ndarray] | None = None,
    mesh_coils: list["Coil"] | None = None,
    all_coils: list["Coil"] | None = None,
) -> np.ndarray:
    """Compute J × B Lorentz body-force at an array of spatial coordinates.

    When use_regularized=True (default): uses the Landreman et al. (2025)
    regularized internal field (Breg + B0 + Bkappa + Bb) for self-field and
    BiotSavart(other_coils) for mutual field. This gives physically correct
    finite values inside the conductor.

    When use_regularized=False: uses BiotSavart(all_coils) only (legacy
    filamentary model, diverges near centerline).

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates, shape ``(n_points, 3)``.
    coils : list
        simsopt ``Coil`` objects (used for B when mesh_coils is None).
    bs : BiotSavart
        Magnetic field evaluator.
    cross_section_area : float
        Winding-pack cross-section area [m²] (width × height).
    width : float
        Cross-section full width [m].
    height : float
        Cross-section full height [m].
    use_regularized : bool
        If True, use full internal field model; if False, legacy BiotSavart only.
    mesh_coils : list, optional
        Coils on which mesh points lie (e.g. unique coils only). When provided,
        used for cKDTree and J assignment so nearest-neighbor maps to correct
        coil.
    all_coils : list, optional
        Full coil set including symmetry copies (e.g. bfield.coils). When
        provided together with mesh_coils, used for mutual-field BiotSavart so
        that B includes contributions from all coils, not just other unique coils.

    Returns
    -------
    force : np.ndarray
        Body-force density, shape ``(n_points, 3)`` [N/m³].
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64, order="C")
    coils_for_tree = mesh_coils if mesh_coils is not None else coils
    all_gamma, all_tangents, all_currents, coil_boundaries = (
        _build_coil_centerline_data(coils_for_tree)
    )

    tree = cKDTree(all_gamma)
    _, nearest_idx = tree.query(coords)

    J_mag = all_currents[nearest_idx] / cross_section_area
    J_vec = J_mag[:, np.newaxis] * all_tangents[nearest_idx]

    if use_regularized:
        from simsopt.field import BiotSavart
        from simsopt.field.selffield import regularization_rect

        reg = float(np.asarray(regularization_rect(width, height)))
        delta = reg / (width * height)
        n_coils = len(coils_for_tree)
        if (
            cached_coil_frames is not None
            and cached_Breg_list is not None
            and len(cached_coil_frames) == n_coils
            and len(cached_Breg_list) == n_coils
        ):
            coil_frames = cached_coil_frames
            Breg_list = cached_Breg_list
        else:
            coil_frames = []
            Breg_list = []
            for coil in coils_for_tree:
                coil_frames.append(_compute_coil_frame(coil))
                Breg_list.append(_compute_Breg_for_coil(coil, width, height))

        # Map nearest_idx (global) -> coil_k; coil k has indices [coil_boundaries[k], coil_boundaries[k+1])
        coil_assignments = (
            np.searchsorted(coil_boundaries, nearest_idx, side="right") - 1
        )
        coil_assignments = np.clip(coil_assignments, 0, n_coils - 1)
        B = np.zeros((len(coords), 3))
        for k in range(n_coils):
            mask = coil_assignments == k
            if not np.any(mask):
                continue
            coords_k = coords[mask]
            local_idx = nearest_idx[mask] - coil_boundaries[k]
            gamma_k = coil_frames[k]["gamma"]
            p_k, q_k = coil_frames[k]["p"], coil_frames[k]["q"]
            delta_vec = coords_k - gamma_k[local_idx]
            u = (2 / width) * np.sum(delta_vec * p_k[local_idx], axis=1)
            v = (2 / height) * np.sum(delta_vec * q_k[local_idx], axis=1)
            inside = (np.abs(u) <= 1) & (np.abs(v) <= 1)
            B_self = np.zeros((len(coords_k), 3))
            if np.any(inside):
                B_self[inside] = _compute_B_internal(
                    coil_frames[k],
                    Breg_list[k],
                    coords_k[inside],
                    local_idx[inside],
                    width,
                    height,
                    delta,
                )
            # Points outside (|u|>1 or |v|>1) or far from conductor: B_self stays 0; only mutual
            # contributes. We never use BiotSavart for self-field (structural mesh is inside
            # winding pack; far points would indicate a bug).
            # Use all_coils (including symmetry copies) when provided for physically correct mutual B
            if all_coils is not None and mesh_coils is not None:
                coil_self = mesh_coils[k]
                others = [c for c in all_coils if c is not coil_self]
            else:
                others = [c for i, c in enumerate(coils_for_tree) if i != k]
            if others:
                if bs_mutual_list is not None:
                    bs_mutual_list[k].set_points(coords_k)
                    B_mutual = np.asarray(bs_mutual_list[k].B()).reshape(-1, 3)
                else:
                    bs_mutual = BiotSavart(others)
                    bs_mutual.set_points(coords_k)
                    B_mutual = np.asarray(bs_mutual.B()).reshape(-1, 3)
            else:
                B_mutual = np.zeros_like(B_self)
            B[mask] = B_self + B_mutual

        return np.cross(J_vec, B)

    # Legacy path: BiotSavart(all_coils) only
    bs.set_points(coords)
    B = np.asarray(bs.B()).reshape(-1, 3)
    return np.cross(J_vec, B)
