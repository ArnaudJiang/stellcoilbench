"""Adaptive search for initial coil geometry (R0, R1).

Iteratively adjusts coil major/minor radius until distance, linking, and
plasma-encirclement constraints are satisfied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from simsopt.geo import SurfaceRZFourier

from ..constants import (
    ADAPTIVE_CONVERGENCE_TOL,
    ADAPTIVE_TOLERANCE,
    CS_DISTANCE_MARGIN_LOOSE,
    CS_DISTANCE_MARGIN_TIGHT,
    CURRENT_SCALING_FACTOR,
    DEFAULT_COIL_QUADPOINTS,
    MAX_ADAPTIVE_ITERATIONS,
    MAX_LINKING_NUMBER,
    MAX_OSCILLATION_COUNT,
    MIN_DISTANCE_FRACTION,
    PLASMA_BOUNDARY_INNER_FACTOR,
    PLASMA_BOUNDARY_OUTER_FACTOR,
    R0_GROWTH_FACTOR,
    R0_SCALE_INIT,
    R0_SCALE_MAX,
    R0_SHRINK_FACTOR_GENTLE,
    R0_SHRINK_FACTOR_MILD,
    R1_GROWTH_FACTOR_LARGE,
    R1_GROWTH_FACTOR_SMALL,
    R1_SCALE_INIT,
    R1_SCALE_MAX,
)

if TYPE_CHECKING:
    pass


def _coils_via_symmetries_compat(
    base_curves: list,
    base_currents: list,
    nfp: int,
    stellsym: bool,
    regularizations: list | None = None,
) -> list:
    """Call ``coils_via_symmetries`` with fallback for older simsopt without *regularizations* kwarg.

    The ``regularizations`` keyword was added in the custom ``auglag_coils``
    branch of simsopt.  When running against an older or upstream version of
    simsopt that does not accept this keyword, ``coils_via_symmetries`` raises
    a ``TypeError`` for the unexpected argument.  We intentionally catch
    ``TypeError`` here so that the rest of the pipeline still works with
    either version.

    Parameters
    ----------
    base_curves : list
        Base curves for coil generation.
    base_currents : list
        Base currents for coil generation.
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether stellarator symmetry is used.
    regularizations : list | None, optional
        Regularization objects passed to ``coils_via_symmetries``.
        If *None*, the call is made without the keyword.

    Returns
    -------
    list
        Coils produced by ``coils_via_symmetries``.
    """
    from simsopt.field import coils_via_symmetries

    if regularizations is not None:
        try:
            return coils_via_symmetries(
                base_curves,
                base_currents,
                nfp,
                stellsym,
                regularizations=regularizations,
            )
        except TypeError:
            return coils_via_symmetries(base_curves, base_currents, nfp, stellsym)
    return coils_via_symmetries(base_curves, base_currents, nfp, stellsym)


def _normalize_current_weights(
    current_weights: list[float] | tuple[float, ...] | None,
    ncoils: int,
) -> list[float]:
    """Return positive per-coil current fractions that sum to one."""
    if current_weights is None:
        return [1.0 / ncoils for _ in range(ncoils)]
    if len(current_weights) != ncoils:
        raise ValueError(
            f"current_weights must have length ncoils={ncoils}, got {len(current_weights)}"
        )
    weights = [float(value) for value in current_weights]
    if any(value <= 0.0 for value in weights):
        raise ValueError("current_weights values must be positive")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("current_weights must have positive sum")
    return [value / total for value in weights]


def _make_base_currents(
    total_current: float,
    ncoils: int,
    current_weights: list[float] | tuple[float, ...] | None = None,
) -> list:
    """Create conditioned base currents for *ncoils* coils.

    Uses the ``Current(I * CURRENT_SCALING_FACTOR) * (1 / CURRENT_SCALING_FACTOR)``
    trick to improve numerical conditioning of the simsopt optimiser, then
    fixes the total-current constraint so that the last current is derived.

    Parameters
    ----------
    total_current : float
        Total current [A] to distribute among the coils.
    ncoils : int
        Number of base coils.

    Returns
    -------
    list
        List of *ncoils* simsopt ``Current`` objects.
    """
    from simsopt.field import Current

    inv_factor = 1.0 / CURRENT_SCALING_FACTOR
    weights = _normalize_current_weights(current_weights, ncoils)
    base = [
        Current(total_current * weights[idx] * CURRENT_SCALING_FACTOR) * inv_factor
        for idx in range(ncoils - 1)
    ]
    total_obj = Current(total_current)
    total_obj.fix_all()
    base.append(total_obj - sum(base))
    return base


def _adaptive_R0_R1_search(
    s: SurfaceRZFourier,
    ncoils: int,
    order: int,
    total_current: float,
    regularizations: list | None,
    numquadpoints: int = DEFAULT_COIL_QUADPOINTS,
    max_adaptive_iterations: int = MAX_ADAPTIVE_ITERATIONS,
    adaptive_tolerance: float = ADAPTIVE_TOLERANCE,
    major_radius_scale: float = 1.0,
    minor_radius_scale: float = 1.0,
    radial_offset: float = 0.0,
    current_weights: list[float] | tuple[float, ...] | None = None,
) -> Tuple[float, float]:
    """Find suitable R0 and R1 for coil initialization via adaptive search.

    Iteratively adjusts the coil major radius (R0) and minor radius (R1) until
    coils satisfy distance, linking, and interlink constraints.

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface.
    ncoils : int
        Number of base coils.
    order : int
        Fourier order for coil curves.
    total_current : float
        Total coil current for temporary coils.
    regularizations : list | None
        Regularization objects for coils_via_symmetries.
    numquadpoints : int, default=DEFAULT_COIL_QUADPOINTS
        Quadrature points along each base coil curve.
    max_adaptive_iterations : int, default=MAX_ADAPTIVE_ITERATIONS
        Maximum iterations for the adaptive search.
    adaptive_tolerance : float, default=ADAPTIVE_TOLERANCE
        Fractional tolerance for distance constraint checks.

    Returns
    -------
    tuple[float, float]
        (R0, R1) - major and minor radius for coil initialization.
    """
    from simsopt.geo import (
        create_equally_spaced_curves,
        CurveSurfaceDistance,
        CurveCurveDistance,
        LinkingNumber,
    )
    from ..path_utils import get_reference_radii

    major_radius, minor_radius_component = get_reference_radii(s)

    min_cs_distance = MIN_DISTANCE_FRACTION * major_radius
    min_cc_distance = MIN_DISTANCE_FRACTION * major_radius

    R0_scale = R0_SCALE_INIT
    R1_scale = R1_SCALE_INIT
    max_R0_scale = R0_SCALE_MAX
    max_R1_scale = R1_SCALE_MAX

    def scaled_radii() -> tuple[float, float]:
        return (
            major_radius * R0_scale * float(major_radius_scale) + float(radial_offset),
            minor_radius_component * R1_scale * float(minor_radius_scale),
        )

    R0, R1 = scaled_radii()

    prev_R0_scale = None
    prev_R1_scale = None
    oscillation_count = 0

    for _ in range(max_adaptive_iterations):
        if prev_R0_scale is not None and prev_R1_scale is not None:
            if (
                abs(R0_scale - prev_R0_scale) < ADAPTIVE_CONVERGENCE_TOL
                and abs(R1_scale - prev_R1_scale) < ADAPTIVE_CONVERGENCE_TOL
            ):
                oscillation_count += 1
                if oscillation_count >= MAX_OSCILLATION_COUNT:
                    break
            else:
                oscillation_count = 0

        prev_R0_scale = R0_scale
        prev_R1_scale = R1_scale

        base_curves = create_equally_spaced_curves(
            ncoils,
            s.nfp,
            stellsym=s.stellsym,
            R0=R0,
            R1=R1,
            order=order,
            numquadpoints=numquadpoints,
        )

        base_currents_temp = _make_base_currents(
            total_current, ncoils, current_weights
        )

        coils_temp = _coils_via_symmetries_compat(
            base_curves,
            base_currents_temp,
            s.nfp,
            s.stellsym,
            regularizations=regularizations,
        )

        curves_temp = [c.curve for c in coils_temp]

        cs_dist = CurveSurfaceDistance(curves_temp, s, 0.0)
        min_cs_sep = cs_dist.shortest_distance()

        cc_dist = CurveCurveDistance(curves_temp, 0.0, num_basecurves=ncoils)
        min_cc_sep = cc_dist.shortest_distance()

        link_num = LinkingNumber(curves_temp, downsample=2)
        linking_number = link_num.J()

        cs_ok = min_cs_sep >= min_cs_distance * (1 - adaptive_tolerance)
        cc_ok = min_cc_sep >= min_cc_distance * (1 - adaptive_tolerance)
        no_coil_interlink = abs(linking_number) < MAX_LINKING_NUMBER

        gamma = s.gamma()
        rs = np.sqrt(gamma[:, :, 0] ** 2 + gamma[:, :, 1] ** 2)
        R_min_surface = np.min(rs)
        R_max_surface = np.max(rs)

        coil_interlinks_plasma = False
        points_inside_hole_count = 0
        points_outside_plasma_count = 0

        for curve in base_curves:
            points = curve.gamma()
            radial_distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
            inside_hole_mask = (
                radial_distances < R_min_surface * PLASMA_BOUNDARY_INNER_FACTOR
            )
            outside_plasma_mask = (
                radial_distances > R_max_surface * PLASMA_BOUNDARY_OUTER_FACTOR
            )

            points_inside_hole_count += np.sum(inside_hole_mask)
            points_outside_plasma_count += np.sum(outside_plasma_mask)

            if np.any(inside_hole_mask) and np.any(outside_plasma_mask):
                coil_interlinks_plasma = True

        plasma_interlink_ok = cs_ok and coil_interlinks_plasma

        if cs_ok and cc_ok and no_coil_interlink and plasma_interlink_ok:
            break

        if R0_scale > max_R0_scale or R1_scale > max_R1_scale:
            R0_scale = min(R0_scale, max_R0_scale)
            R1_scale = min(R1_scale, max_R1_scale)
            R0, R1 = scaled_radii()
            break

        # Priority-based adjustment: fix only ONE constraint per iteration
        if not plasma_interlink_ok:
            if points_inside_hole_count == 0:
                R1_scale *= R1_GROWTH_FACTOR_LARGE
                if min_cs_sep > min_cs_distance * CS_DISTANCE_MARGIN_TIGHT:
                    R0_scale *= R0_SHRINK_FACTOR_MILD
                R0, R1 = scaled_radii()
            elif points_outside_plasma_count == 0:
                R1_scale *= R1_GROWTH_FACTOR_LARGE
                if min_cs_sep > min_cs_distance * CS_DISTANCE_MARGIN_LOOSE:
                    R0_scale *= R0_SHRINK_FACTOR_GENTLE
                R0, R1 = scaled_radii()
            else:
                R1_scale *= R1_GROWTH_FACTOR_SMALL
                R0, R1 = scaled_radii()
        elif not cs_ok:
            R0_scale *= R0_GROWTH_FACTOR
            R0, R1 = scaled_radii()
        elif not cc_ok:
            R0_scale *= R0_GROWTH_FACTOR
            R0, R1 = scaled_radii()
        elif not no_coil_interlink:
            R0_scale *= R0_GROWTH_FACTOR
            R0, R1 = scaled_radii()

    return R0, R1
