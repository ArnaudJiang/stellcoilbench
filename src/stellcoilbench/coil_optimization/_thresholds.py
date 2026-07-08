"""
Threshold computation for coil optimization.

Scales constraint thresholds by plasma minor radius so that constraints
are dimensionless across reactor scales (ARIES-CS reference).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from ..config_scheme import ARIES_CS_MINOR_RADIUS
from ..path_utils import get_reference_radii

if TYPE_CHECKING:
    from simsopt.geo import SurfaceRZFourier


_MAX_FINITE_BUILD_WIDTH: float = 0.35
"""Winding-pack cross-section width [m] at ARIES-CS reactor scale (a0 = 1)."""

_MIN_FINITE_BUILD_WIDTH: float = 0.05
"""Absolute lower bound for finite-build width [m] after scaling."""

_MIN_CC_TO_FB_RATIO: float = 2.1
"""cc_threshold must be at least this many times the finite-build width (extra space between coils)."""


def _compute_thresholds_from_surface(
    s: "SurfaceRZFourier",
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute constraint thresholds scaled by plasma minor radius.

    Geometric thresholds are specified at **ARIES-CS reactor scale** (a = 1.7 m)
    and are always rescaled to device scale via ``a0 = ARIES_CS_MINOR_RADIUS /
    minor_radius``.  This applies regardless of whether values come from
    built-in defaults or from explicit user/autopilot overrides in kwargs.

    Scaling rules:
      - Length-like thresholds (length, cc, cs): divided by a0
      - *_threshold_device overrides: used directly in device scale
      - Curvature-like thresholds (curvature, msc): multiplied by a0
      - Force threshold (N/m, ∝ I²/d): divided by a0
      - Torque threshold (MN, ∝ I²/d·r ∝ I²): no a0 scaling
      - flux_threshold: used as-is (dimensionless)
      - finite_build_width: max(_MAX_FINITE_BUILD_WIDTH / a0, 0.05 m)
      - cc_threshold: divided by a0, then clamped ≥ 2.1 × finite_build_width

    When ``finite_build_width`` is in kwargs (e.g. from CLI --finite-build-width
    or case post_processing), it is used directly (device scale [m], no a0
    scaling) and cc_threshold is set to 2.1 × finite_build_width.

    Note: force/torque thresholds receive only the geometric a0 scaling here.
    The current-dependent scaling (current_scale_factor = (I_device/I_reactor)²)
    is applied separately in ``_optimize_coils_loop_impl``.

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface (used for major/minor radius).
    kwargs : Dict[str, Any]
        Reactor-scale threshold overrides (e.g. cc_threshold, cs_threshold).

    Returns
    -------
    Dict[str, Any]
        Thresholds dict with keys: length_threshold, flux_threshold, cc_threshold,
        cs_threshold, msc_threshold, curvature_threshold, force_threshold,
        torque_threshold, finite_build_width, major_radius, minor_radius, a0.
    """
    length_threshold = kwargs.get("length_threshold", 200.0)
    flux_threshold = kwargs.get("flux_threshold", 1e-8)
    cc_threshold = kwargs.get("cc_threshold", 0.8)
    cs_threshold = kwargs.get("cs_threshold", 1.3)
    msc_threshold = kwargs.get("msc_threshold", 1.0)
    curvature_threshold = kwargs.get("curvature_threshold", 1.0)
    torsion_threshold = kwargs.get("torsion_threshold", 1.0)
    length_variance_threshold = kwargs.get("length_variance_threshold", 0.0)
    force_threshold = kwargs.get("force_threshold", 200.0)
    torque_threshold = kwargs.get("torque_threshold", 200.0)

    major_radius, minor_radius = get_reference_radii(s)
    minor_radius = float(minor_radius)
    a0 = ARIES_CS_MINOR_RADIUS / minor_radius

    # When user specifies finite_build_width: use directly (device scale)
    user_fb = kwargs.get("finite_build_width")
    if user_fb is not None:
        finite_build_width = float(user_fb)
        if "cc_threshold" in kwargs:
            cc_user_device = cc_threshold / a0
            cc_threshold = max(cc_user_device, _MIN_CC_TO_FB_RATIO * finite_build_width)
        else:
            cc_threshold = _MIN_CC_TO_FB_RATIO * finite_build_width
        length_threshold /= a0
        cs_threshold /= a0
        curvature_threshold *= a0
        torsion_threshold *= a0
        msc_threshold *= a0
        force_threshold /= a0
    else:
        length_threshold /= a0
        cc_threshold /= a0
        cs_threshold /= a0
        curvature_threshold *= a0
        torsion_threshold *= a0
        msc_threshold *= a0
        force_threshold /= a0
        finite_build_width = max(_MAX_FINITE_BUILD_WIDTH / a0, _MIN_FINITE_BUILD_WIDTH)
        cc_threshold = max(cc_threshold, _MIN_CC_TO_FB_RATIO * finite_build_width)
    device_overrides = {
        "length_threshold": "length_threshold_device",
        "cc_threshold": "cc_threshold_device",
        "cs_threshold": "cs_threshold_device",
        "curvature_threshold": "curvature_threshold_device",
        "torsion_threshold": "torsion_threshold_device",
        "msc_threshold": "msc_threshold_device",
        "force_threshold": "force_threshold_device",
        "torque_threshold": "torque_threshold_device",
        "length_variance_threshold": "length_variance_threshold_device",
    }
    values = {
        "length_threshold": length_threshold,
        "cc_threshold": cc_threshold,
        "cs_threshold": cs_threshold,
        "curvature_threshold": curvature_threshold,
        "torsion_threshold": torsion_threshold,
        "msc_threshold": msc_threshold,
        "force_threshold": force_threshold,
        "torque_threshold": torque_threshold,
        "length_variance_threshold": length_variance_threshold / (a0**2),
    }
    for target_key, override_key in device_overrides.items():
        if kwargs.get(override_key) is not None:
            values[target_key] = float(kwargs[override_key])

    return {
        "length_threshold": values["length_threshold"],
        "flux_threshold": flux_threshold,
        "cc_threshold": values["cc_threshold"],
        "cs_threshold": values["cs_threshold"],
        "msc_threshold": values["msc_threshold"],
        "curvature_threshold": values["curvature_threshold"],
        "torsion_threshold": values["torsion_threshold"],
        "force_threshold": values["force_threshold"],
        "torque_threshold": values["torque_threshold"],
        "length_variance_threshold": values["length_variance_threshold"],
        "finite_build_width": finite_build_width,
        "major_radius": major_radius,
        "minor_radius": minor_radius,
        "a0": a0,
    }


def _get_optimization_thresholds(
    s: "SurfaceRZFourier",
    kwargs: Dict[str, Any],
    *,
    is_continuation_step: bool = False,
    cached: Dict[str, Any] | None = None,
    coil_width_default: float = 0.4,
) -> Dict[str, Any]:
    """
    Get full optimization thresholds for the coil optimization loop.

    On continuation steps, uses cached thresholds when available. Otherwise
    computes from surface and adds arclength_variation_threshold and coil_width.

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface.
    kwargs : Dict[str, Any]
        User overrides and _cached_thresholds for continuation.
    is_continuation_step : bool, optional
        If True and cached is provided, return cached thresholds.
    cached : Dict[str, Any] | None, optional
        Cached thresholds from previous Fourier continuation step.
    coil_width_default : float, optional
        Default coil width before minor-radius scaling.

    Returns
    -------
    Dict[str, Any]
        Full thresholds dict including arclength_variation_threshold and coil_width.
    """
    if is_continuation_step and cached is not None:
        mjr, mnr = get_reference_radii(s)
        return {
            **cached,
            "major_radius": cached.get("major_radius", mjr),
            "minor_radius": cached.get("minor_radius", mnr),
            "a0": cached.get("a0", ARIES_CS_MINOR_RADIUS / float(mnr)),
        }
    th = _compute_thresholds_from_surface(s, kwargs)
    if kwargs.get("arclength_variation_threshold_device") is not None:
        th["arclength_variation_threshold"] = float(
            kwargs["arclength_variation_threshold_device"]
        )
    else:
        th["arclength_variation_threshold"] = kwargs.get(
            "arclength_variation_threshold", 0.0
        )
        th["arclength_variation_threshold"] *= th["a0"] ** 2
    th["coil_width"] = coil_width_default / th["a0"]
    return th


def _get_base_scaling_for_term(term_name: str, major_radius: float) -> float:
    """
    Return base scaling for l1/l1_threshold (linear penalty) terms.

    Parameters
    ----------
    term_name : str
        Constraint name (e.g. total_length, coil_curvature).
    major_radius : float
        Plasma major radius [m].

    Returns
    -------
    float
        Base scaling factor (1/R0, 1/R0^2, R0, R0^2, or 1.0).
    """
    if term_name == "total_length":
        return 1.0 / major_radius
    elif term_name in ["coil_coil_distance", "coil_surface_distance"]:
        return 1.0 / (major_radius**2)
    elif term_name in ["coil_curvature", "coil_torsion"]:
        return major_radius
    elif term_name == "coil_mean_squared_curvature":
        return major_radius**2
    elif term_name == "coil_length_variance":
        return 1.0 / (major_radius**2)
    elif term_name == "coil_arclength_variation":
        return 1.0 / (major_radius**2)
    elif term_name == "linking_number":
        return 1.0
    elif term_name in ["coil_coil_force", "coil_coil_torque"]:
        return 1.0
    return 1.0


def get_full_thresholds(
    s: "SurfaceRZFourier",
    kwargs: Dict[str, Any],
    *,
    is_continuation_step: bool = False,
    cached: Dict[str, Any] | None = None,
    coil_objective_terms: Dict[str, Any] | None = None,
    coil_width_default: float = 0.4,
) -> Dict[str, Any]:
    """Compute full optimization thresholds with force/torque resolution.

    Single source of truth for threshold computation. Merges:
    - ARIES-CS scaling via plasma minor radius
    - User overrides from kwargs (threshold_kwargs from case config)
    - Force/torque resolution from coil_objective_terms (lp vs lp_threshold)

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface (for major/minor radius scaling).
    kwargs : Dict[str, Any]
        Threshold overrides (length_threshold, cc_threshold, etc.) and
        optional _cached_thresholds for Fourier continuation.
    is_continuation_step : bool, optional
        If True and cached provided, return cached thresholds.
    cached : Dict[str, Any] | None, optional
        Cached thresholds from previous Fourier continuation step.
    coil_objective_terms : Dict[str, Any] | None, optional
        Objective term config for force/torque resolution (lp vs lp_threshold).
    coil_width_default : float, optional
        Default coil width before scaling.

    Returns
    -------
    Dict[str, Any]
        Full thresholds with force_threshold and torque_threshold resolved.
    """
    from ._constraint_builders import _resolve_force_torque_thresholds

    th = _get_optimization_thresholds(
        s,
        kwargs,
        is_continuation_step=is_continuation_step,
        cached=cached,
        coil_width_default=coil_width_default,
    )
    force_thresh, torque_thresh = _resolve_force_torque_thresholds(
        th["force_threshold"],
        th["torque_threshold"],
        coil_objective_terms,
    )
    th["force_threshold"] = force_thresh
    th["torque_threshold"] = torque_thresh
    return th
