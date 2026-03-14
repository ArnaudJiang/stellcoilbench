"""
Constraint objective builders for coil optimization.

Builds flux, distance, length, curvature, force, and torque constraint
objectives from coil_objective_terms. Computes constraint scaling for
dimensionless optimization across reactor scales.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ._thresholds import _get_base_scaling_for_term
from ..mpi_utils import proc0_warning

if TYPE_CHECKING:
    from simsopt.geo import SurfaceRZFourier


def _identity(obj: Any, thresh: float) -> Any:
    """Return *obj* unchanged — used for l1/lp penalty terms."""
    return obj


def _make_qp(max_thresh: float | None = 0.0) -> Callable[[Any, float], Any]:
    """Factory for QuadraticPenalty(obj, thresh, 'max'). max_thresh=None uses caller thresh."""

    def _qp(obj: Any, thresh: float) -> Any:
        from simsopt.objectives import QuadraticPenalty

        t = max_thresh if max_thresh is not None else thresh
        return QuadraticPenalty(obj, t, "max")

    return _qp


def _make_sum_qp(max_thresh: float | None = 0.0) -> Callable[[Any, float], Any]:
    """Factory for sum of QuadraticPenalty over list. max_thresh=None uses caller thresh."""

    def _sum_qp(obj: Any, thresh: float) -> Any:
        from simsopt.objectives import QuadraticPenalty

        t = max_thresh if max_thresh is not None else thresh
        return sum(QuadraticPenalty(j, t, "max") for j in obj)

    return _sum_qp


_qp_max_zero = _make_qp(0.0)
_qp_max_thresh = _make_qp(None)
_sum_qp_max_zero = _make_sum_qp(0.0)
_sum_qp_max_thresh = _make_sum_qp(None)


def _sum_identity(obj: Any, thresh: float) -> Any:
    """Sum of a list of objectives — used for l1 on per-coil terms."""
    return sum(obj)


_TERM_MAP: dict[str, dict[str, Callable[[Any, float], Any]]] = {
    "total_length": {
        "l1": _identity,
        "l1_threshold": _identity,
        "l2": _qp_max_zero,
        "l2_threshold": _qp_max_thresh,
    },
    "coil_curvature": {
        "lp": _identity,
        "lp_threshold": _identity,
    },
    "coil_arclength_variation": {
        "l2": _sum_qp_max_zero,
        "l2_threshold": _sum_qp_max_thresh,
        "l1": _sum_identity,
        "l1_threshold": _sum_identity,
    },
    "coil_mean_squared_curvature": {
        "l2": _sum_qp_max_zero,
        "l2_threshold": _sum_qp_max_thresh,
        "l1": _sum_identity,
        "l1_threshold": _sum_identity,
    },
    "linking_number": {
        "": _identity,
    },
    "coil_coil_force": {
        "lp": _identity,
        "lp_threshold": _identity,
    },
    "coil_coil_torque": {
        "lp": _identity,
        "lp_threshold": _identity,
    },
    "coil_torsion": {
        "lp": _identity,
        "lp_threshold": _identity,
    },
    "structural_stress": {
        "l1": _identity,
        "l1_threshold": _identity,
        "l2": _qp_max_zero,
        "l2_threshold": _qp_max_thresh,
    },
}
"""Module-level mapping of term names to their penalty-builder functions.

Each entry maps an option string (e.g. ``"l2"``, ``"lp_threshold"``) to a
callable ``(obj, thresh) -> constraint_objective``.
"""

_NAME_MAP: dict[str, str] = {
    "total_length": "Length",
    "coil_mean_squared_curvature": "MSC",
    "coil_arclength_variation": "Arclength Var",
    "coil_curvature": "κ",
    "coil_torsion": "ζ",
    "linking_number": "Link #",
    "coil_coil_force": "Force",
    "coil_coil_torque": "Torque",
    "structural_stress": "σ_vm",
}
"""Human-readable display names for each constraint term."""


def _build_constraint_from_term(
    term_name: str,
    term_value: str,
    obj_map: dict[str, Any],
    thresh_map: dict[str, float | None],
    coil_objective_terms: dict[str, Any],
) -> tuple[Any, int, str | None] | None:
    """Build a single constraint from a coil_objective_terms entry.

    Encapsulates _TERM_MAP lookup, penalty_fn call, and p_value extraction.
    Returns (constraint, p_value, display_name) or None if term should be skipped.

    Parameters
    ----------
    term_name, term_value : str
        Entry from coil_objective_terms (e.g. "total_length", "l2_threshold").
    obj_map : dict
        Mapping of term names to simsopt objectives (Jls, Jcs, etc.).
    thresh_map : dict
        Mapping of term names to threshold values.
    coil_objective_terms : dict
        Full config for p-value lookup.

    Returns
    -------
    tuple or None
        (constraint_obj, p_value, display_name) or None if skipped.
    """
    if term_name.endswith("_p"):
        return None
    if term_name not in _TERM_MAP:
        return None
    if term_name == "structural_stress":
        return None
    if term_name == "coil_torsion" and "coil_torsion" not in obj_map:
        return None

    penalty_options = _TERM_MAP[term_name]
    if term_value not in penalty_options:
        proc0_warning(f"Unknown option '{term_value}' for {term_name}, skipping")
        return None

    penalty_fn = penalty_options[term_value]
    thresh = thresh_map.get(term_name) or 0.0
    constraint = penalty_fn(obj_map[term_name], thresh)

    p_value = 2
    if term_value in ["lp", "lp_threshold"]:
        p_value = coil_objective_terms.get(f"{term_name}_p", 2)

    display_name = _NAME_MAP.get(term_name)
    return (constraint, p_value, display_name)


def _extract_p_values(
    coil_objective_terms: dict[str, Any] | None,
) -> tuple[int, int, int, int]:
    """Extract curvature, force, torque, and torsion Lp-norm exponents from config.

    Parameters
    ----------
    coil_objective_terms : dict[str, Any] | None
        Mapping of objective term names to their option strings.  May be
        ``None``, in which case all exponents default to ``2``.

    Returns
    -------
    tuple[int, int, int, int]
        ``(curvature_p, force_p, torque_p, torsion_p)``.
    """
    if coil_objective_terms is None:
        return 2, 2, 2, 2
    curvature_p: int = coil_objective_terms.get("coil_curvature_p", 2)
    force_p: int = coil_objective_terms.get("coil_coil_force_p", 2)
    torque_p: int = coil_objective_terms.get("coil_coil_torque_p", 2)
    torsion_p: int = coil_objective_terms.get("coil_torsion_p", 2)
    return curvature_p, force_p, torque_p, torsion_p


def _compute_constraint_scaling_for_term(
    term_name: str,
    term_value: str,
    major_radius: float,
    total_current: float,
    p_value: int,
    base_scaling: float,
) -> float:
    """
    Compute scaling factor to make weight * constraint dimensionless.

    Different formulas for l2/l2_threshold vs lp/lp_threshold. Ensures
    optimization is scale-invariant across reactor sizes.

    Parameters
    ----------
    term_name : str
        Constraint name (e.g. total_length, coil_curvature, coil_coil_force).
    term_value : str
        Option (l2, l2_threshold, lp, lp_threshold, l1, l1_threshold, "").
    major_radius : float
        Plasma major radius [m].
    total_current : float
        Total coil current [A].
    p_value : int
        Lp norm exponent for lp/lp_threshold terms.
    base_scaling : float
        Base scaling from _get_base_scaling_for_term.

    Returns
    -------
    float
        Scaling factor to multiply constraint weight.
    """
    if term_value in ["l2", "l2_threshold"]:
        if term_name == "total_length":
            return base_scaling / major_radius
        elif term_name in ["coil_curvature", "coil_torsion"]:
            return base_scaling * major_radius
        elif term_name == "coil_mean_squared_curvature":
            return base_scaling * (major_radius**2)
        elif term_name == "coil_arclength_variation":
            return base_scaling / (major_radius**2)
        return base_scaling
    elif term_value in ["lp", "lp_threshold"]:
        if term_name in ["coil_curvature", "coil_torsion"]:
            return major_radius ** (p_value - 1)
        elif term_name in ["coil_coil_force", "coil_coil_torque"]:
            # Simsopt LpCurveForce/LpCurveTorque output (MN/m)^p and (MN)^p; cancel 1e6^p
            return (
                (major_radius ** (p_value - 1))
                * (1e6**p_value)
                / (total_current ** (2 * p_value))
            )
        elif term_name in [
            "total_length",
            "coil_coil_distance",
            "coil_surface_distance",
        ]:
            return base_scaling / (major_radius ** (p_value - 1))
        elif term_name == "coil_mean_squared_curvature":
            return base_scaling * (major_radius ** (2 * p_value - 2))
        elif term_name == "coil_arclength_variation":
            return base_scaling / (major_radius ** (2 * p_value - 2))
        return base_scaling
    elif term_value == "":
        return base_scaling
    else:
        return base_scaling


def _build_c_list_and_constraint_scaling_from_coil_objective_terms(
    Jf: Any,
    Jccdist: Any,
    Jcsdist: Any,
    Jls: list,
    Jcs: list,
    Jalenvar: list,
    Jmscs: list,
    Jlink: Any,
    Jforce: Any,
    Jtorque: Any,
    coil_objective_terms: dict[str, Any] | None,
    thresholds: dict[str, float],
    major_radius: float,
    total_current: float,
    *,
    Jts: list | None = None,
) -> tuple[list, dict[int, float], int, int, list, dict[int, str]]:
    """
    Build constraint list and scaling from coil_objective_terms.

    Always includes flux (Jf), coil-coil distance (Jccdist), coil-surface distance
    (Jcsdist). Adds length, curvature, arclength_variation, MSC, linking_number,
    force, torque based on coil_objective_terms. Computes constraint_scaling for
    dimensionless weights.

    Parameters
    ----------
    Jf, Jccdist, Jcsdist : objectives
        Flux and distance objectives.
    Jls, Jcs, Jalenvar, Jmscs : list
        Per-coil objectives (length, curvature, arclength variation, MSC).
    Jlink, Jforce, Jtorque : objectives
        Linking number and force/torque.
    Jts : list | None
        Per-coil LpCurveTorsion objectives.  If None, coil_torsion term is skipped.
    coil_objective_terms : dict[str, Any] | None
        Case config specifying which terms and options (l2, lp_threshold, etc.).
    thresholds : dict[str, float]
        Threshold values for each constraint type.
    major_radius, total_current : float
        For constraint scaling.

    Returns
    -------
    tuple
        (c_list, constraint_scaling, cc_distance_idx, cs_distance_idx,
         constraint_names_and_thresholds, constraint_idx_to_term).
    """
    c_list = [Jf]
    cc_distance_idx = len(c_list)
    c_list.append(Jccdist)
    cs_distance_idx = len(c_list)
    c_list.append(Jcsdist)
    constraint_names_and_thresholds = [
        ("CC Distance", thresholds["cc_threshold"]),
        ("CS Distance", thresholds["cs_threshold"]),
    ]
    constraint_scaling = {
        cc_distance_idx: 1.0 / (major_radius**2),
        cs_distance_idx: 1.0 / (major_radius**2),
    }
    constraint_idx_to_term = {}

    if not coil_objective_terms:
        return (
            c_list,
            constraint_scaling,
            cc_distance_idx,
            cs_distance_idx,
            constraint_names_and_thresholds,
            constraint_idx_to_term,
        )

    length_threshold = thresholds["length_threshold"]
    curvature_threshold = thresholds["curvature_threshold"]
    torsion_threshold = thresholds.get("torsion_threshold", 0.0)
    arclength_variation_threshold = thresholds.get("arclength_variation_threshold", 0.0)
    msc_threshold = thresholds["msc_threshold"]
    force_threshold = thresholds["force_threshold"]
    torque_threshold = thresholds["torque_threshold"]

    obj_map: dict[str, Any] = {
        "total_length": sum(Jls),
        "coil_curvature": sum(Jcs),
        "coil_arclength_variation": Jalenvar,
        "coil_mean_squared_curvature": Jmscs,
        "linking_number": Jlink,
        "coil_coil_force": Jforce,
        "coil_coil_torque": Jtorque,
    }
    thresh_map: dict[str, float | None] = {
        "total_length": length_threshold,
        "coil_curvature": curvature_threshold,
        "coil_arclength_variation": arclength_variation_threshold,
        "coil_mean_squared_curvature": msc_threshold,
        "linking_number": None,
        "coil_coil_force": force_threshold,
        "coil_coil_torque": torque_threshold,
    }
    if Jts is not None:
        obj_map["coil_torsion"] = sum(Jts)
        thresh_map["coil_torsion"] = torsion_threshold

    for term_name, term_value in coil_objective_terms.items():
        result = _build_constraint_from_term(
            term_name,
            term_value,
            obj_map,
            thresh_map,
            coil_objective_terms,
        )
        if result is None:
            continue

        constraint, p_value, display_name = result
        constraint_idx = len(c_list)
        c_list.append(constraint)
        base_scaling = _get_base_scaling_for_term(term_name, major_radius)
        constraint_scaling[constraint_idx] = _compute_constraint_scaling_for_term(
            term_name, term_value, major_radius, total_current, p_value, base_scaling
        )
        if display_name is not None:
            constraint_names_and_thresholds.append(
                (display_name, thresh_map[term_name])
            )
        constraint_idx_to_term[constraint_idx] = term_name

    return (
        c_list,
        constraint_scaling,
        cc_distance_idx,
        cs_distance_idx,
        constraint_names_and_thresholds,
        constraint_idx_to_term,
    )


def _resolve_force_torque_thresholds(
    force_threshold: float,
    torque_threshold: float,
    coil_objective_terms: dict[str, Any] | None,
) -> tuple[float, float]:
    """
    Resolve effective force and torque thresholds from config.

    When ``coil_objective_terms`` specifies an ``lp_threshold`` variant for
    force/torque the caller-supplied *force_threshold* / *torque_threshold*
    is kept.  For the plain ``lp`` variant (no threshold) the effective
    threshold is set to ``0.0``.  If ``coil_objective_terms`` is ``None``
    or the term is absent, the original thresholds are returned unchanged.

    Parameters
    ----------
    force_threshold : float
        Base force threshold from the case config [N].
    torque_threshold : float
        Base torque threshold from the case config [N·m].
    coil_objective_terms : dict[str, Any] | None
        Mapping of objective term names to their option strings
        (e.g. ``"lp"``, ``"lp_threshold"``).  May be ``None``.

    Returns
    -------
    tuple[float, float]
        ``(force_thresh, torque_thresh)`` — the effective thresholds to
        pass to ``LpCurveForce`` / ``LpCurveTorque``.
    """
    force_thresh: float = force_threshold
    torque_thresh: float = torque_threshold

    if coil_objective_terms:
        if coil_objective_terms.get("coil_coil_force") and "threshold" in str(
            coil_objective_terms.get("coil_coil_force", "")
        ):
            force_thresh = force_threshold
        else:
            force_thresh = 0.0
        if coil_objective_terms.get("coil_coil_torque") and "threshold" in str(
            coil_objective_terms.get("coil_coil_torque", "")
        ):
            torque_thresh = torque_threshold
        else:
            torque_thresh = 0.0

    return force_thresh, torque_thresh


def _build_modular_coil_constraint_objects(
    curves: list,
    base_curves: list,
    coils: list,
    ncoils: int,
    s: SurfaceRZFourier,
    cc_threshold: float,
    cs_threshold: float,
    curvature_threshold: float,
    torsion_threshold: float,
    force_threshold: float,
    torque_threshold: float,
    coil_objective_terms: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Build constraint objectives for modular (non-dipole) coils.

    Creates CurveCurveDistance, CurveSurfaceDistance, LinkingNumber,
    LpCurveCurvature, MeanSquaredCurvature, ArclengthVariation, LpCurveForce,
    LpCurveTorque. Force/torque thresholds are set to 0 for lp (no threshold)
    or to force_threshold/torque_threshold for lp_threshold.

    Parameters
    ----------
    curves, base_curves : list
        All curves and base (unique) curves.
    coils : list
        Coil objects (for force/torque).
    ncoils : int
        Number of base coils.
    s : Surface
        Plasma surface (for coil-surface distance).
    cc_threshold, cs_threshold, curvature_threshold, torsion_threshold : float
        Distance, curvature, and torsion thresholds.
    force_threshold, torque_threshold : float
        Force/torque thresholds (used only for lp_threshold option).
    coil_objective_terms : dict[str, Any] | None
        Case config for curvature_p, force_p, torque_p, torsion_p and lp vs lp_threshold.

    Returns
    -------
    dict[str, Any]
        Keys: Jls, Jccdist, Jcsdist, Jalenvar, Jcs, Jts, Jlink, Jforce, Jtorque, Jmscs.
    """
    from simsopt.geo import (
        CurveCurveDistance,
        CurveSurfaceDistance,
        LinkingNumber,
        LpCurveCurvature,
        LpCurveTorsion,
        CurveLength,
        ArclengthVariation,
        MeanSquaredCurvature,
    )
    from simsopt.field.force import LpCurveForce, LpCurveTorque

    curvature_p, force_p, torque_p, torsion_p = _extract_p_values(coil_objective_terms)
    # force_threshold, torque_threshold are already resolved in get_full_thresholds
    force_thresh, torque_thresh = force_threshold, torque_threshold

    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, cc_threshold, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, cs_threshold)
    Jalenvar = [ArclengthVariation(c) for c in base_curves]
    Jcs = [LpCurveCurvature(c, curvature_p, curvature_threshold) for c in base_curves]
    Jts = [LpCurveTorsion(c, torsion_p, torsion_threshold) for c in base_curves]
    Jlink = LinkingNumber(curves, downsample=2)
    Jforce = LpCurveForce(
        coils[:ncoils], coils, p=force_p, threshold=force_thresh, downsample=2
    )
    Jtorque = LpCurveTorque(
        coils[:ncoils], coils, p=torque_p, threshold=torque_thresh, downsample=2
    )
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

    return {
        "Jls": Jls,
        "Jccdist": Jccdist,
        "Jcsdist": Jcsdist,
        "Jalenvar": Jalenvar,
        "Jcs": Jcs,
        "Jts": Jts,
        "Jlink": Jlink,
        "Jforce": Jforce,
        "Jtorque": Jtorque,
        "Jmscs": Jmscs,
    }
