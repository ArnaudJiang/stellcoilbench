"""Reactor-scale constraints and composite scoring."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

N_TURNS_MODEL: int = 500
REACTOR_SCALE_CONSTRAINTS: List[Dict[str, Any]] = [
    {
        "metric": "coils_linked_to_surface",
        "source": "metrics",
        "bound": True,
        "direction": "eq",
        "hard": True,
        "label": "Coils linked to plasma surface",
        "units": "(boolean)",
    },
    {
        "metric": "final_linking_number",
        "source": "metrics",
        "bound": 0.5,
        "direction": "max",
        "transform": abs,
        "hard": True,
        "label": r"Coil-coil linking number (:math:`|\text{LN}| \approx 0`)",
        "units": "(dimensionless)",
    },
    {
        "metric": "avg_BdotN_over_B",
        "source": "metrics",
        "bound": 1e-2,
        "direction": "max",
        "label": r"avg :math:`\bar{B}_n`",
        "units": "(dimensionless)",
        "composite_score_label": r"avg :math:`\langle B{\cdot}n\rangle / \langle B\rangle`",
    },
    {
        "metric": "reactor_scale_min_cs_separation",
        "source": "reactor_scale_metrics",
        "bound": 1.3,
        "direction": "min",
        "label": "Minimum coil-surface distance",
        "units": "m",
        "composite_score_label": "Min coil-surface distance",
    },
    {
        "metric": "reactor_scale_min_cc_separation",
        "source": "reactor_scale_metrics",
        "bound": 0.7,
        "direction": "min",
        "label": "Minimum coil-coil distance",
        "units": "m",
        "composite_score_label": "Min coil-coil distance",
    },
    {
        "metric": "reactor_scale_total_length",
        "source": "reactor_scale_metrics",
        "bound": 220.0,
        "direction": "max",
        "label": "Total coil length",
        "units": "m",
        "margin_value_rst": r"L",
    },
    {
        "metric": "reactor_scale_max_curvature",
        "source": "reactor_scale_metrics",
        "bound": 1.0,
        "direction": "max",
        "label": r"Max curvature :math:`\kappa`",
        "units": "m⁻¹",
        "margin_value_rst": r"\kappa_{\max}",
    },
    {
        "metric": "reactor_scale_mean_squared_curvature",
        "source": "reactor_scale_metrics",
        "bound": 1.0,
        "direction": "max",
        "transform": math.sqrt,
        "label": r"Max :math:`\sqrt{\text{MSC}}` (RMS curvature)",
        "units": "m⁻¹",
        "composite_score_label": r"RMS curvature :math:`\sqrt{\text{MSC}}`",
        "margin_value_rst": r"\sqrt{\text{MSC}}",
    },
    {
        "metric": "reactor_scale_arclength_variation",
        "source": "reactor_scale_metrics",
        "bound": 1.0,
        "direction": "max",
        "transform": math.sqrt,
        "label": r"Arclength variation :math:`\sqrt{\text{Var}}`",
        "units": "m",
        "margin_value_rst": r"\sqrt{\text{Var}}",
    },
    {
        "metric": "total_superconductor_length_km",
        "source": "reactor_scale_metrics",
        "bound": 100.0,
        "direction": "max",
        "label": r"Total superconductor length :math:`L_{\text{SC}}`",
        "units": "km",
        "margin_value_rst": r"L_{\text{SC}}",
    },
    {
        "metric": "N_turns_per_coil",
        "source": "reactor_scale_metrics",
        "bound": N_TURNS_MODEL,
        "direction": "max",
        "transform": lambda x: max(x) if isinstance(x, list) and x else 0,
        "hard": True,
        "label": f"Max turns per coil (:math:`N_{{\\text{{turns}}}} \\leq {N_TURNS_MODEL}`)",
        "units": "(turns)",
    },
    {
        "metric": "finite_build_cc_clearance",
        "source": "reactor_scale_metrics",
        "bound": 0.0,
        "direction": "min",
        "hard": True,
        "label": r"Finite-build coil-coil clearance (:math:`d_{\text{cc}} > w_{\text{WP}}`)",
        "units": "m",
    },
]


def _is_constraint_violated(
    value: Any,
    bound: Any,
    direction: str,
) -> bool:
    """Check whether a single constraint is violated.

    Parameters
    ----------
    value : numeric
        The metric value (possibly after a transform).
    bound : numeric or bool
        The constraint bound.
    direction : ``"max"`` | ``"min"`` | ``"eq"``
        Constraint direction.

    Returns
    -------
    bool
        True when the constraint is violated.
    """
    if direction == "max":
        return value > bound
    if direction == "min":
        return value < bound
    if direction == "eq":
        return value != bound
    return False


def _resolve_constraint_value(
    constraint: Dict[str, Any],
    metrics: Dict[str, Any],
    reactor_scale_metrics: Dict[str, Any],
) -> Any | None:
    """Look up the raw metric for *constraint* and apply its transform.

    Parameters
    ----------
    constraint : dict
        A single entry from :data:`REACTOR_SCALE_CONSTRAINTS`.
    metrics : dict
        Device-scale metrics.
    reactor_scale_metrics : dict
        Reactor-scale metrics.

    Returns
    -------
    value or None
        Transformed value, or *None* when the metric is absent.
    """
    source_dict = (
        reactor_scale_metrics
        if constraint["source"] == "reactor_scale_metrics"
        else metrics
    )
    raw_value = source_dict.get(constraint["metric"])
    if raw_value is None:
        return None
    transform = constraint.get("transform")
    return transform(raw_value) if transform is not None else raw_value


def normalize_submission_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize submission data to canonical structure.

    Ensures consistent layout for leaderboard processing:
    - Maps ``final_normalized_squared_flux`` → ``final_squared_flux``.
    - Promotes top-level metric keys into a ``metrics`` dict when missing.
    - Preserves ``metadata`` and ensures it exists.

    Parameters
    ----------
    data : dict
        Raw submission data (from results.json or zip contents).

    Returns
    -------
    dict
        Copy of data with normalized ``metrics`` and ``metadata`` keys.
    """
    out = dict(data)
    metrics = out.get("metrics") or {}
    if not metrics and (
        "final_squared_flux" in out or "final_normalized_squared_flux" in out
    ):
        metadata_keys = {
            "metadata",
            "method_version",
            "contact",
            "hardware",
            "run_date",
            "output_directory",
            "lagrange_multipliers",
            "iterations_used",
            "walltime_sec",
        }
        metrics = {k: v for k, v in out.items() if k not in metadata_keys}
        if not out.get("metadata"):
            out["metadata"] = {
                k: out.get(k) for k in ["contact", "hardware", "run_date"] if k in out
            }
    if (
        "final_normalized_squared_flux" in metrics
        and "final_squared_flux" not in metrics
    ):
        metrics["final_squared_flux"] = metrics.pop("final_normalized_squared_flux")
    out["metrics"] = metrics
    return out


def check_reactor_constraints(
    metrics: Dict[str, Any],
    reactor_scale_metrics: Dict[str, Any],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Check whether a submission meets all reactor-scale engineering constraints.

    Evaluates each constraint in :data:`REACTOR_SCALE_CONSTRAINTS` against
    the provided metrics. Hard violations indicate infeasibility; soft
    violations are recorded but do not affect the pass/fail flag.

    Parameters
    ----------
    metrics : dict
        Device-scale metrics (e.g. ``coils_linked_to_surface``, ``final_linking_number``).
    reactor_scale_metrics : dict
        Reactor-scale metrics (e.g. separations, length, curvature, turns).

    Returns
    -------
    passes_hard : bool
        True if no hard constraints are violated.
    violations : list of dict
        All violated constraints, each with ``label``, ``metric``, ``value``,
        ``bound``, ``direction``, ``units``, and ``hard``.
    """
    violations: List[Dict[str, Any]] = []
    for constraint in REACTOR_SCALE_CONSTRAINTS:
        value = _resolve_constraint_value(constraint, metrics, reactor_scale_metrics)
        if value is None:
            continue
        if _is_constraint_violated(value, constraint["bound"], constraint["direction"]):
            violations.append(
                {
                    "label": constraint["label"],
                    "metric": constraint["metric"],
                    "value": value,
                    "bound": constraint["bound"],
                    "direction": constraint["direction"],
                    "units": constraint.get("units", ""),
                    "hard": constraint.get("hard", False),
                }
            )
    has_hard_violation = any(v["hard"] for v in violations)
    return (not has_hard_violation), violations


def compute_composite_score(
    metrics: Dict[str, Any],
    reactor_scale_metrics: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Compute a composite feasibility/quality score.

    Combines soft constraints via a geometric mean of exponential margin
    factors. Hard constraint violations return 0.0 immediately.

    Score interpretation:
    - 0.0: Hard infeasibility (e.g. coils delinked, interlinked).
    - < 1: One or more soft constraints violated on average.
    - 1.0: All constraints met exactly.
    - > 1: Constraints met with engineering margin.

    Parameters
    ----------
    metrics : dict
        Device-scale metrics.
    reactor_scale_metrics : dict
        Reactor-scale metrics.

    Returns
    -------
    score : float | None
        Composite score; 0.0 for infeasible, > 0 for feasible, None when
        no soft-constraint metrics are available.
    details : dict
        Diagnostic info: ``factors`` (per-constraint margins), ``infeasible``,
        ``reason``, ``n_factors``, ``mean_margin``.
    """
    details: Dict[str, Any] = {"factors": {}, "infeasible": False}
    for c in REACTOR_SCALE_CONSTRAINTS:
        if not c.get("hard", False):
            continue
        value = _resolve_constraint_value(c, metrics, reactor_scale_metrics)
        if value is None:
            continue
        if _is_constraint_violated(value, c["bound"], c["direction"]):
            details["infeasible"] = True
            details["reason"] = f"{c['label']}: value={value}, bound={c['bound']}"
            return 0.0, details

    exponents: List[float] = []
    for c in REACTOR_SCALE_CONSTRAINTS:
        if c.get("hard", False):
            continue
        bound = c["bound"]
        if bound == 0:
            continue
        value = _resolve_constraint_value(c, metrics, reactor_scale_metrics)
        if value is None:
            continue
        if c["direction"] == "max":
            exponent = 1.0 - value / bound
        else:
            exponent = value / bound - 1.0
        exponents.append(exponent)
        details["factors"][c["metric"]] = {
            "value": float(value),
            "bound": float(bound),
            "direction": c["direction"],
            "margin": float(exponent),
            "factor": float(math.exp(exponent)),
        }

    if not exponents:
        details["reason"] = "No metrics available for scoring"
        return None, details
    mean_exponent = sum(exponents) / len(exponents)
    score = math.exp(mean_exponent)
    details["n_factors"] = len(exponents)
    details["mean_margin"] = float(mean_exponent)
    return float(score), details
