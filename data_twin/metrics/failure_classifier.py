"""Multi-label failure classifier for core Data Twin records."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from data_twin.metrics import MetricThresholds, as_bool, as_float, derive_geometry_metrics

FAILURE_LABELS = [
    "success",
    "optimizer_max_iter",
    "optimizer_nan",
    "optimizer_timeout",
    "poor_Bn",
    "curvature_exceeded",
    "torsion_spike",
    "coil_coil_too_close",
    "coil_plasma_too_close",
    "self_intersection",
    "missing_output",
    "runtime_error",
    "unknown_failure",
]


def _text_has(text: str, *needles: str) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _is_nan_metric(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def classify_primary_failure(
    run_record: dict[str, Any],
    metrics: dict[str, Any],
    *,
    thresholds: MetricThresholds = MetricThresholds(),
    require_output: bool = False,
) -> str:
    """Return the primary stable failure label for one run."""
    status = str(run_record.get("status", "")).upper()
    failure_reason = str(run_record.get("failure_reason", ""))

    if require_output:
        optimized_path = str(run_record.get("optimized_coils_path", "")).strip()
        if optimized_path in {"", "not_available", "unknown", "nan"}:
            if status in {"FAILED", "COMPLETED", "DONE"}:
                return "missing_output"
        elif not Path(optimized_path).exists() and status in {"FAILED", "COMPLETED", "DONE"}:
            return "missing_output"

    if _text_has(failure_reason, "nan") or any(_is_nan_metric(metrics.get(key)) for key in ("final_objective", "final_squared_flux")):
        return "optimizer_nan"
    if _text_has(failure_reason, "timeout", "timed out") or status == "TIMEOUT":
        return "optimizer_timeout"
    if _text_has(failure_reason, "max_iter", "maximum iteration", "max iteration"):
        return "optimizer_max_iter"
    if status == "FAILED":
        return "runtime_error" if failure_reason else "unknown_failure"
    if status == "MISSING_OUTPUT":
        return "missing_output"

    derived = derive_geometry_metrics(metrics, thresholds)
    if as_bool(metrics.get("self_intersection_flag")) or derived["self_intersection_flag"]:
        return "self_intersection"

    max_curv = as_float(metrics.get("max_curvature"))
    if max_curv is not None and max_curv > thresholds.max_curvature:
        return "curvature_exceeded"

    if derived["torsion_spike_count"]:
        return "torsion_spike"

    max_torsion = as_float(metrics.get("max_torsion"))
    if max_torsion is not None and max_torsion > thresholds.max_torsion * 1.5:
        return "torsion_spike"

    cc = as_float(metrics.get("min_coil_coil_distance"))
    if cc is not None and cc < thresholds.min_coil_coil_distance:
        return "coil_coil_too_close"

    cs = as_float(metrics.get("min_coil_plasma_distance"))
    if cs is not None and cs < thresholds.min_coil_plasma_distance:
        return "coil_plasma_too_close"

    mean_bn = as_float(metrics.get("mean_abs_Bn"))
    max_bn = as_float(metrics.get("max_abs_Bn"))
    if mean_bn is not None and mean_bn > thresholds.poor_mean_abs_Bn:
        return "poor_Bn"
    if max_bn is not None and max_bn > thresholds.poor_max_abs_Bn:
        return "poor_Bn"

    return "success"


def classify_labels(run_record: dict[str, Any], metrics: dict[str, Any], *, require_output: bool = False) -> list[str]:
    primary = classify_primary_failure(run_record, metrics, require_output=require_output)
    labels = [] if primary == "success" else [primary]
    # Add secondary labels without changing primary taxonomy order.
    for name, threshold, direction in (
        ("poor_Bn", 0.005, "gt"),
        ("curvature_exceeded", 5.0, "gt"),
        ("coil_coil_too_close", 0.2, "lt"),
        ("coil_plasma_too_close", 0.2, "lt"),
    ):
        field = {
            "poor_Bn": "mean_abs_Bn",
            "curvature_exceeded": "max_curvature",
            "coil_coil_too_close": "min_coil_coil_distance",
            "coil_plasma_too_close": "min_coil_plasma_distance",
        }[name]
        try:
            value = float(metrics.get(field))
        except (TypeError, ValueError):
            continue
        if (direction == "gt" and value > threshold) or (direction == "lt" and value < threshold):
            if name not in labels:
                labels.append(name)
    return labels or ["success"]
