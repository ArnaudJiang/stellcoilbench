"""Metric helpers for JSONL-backed Data Twin records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

MISSING_SENTINELS = {"", "nan", "unknown", "not_available", None}


@dataclass(frozen=True)
class MetricThresholds:
    max_curvature: float = 5.0
    max_torsion: float = 10.0
    min_coil_coil_distance: float = 0.20
    min_coil_plasma_distance: float = 0.20
    poor_mean_abs_Bn: float = 0.005
    poor_max_abs_Bn: float = 0.05
    kink_curvature_ratio: float = 3.0
    kink_torsion_ratio: float = 3.0


def as_float(value: Any) -> float | None:
    if value in MISSING_SENTINELS:
        return None
    if isinstance(value, str) and value.strip().lower() in MISSING_SENTINELS:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_bool(value: Any) -> bool | None:
    if value in MISSING_SENTINELS:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def geometry_health_score(metrics: dict[str, Any], thresholds: MetricThresholds = MetricThresholds()) -> float | None:
    terms: list[float] = []
    max_curv = as_float(metrics.get("max_curvature"))
    max_torsion = as_float(metrics.get("max_torsion"))
    cc = as_float(metrics.get("min_coil_coil_distance"))
    cs = as_float(metrics.get("min_coil_plasma_distance"))
    if max_curv is not None and max_curv > 0:
        terms.append(min(1.0, thresholds.max_curvature / max_curv))
    if max_torsion is not None and max_torsion > 0:
        terms.append(min(1.0, thresholds.max_torsion / max_torsion))
    if cc is not None:
        terms.append(min(1.0, max(0.0, cc / thresholds.min_coil_coil_distance)))
    if cs is not None:
        terms.append(min(1.0, max(0.0, cs / thresholds.min_coil_plasma_distance)))
    return sum(terms) / len(terms) if terms else None


def derive_geometry_metrics(metrics: dict[str, Any], thresholds: MetricThresholds = MetricThresholds()) -> dict[str, Any]:
    max_curv = as_float(metrics.get("max_curvature"))
    p95_curv = as_float(metrics.get("p95_curvature"))
    mean_curv = as_float(metrics.get("mean_curvature"))
    max_torsion = as_float(metrics.get("max_torsion"))
    p95_torsion = as_float(metrics.get("p95_torsion"))
    mean_abs_torsion = as_float(metrics.get("mean_abs_torsion"))
    cc = as_float(metrics.get("min_coil_coil_distance"))
    curvature_spike_count = 0
    if max_curv is not None:
        if p95_curv and p95_curv > 0:
            curvature_spike_count = int(max_curv / p95_curv >= thresholds.kink_curvature_ratio)
        elif mean_curv and mean_curv > 0:
            curvature_spike_count = int(max_curv / mean_curv >= thresholds.kink_curvature_ratio)
        elif max_curv > thresholds.max_curvature * 1.5:
            curvature_spike_count = 1
    torsion_spike_count = 0
    if max_torsion is not None:
        if p95_torsion and p95_torsion > 0:
            torsion_spike_count = int(max_torsion / p95_torsion >= thresholds.kink_torsion_ratio)
        elif mean_abs_torsion and mean_abs_torsion > 0:
            torsion_spike_count = int(max_torsion / mean_abs_torsion >= thresholds.kink_torsion_ratio)
        elif max_torsion > thresholds.max_torsion * 1.5:
            torsion_spike_count = 1
    self_intersection = as_bool(metrics.get("self_intersection_flag"))
    if self_intersection is None:
        self_intersection = bool(cc is not None and cc <= 0)
    kink_flag = as_bool(metrics.get("kink_flag"))
    if kink_flag is None:
        kink_flag = bool(curvature_spike_count or torsion_spike_count)
    return {
        "self_intersection_flag": self_intersection,
        "kink_flag": kink_flag,
        "curvature_spike_count": curvature_spike_count,
        "torsion_spike_count": torsion_spike_count,
        "geometry_health_score": geometry_health_score(metrics, thresholds),
    }


def physics_score(metrics: dict[str, Any], thresholds: MetricThresholds = MetricThresholds()) -> float | None:
    terms = []
    mean_bn = as_float(metrics.get("mean_abs_Bn"))
    max_bn = as_float(metrics.get("max_abs_Bn"))
    if mean_bn is not None and mean_bn > 0:
        terms.append(min(1.0, thresholds.poor_mean_abs_Bn / mean_bn))
    if max_bn is not None and max_bn > 0:
        terms.append(min(1.0, thresholds.poor_max_abs_Bn / max_bn))
    return sum(terms) / len(terms) if terms else None


def balanced_score(metrics: dict[str, Any], thresholds: MetricThresholds = MetricThresholds()) -> float | None:
    geometry = geometry_health_score(metrics, thresholds)
    physics = physics_score(metrics, thresholds)
    values = [value for value in (geometry, physics) if value is not None]
    return sum(values) / len(values) if values else None
