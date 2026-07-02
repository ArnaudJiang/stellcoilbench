"""Simple evaluation creation from metric records."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from data_twin.core.ids import make_id
from data_twin.core.models import EvaluationRecord
from data_twin.metrics.failure_classifier import classify_labels
from data_twin.storage.jsonl_store import JsonlStore


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_low(value: Any, target: float) -> float | None:
    number = _as_float(value)
    if number is None or number <= 0:
        return None
    return min(1.0, target / number)


def _score_high(value: Any, target: float) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return min(1.0, max(0.0, number / target))


def _mean(values: list[float | None]) -> float | None:
    good = [v for v in values if v is not None]
    return sum(good) / len(good) if good else None


def evaluate_campaign(campaign_root: Path | str, campaign_id: str) -> int:
    store = JsonlStore(campaign_root)
    metric_rows = store.read("metrics.jsonl")
    by_run: dict[str, dict[str, Any]] = defaultdict(dict)
    case_by_run = {}
    for row in metric_rows:
        if row.get("available"):
            by_run[row["run_id"]][row["metric_name"]] = row.get("metric_value")
        case_by_run[row["run_id"]] = row.get("case_id")
    runs = {row["run_id"]: row for row in store.read("runs.jsonl")}
    existing = {row.get("run_id") for row in store.read("evaluations.jsonl")}
    count = 0
    for run_id, metrics in by_run.items():
        if run_id in existing:
            continue
        physics = _mean([_score_low(metrics.get("mean_abs_Bn"), 0.005), _score_low(metrics.get("max_abs_Bn"), 0.05)])
        geometry = _mean([
            _score_low(metrics.get("max_curvature"), 5.0),
            _score_high(metrics.get("min_coil_coil_distance"), 0.2),
            _score_high(metrics.get("min_coil_plasma_distance"), 0.2),
        ])
        numerical = 1.0 if runs.get(run_id, {}).get("status") == "completed" else 0.0
        balanced = _mean([physics, geometry, numerical])
        labels = classify_labels(runs.get(run_id, {}), metrics)
        evaluation = EvaluationRecord(
            evaluation_id=make_id("evaluation", {"run_id": run_id, "metrics": metrics}),
            campaign_id=campaign_id,
            case_id=case_by_run.get(run_id, ""),
            run_id=run_id,
            physics_score=physics,
            geometry_score=geometry,
            numerical_score=numerical,
            balanced_score=balanced,
            constraint_status="pass" if labels == ["success"] else "fail",
            failure_labels=labels,
            summary="; ".join(labels),
        )
        store.append("evaluations.jsonl", evaluation.to_dict())
        count += 1
    return count
