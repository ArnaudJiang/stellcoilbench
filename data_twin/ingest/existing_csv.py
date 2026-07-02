"""Ingest existing round record CSVs into core JSONL records."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from data_twin.core.hashing import parameter_hash
from data_twin.core.ids import make_id
from data_twin.core.models import CaseRecord, EventRecord, MetricRecord, RunRecord, now_iso
from data_twin.metrics.registry import METRIC_TYPES
from data_twin.storage.jsonl_store import JsonlStore


FIELD_MAP = {
    "avg_Bn_over_B": "mean_abs_Bn",
    "avg_BdotN_over_B": "mean_abs_Bn",
    "max_Bn_over_B": "max_abs_Bn",
    "max_BdotN_over_B": "max_abs_Bn",
    "total_length": "total_coil_length",
    "max_length": "max_coil_length",
    "cc": "min_coil_coil_distance",
    "cs": "min_coil_plasma_distance",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _status(row: dict[str, str]) -> str:
    if str(row.get("status", "")).upper() in {"DONE", "COMPLETED", "SKIPPED"}:
        return "completed"
    if str(row.get("status", "")).upper() == "FAILED" or str(row.get("success", "")).lower() == "false":
        return "failed"
    return "unknown"


def ingest_csv(campaign_root: Path | str, campaign_id: str, input_csv: Path | str) -> dict[str, int]:
    root = Path(campaign_root)
    store = JsonlStore(root)
    rows = _read(Path(input_csv))
    counts = {"cases": 0, "runs": 0, "metrics": 0}
    existing_cases = {row.get("case_id") for row in store.read("cases.jsonl")}
    existing_runs = {row.get("run_id") for row in store.read("runs.jsonl")}
    for row in rows:
        case_id = row.get("parent_case_id") or row.get("case_id") or row.get("run_id")
        run_id = row.get("run_id") or make_id("run", row)
        parameters: dict[str, Any] = {
            "source_case_id": case_id,
            "ncoils": row.get("ncoils") or row.get("n_coils"),
            "order": row.get("order") or row.get("n_modes"),
            "queue": row.get("queue"),
            "seed": row.get("seed"),
        }
        if case_id not in existing_cases:
            case = CaseRecord(
                case_id=case_id,
                campaign_id=campaign_id,
                parent_case_ids=[],
                proposal_source="existing_csv",
                proposal_reason=f"ingested from {input_csv}",
                parameter_hash=parameter_hash(parameters),
                parameters=parameters,
                status="completed" if _status(row) == "completed" else "failed",
                tags=["ingested"],
            )
            store.append("cases.jsonl", case.to_dict())
            existing_cases.add(case_id)
            counts["cases"] += 1
        if run_id not in existing_runs:
            run = RunRecord(
                run_id=run_id,
                case_id=case_id,
                campaign_id=campaign_id,
                backend="stellcoilbench",
                status=_status(row),
                failure_reason=row.get("failure_reason", ""),
                runtime_seconds=float(row["walltime_sec"]) if row.get("walltime_sec") else None,
                notes="ingested_from_existing_csv",
            )
            store.append("runs.jsonl", run.to_dict())
            existing_runs.add(run_id)
            counts["runs"] += 1
        metric_values: dict[str, Any] = {}
        for source, target in FIELD_MAP.items():
            if row.get(source) not in (None, ""):
                metric_values[target] = row[source]
        if row.get("walltime_sec"):
            metric_values["runtime_seconds"] = row["walltime_sec"]
        metric_values["failure_reason"] = row.get("failure_reason") or ("success" if _status(row) == "completed" else "unknown_failure")
        for metric_name in set(METRIC_TYPES) | set(metric_values):
            available = metric_name in metric_values and metric_values[metric_name] not in ("", None)
            metric = MetricRecord(
                metric_id=make_id("metric", {"run_id": run_id, "name": metric_name}),
                campaign_id=campaign_id,
                case_id=case_id,
                run_id=run_id,
                metric_name=metric_name,
                metric_value=metric_values.get(metric_name),
                metric_type=METRIC_TYPES.get(metric_name, "diagnostic"),
                extraction_method="existing_csv",
                available=available,
            )
            store.append("metrics.jsonl", metric.to_dict())
            counts["metrics"] += 1
    store.append(
        "events.jsonl",
        EventRecord(
            event_id=make_id("event", {"input": str(input_csv), "rows": len(rows)}),
            timestamp=now_iso(),
            campaign_id=campaign_id,
            object_type="campaign",
            object_id=campaign_id,
            event_type="csv_ingested",
            message=f"Ingested {len(rows)} rows from {input_csv}",
        ).to_dict(),
    )
    return counts
