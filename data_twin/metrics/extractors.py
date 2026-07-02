"""Metric extraction from indexed artifacts."""

from __future__ import annotations

from pathlib import Path

from data_twin.core.ids import make_id
from data_twin.core.models import MetricRecord
from data_twin.ingest.stellcoilbench_outputs import load_summary_metrics
from data_twin.metrics.registry import METRIC_TYPES
from data_twin.storage.jsonl_store import JsonlStore


SUMMARY_MAP = {
    "avg_BdotN_over_B": "mean_abs_Bn",
    "max_BdotN_over_B": "max_abs_Bn",
    "final_total_length": "total_coil_length",
    "final_max_curvature": "max_curvature",
    "final_max_torsion": "max_torsion",
    "final_min_cc_separation": "min_coil_coil_distance",
    "final_min_cs_separation": "min_coil_plasma_distance",
}


def extract_metrics(campaign_root: Path | str, campaign_id: str) -> int:
    root = Path(campaign_root)
    store = JsonlStore(root)
    existing = {(m.get("run_id"), m.get("metric_name"), m.get("source_artifact_id")) for m in store.read("metrics.jsonl")}
    count = 0
    for artifact in store.read("artifacts.jsonl"):
        if artifact.get("artifact_type") not in {"final_summary_json", "results_json"}:
            continue
        path = Path(artifact.get("path", ""))
        if not path.exists():
            continue
        raw = load_summary_metrics(path)
        for source, target in SUMMARY_MAP.items():
            if source not in raw:
                continue
            key = (artifact["run_id"], target, artifact["artifact_id"])
            if key in existing:
                continue
            metric = MetricRecord(
                metric_id=make_id("metric", key),
                campaign_id=campaign_id,
                case_id=artifact["case_id"],
                run_id=artifact["run_id"],
                metric_name=target,
                metric_value=raw[source],
                metric_type=METRIC_TYPES.get(target, "diagnostic"),
                source_artifact_id=artifact["artifact_id"],
                extraction_method="stellcoilbench_summary_json",
                available=True,
            )
            store.append("metrics.jsonl", metric.to_dict())
            count += 1
    return count
