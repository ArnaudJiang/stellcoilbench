"""CSV exports for Data Twin campaigns."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from data_twin.storage.jsonl_store import JsonlStore


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_campaign(root: Path | str) -> Path:
    root = Path(root)
    store = JsonlStore(root)
    out = root / "exports"
    _write(out / "cases.csv", store.read("cases.jsonl"))
    _write(out / "runs.csv", store.read("runs.jsonl"))
    _write(out / "metrics_long.csv", store.read("metrics.jsonl"))
    _write(out / "evaluations.csv", store.read("evaluations.jsonl"))
    _write(out / "decisions.csv", store.read("decisions.jsonl"))

    metrics = store.read("metrics.jsonl")
    wide: dict[tuple[str, str], dict[str, Any]] = {}
    for metric in metrics:
        key = (metric.get("case_id", ""), metric.get("run_id", ""))
        row = wide.setdefault(key, {"case_id": key[0], "run_id": key[1]})
        row[metric.get("metric_name", "")] = metric.get("metric_value") if metric.get("available") else "not_available"
    _write(out / "metrics_wide.csv", list(wide.values()))
    return out
