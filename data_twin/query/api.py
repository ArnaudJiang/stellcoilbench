"""Python query API for Data Twin campaigns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data_twin.query.filters import match
from data_twin.query.lineage import lineage_for
from data_twin.storage.jsonl_store import JsonlStore


class DataTwin:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.store = JsonlStore(self.root)

    @classmethod
    def open(cls, root: Path | str) -> "DataTwin":
        return cls(root)

    def cases(self, **filters: Any) -> list[dict[str, Any]]:
        return [row for row in self.store.read("cases.jsonl") if match(row, **filters)]

    def runs(self, **filters: Any) -> list[dict[str, Any]]:
        return [row for row in self.store.read("runs.jsonl") if match(row, **filters)]

    def artifacts(self, **filters: Any) -> list[dict[str, Any]]:
        return [row for row in self.store.read("artifacts.jsonl") if match(row, **filters)]

    def metrics(self, **filters: Any) -> list[dict[str, Any]]:
        return [row for row in self.store.read("metrics.jsonl") if match(row, **filters)]

    def evaluations(self, **filters: Any) -> list[dict[str, Any]]:
        return [row for row in self.store.read("evaluations.jsonl") if match(row, **filters)]

    def top_cases(self, metric: str, ascending: bool = True, n: int = 10) -> list[dict[str, Any]]:
        metrics = [row for row in self.metrics(metric_name=metric) if row.get("available")]
        metrics.sort(key=lambda row: float(row.get("metric_value")), reverse=not ascending)
        return metrics[:n]

    def lineage(self, case_id: str) -> dict[str, Any]:
        return lineage_for(self.cases(), case_id)
