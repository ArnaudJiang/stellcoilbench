"""Campaign validation for JSONL Data Twin records."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from data_twin.core.models import MODEL_BY_FILE
from data_twin.storage.jsonl_store import JsonlStore


VALID_CASE_STATUS = {"draft", "proposed", "queued", "running", "completed", "failed", "evaluated", "ranked", "selected", "archived"}
VALID_RUN_STATUS = {"pending", "running", "completed", "failed", "timeout", "cancelled", "missing_output", "unknown"}
VALID_REVIEW_STATUS = {"comment", "approved", "needs_changes", "blocked"}


def _missing_required(model: type, row: dict[str, Any]) -> list[str]:
    return [field for field in model.required_fields if row.get(field) in (None, "")]


def validate_campaign(root: Path | str, *, allow_duplicate_parameter_hash: bool = False) -> list[str]:
    root = Path(root)
    store = JsonlStore(root)
    errors: list[str] = []
    records = {filename: store.read(filename) for filename in MODEL_BY_FILE}
    for filename, model in MODEL_BY_FILE.items():
        for index, row in enumerate(records[filename], start=1):
            missing = _missing_required(model, row)
            if missing:
                errors.append(f"{filename}:{index} missing required fields: {missing}")

    cases = {row.get("case_id"): row for row in records["cases.jsonl"]}
    runs = {row.get("run_id"): row for row in records["runs.jsonl"]}
    artifacts = records["artifacts.jsonl"]
    metrics = records["metrics.jsonl"]
    evaluations = records["evaluations.jsonl"]
    reviews = records["reviews.jsonl"]

    for row in records["cases.jsonl"]:
        if row.get("status") not in VALID_CASE_STATUS:
            errors.append(f"case {row.get('case_id')} has invalid status {row.get('status')}")
        for parent in row.get("parent_case_ids") or []:
            if parent not in cases:
                errors.append(f"case {row.get('case_id')} parent_case_id does not exist: {parent}")

    for row in records["runs.jsonl"]:
        if row.get("case_id") not in cases:
            errors.append(f"run {row.get('run_id')} references missing case_id {row.get('case_id')}")
        if row.get("status") not in VALID_RUN_STATUS:
            errors.append(f"run {row.get('run_id')} has invalid status {row.get('status')}")

    for table_name, rows in (("artifact", artifacts), ("metric", metrics), ("evaluation", evaluations)):
        for row in rows:
            if row.get("case_id") not in cases:
                errors.append(f"{table_name} references missing case_id {row.get('case_id')}")
            if row.get("run_id") not in runs:
                errors.append(f"{table_name} references missing run_id {row.get('run_id')}")

    for row in reviews:
        if row.get("status") not in VALID_REVIEW_STATUS:
            errors.append(f"review {row.get('review_id')} has invalid status {row.get('status')}")

    hashes = [row.get("parameter_hash") for row in records["cases.jsonl"] if row.get("parameter_hash")]
    if not allow_duplicate_parameter_hash:
        for parameter_hash, count in Counter(hashes).items():
            if count > 1:
                errors.append(f"duplicate parameter_hash in campaign: {parameter_hash}")
    return errors
