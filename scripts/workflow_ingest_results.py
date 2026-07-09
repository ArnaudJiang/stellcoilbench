#!/usr/bin/env python3
"""Ingest generic workflow result records into Data Twin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_twin.core.ids import make_id
from data_twin.core.models import EventRecord, RunRecord, now_iso
from data_twin.evaluation.scoring import evaluate_campaign
from data_twin.metrics.extractors import extract_metrics
from data_twin.storage.artifact_store import attach_artifact
from data_twin.storage.jsonl_store import JsonlStore


def _write_lifecycle(root: Path, state: str, metadata: dict[str, Any]) -> None:
    (root / "lifecycle.json").write_text(
        json.dumps(
            {
                "campaign": root.name,
                "state": state,
                "metadata": metadata,
                "updated_at": now_iso(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _append_event(
    store: JsonlStore,
    campaign_id: str,
    event_type: str,
    message: str,
    metadata: dict[str, Any],
) -> None:
    store.append(
        "events.jsonl",
        EventRecord(
            event_id=make_id(
                "event",
                {"campaign": campaign_id, "event_type": event_type, "time": now_iso()},
            ),
            timestamp=now_iso(),
            campaign_id=campaign_id,
            object_type="campaign",
            object_id=campaign_id,
            event_type=event_type,
            message=message,
            metadata=metadata,
        ).to_dict(),
    )


def _read_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_exists(store: JsonlStore, run_id: str, path: Path, artifact_type: str) -> bool:
    artifact_id = make_id(
        "artifact",
        {"run_id": run_id, "path": str(path), "type": artifact_type},
    )
    return any(row.get("artifact_id") == artifact_id for row in store.read("artifacts.jsonl"))


def _upsert_unique(store: JsonlStore, filename: str, key: str, record: dict[str, Any]) -> bool:
    rows = store.read(filename)
    value = record.get(key)
    changed = False
    replaced = False
    new_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get(key) == value:
            replaced = True
            if row != record:
                changed = True
                new_rows.append(record)
            else:
                new_rows.append(row)
        else:
            new_rows.append(row)
    if not replaced:
        new_rows.append(record)
        changed = True
    if changed:
        store.write_all(filename, new_rows)
    return changed


def _append_status_run(
    *,
    store: JsonlStore,
    campaign_id: str,
    run_id: str,
    case_id: str,
    record: dict[str, Any],
    existing_run: dict[str, Any] | None,
) -> bool:
    status = str(record.get("status") or "unknown").lower()
    if status not in {"completed", "failed"}:
        status = "completed" if record.get("success") is not False else "failed"
    run = RunRecord(
        run_id=run_id,
        case_id=case_id,
        campaign_id=campaign_id,
        generation_index=int((existing_run or {}).get("generation_index") or 0),
        backend=str(record.get("backend") or (existing_run or {}).get("backend") or ""),
        backend_version=str((existing_run or {}).get("backend_version") or ""),
        command=str((existing_run or {}).get("command") or ""),
        workdir=str((existing_run or {}).get("workdir") or REPO),
        status=status,
        failure_reason=str(record.get("failure_reason") or ""),
        runtime_seconds=float(record["walltime_sec"]) if record.get("walltime_sec") else None,
        config_snapshot_path=str((existing_run or {}).get("config_snapshot_path") or ""),
        environment_snapshot=dict((existing_run or {}).get("environment_snapshot") or {}),
        git_commit=str((existing_run or {}).get("git_commit") or ""),
        notes="ingested_from_workflow_results",
    )
    return _upsert_unique(store, "runs.jsonl", "run_id", run.to_dict())


def ingest_results(campaign_id: str, results_dir: Path, expected: int | None = None) -> dict[str, int]:
    root = REPO / "experiments/data_twin" / campaign_id
    if not root.exists():
        raise SystemExit(f"Campaign does not exist: {root}")
    record_paths = sorted(results_dir.glob("runs/*/record.json"))
    if expected is not None and len(record_paths) != expected:
        raise SystemExit(f"Found {len(record_paths)} record.json files, expected {expected}")
    store = JsonlStore(root)
    existing_runs = {row.get("run_id"): row for row in store.read("runs.jsonl")}
    counts = {"records": len(record_paths), "runs": 0, "artifacts": 0, "metrics": 0, "evaluations": 0}

    for record_path in record_paths:
        record = _read_record(record_path)
        run_id = str(record.get("run_id") or record_path.parent.name)
        case_id = str((existing_runs.get(run_id) or {}).get("case_id") or run_id)
        changed_run = _append_status_run(
            store=store,
            campaign_id=campaign_id,
            run_id=run_id,
            case_id=case_id,
            record=record,
            existing_run=existing_runs.get(run_id),
        )
        counts["runs"] += int(changed_run)
        for artifact_type, path in (
            ("final_summary_json", record_path),
            ("results_json", record_path.parent / "results.json"),
            ("case_config", record_path.parent / "case.yaml"),
            ("coils_json", record_path.parent / "coils.json"),
        ):
            if not path.exists() or _artifact_exists(store, run_id, path, artifact_type):
                continue
            attach_artifact(
                root,
                campaign_id=campaign_id,
                case_id=case_id,
                run_id=run_id,
                artifact_path=path,
                artifact_type=artifact_type,
                description=f"Ingested {artifact_type} from workflow results.",
            )
            counts["artifacts"] += 1

    counts["metrics"] = extract_metrics(root, campaign_id)
    counts["evaluations"] = evaluate_campaign(root, campaign_id)
    _append_event(
        store,
        campaign_id,
        "results_ingested",
        f"Ingested {len(record_paths)} workflow records.",
        {"results_dir": str(results_dir), **counts},
    )
    _write_lifecycle(root, "ingested", {"results_dir": str(results_dir), **counts})
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int)
    args = parser.parse_args()
    print(json.dumps(ingest_results(args.campaign, args.results_dir, args.expected), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
