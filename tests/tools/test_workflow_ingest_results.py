from __future__ import annotations

import json
from pathlib import Path

from data_twin.core.models import CaseRecord, RunRecord
from data_twin.storage.jsonl_store import JsonlStore
from scripts.workflow_ingest_results import ingest_results


def test_ingest_results_upserts_existing_run_status(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    campaign_id = "camp_ingest"
    root = repo / "experiments" / "data_twin" / campaign_id
    root.mkdir(parents=True)
    store = JsonlStore(root)
    store.append(
        "cases.jsonl",
        CaseRecord(case_id="run_a", campaign_id=campaign_id).to_dict(),
    )
    store.append(
        "runs.jsonl",
        RunRecord(
            run_id="run_a",
            case_id="run_a",
            campaign_id=campaign_id,
            backend="simsopt",
            status="pending",
            command="run command",
        ).to_dict(),
    )

    results_dir = tmp_path / "results"
    run_dir = results_dir / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    record = {
        "run_id": "run_a",
        "backend": "simsopt",
        "status": "completed",
        "success": True,
        "walltime_sec": 1.5,
        "avg_BdotN_over_B": 0.01,
    }
    (run_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")

    monkeypatch.setattr("scripts.workflow_ingest_results.REPO", repo)

    counts = ingest_results(campaign_id, results_dir, expected=1)
    runs = JsonlStore(root).read("runs.jsonl")

    assert counts["records"] == 1
    assert counts["runs"] == 1
    assert len([row for row in runs if row["run_id"] == "run_a"]) == 1
    assert runs[0]["status"] == "completed"
    assert runs[0]["command"] == "run command"
