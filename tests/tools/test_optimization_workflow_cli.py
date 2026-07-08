from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from data_twin.core.state import init_campaign


REPO = Path(__file__).resolve().parents[2]


def _campaign_config(tmp_path: Path, campaign_id: str) -> Path:
    path = tmp_path / f"{campaign_id}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "campaign_id": campaign_id,
                "name": campaign_id,
                "storage": {"root": str(tmp_path / "data_twin")},
            }
        ),
        encoding="utf-8",
    )
    return path


def _run_workflow(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, "scripts/optimization_workflow.py", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_review_and_decision_commands_write_collaboration_records(tmp_path: Path) -> None:
    root = tmp_path / "data_twin"
    index_path = root / "data_twin_index.sqlite"
    init_campaign(_campaign_config(tmp_path, "camp_review"))

    review = _run_workflow(
        "review",
        "--campaign",
        "camp_review",
        "--data-twin-root",
        str(root),
        "--index-path",
        str(index_path),
        "--review-status",
        "approved",
        "--by",
        "bob",
        "--note",
        "preflight checked",
    )
    decision = _run_workflow(
        "decide",
        "--campaign",
        "camp_review",
        "--data-twin-root",
        str(root),
        "--index-path",
        str(index_path),
        "--decision",
        "refine",
        "--reason",
        "cs gap limits feasibility",
        "--next-action",
        "increase cs weight",
    )
    status = _run_workflow(
        "status",
        "--campaign",
        "camp_review",
        "--data-twin-root",
        str(root),
        "--index-path",
        str(index_path),
    )

    assert review["review"]["status"] == "approved"
    assert decision["decision"]["decision"] == "refine"
    assert status["counts"]["reviews"] == 1
    assert status["counts"]["decisions"] == 1
