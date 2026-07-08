from __future__ import annotations

from pathlib import Path

import yaml

from data_twin.core.models import CaseRecord, MetricRecord, RunRecord
from data_twin.core.state import init_campaign
from data_twin.storage.jsonl_store import JsonlStore
from data_twin.storage.sqlite_index import campaign_status, compare_campaigns, rebuild_index


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


def _write_campaign(tmp_path: Path, campaign_id: str, metric_value: float) -> None:
    root = init_campaign(_campaign_config(tmp_path, campaign_id))
    store = JsonlStore(root)
    store.append("cases.jsonl", CaseRecord(case_id=f"{campaign_id}_case", campaign_id=campaign_id).to_dict())
    store.append(
        "runs.jsonl",
        RunRecord(
            run_id=f"{campaign_id}_run",
            case_id=f"{campaign_id}_case",
            campaign_id=campaign_id,
            status="completed",
        ).to_dict(),
    )
    store.append(
        "metrics.jsonl",
        MetricRecord(
            metric_id=f"{campaign_id}_metric",
            campaign_id=campaign_id,
            case_id=f"{campaign_id}_case",
            run_id=f"{campaign_id}_run",
            metric_name="avg_BdotN_over_B",
            metric_value=metric_value,
            available=True,
        ).to_dict(),
    )


def test_sqlite_index_rebuild_status_and_compare(tmp_path: Path) -> None:
    root = tmp_path / "data_twin"
    index_path = root / "data_twin_index.sqlite"
    _write_campaign(tmp_path, "camp_a", 0.2)
    _write_campaign(tmp_path, "camp_b", 0.1)

    counts = rebuild_index(root=root, index_path=index_path)
    status = campaign_status("camp_a", root=root, index_path=index_path)
    comparison = compare_campaigns(["camp_a", "camp_b"], root=root, index_path=index_path)

    assert counts["campaigns"] == 2
    assert status["counts"]["runs"] == 1
    assert status["runs_by_status"] == {"completed": 1}
    assert comparison["metrics"]["camp_b"]["avg_BdotN_over_B"]["min"] == 0.1
