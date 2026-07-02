from data_twin.core.models import CaseRecord, MetricRecord, RunRecord
from data_twin.report.campaign_report import write_campaign_reports
from data_twin.storage.jsonl_store import JsonlStore


def test_campaign_reports_are_written(tmp_path) -> None:
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    store = JsonlStore(campaign_root)
    store.append("cases.jsonl", CaseRecord(case_id="case_a", campaign_id="camp").to_dict())
    store.append("runs.jsonl", RunRecord(run_id="run_a", case_id="case_a", campaign_id="camp", status="completed").to_dict())
    store.append(
        "metrics.jsonl",
        MetricRecord(
            metric_id="metric_a",
            campaign_id="camp",
            case_id="case_a",
            run_id="run_a",
            metric_name="final_squared_flux",
            metric_value=0.1,
            metric_type="physics",
            available=True,
        ).to_dict(),
    )

    out = write_campaign_reports(campaign_root, tmp_path / "reports")

    assert (out / "campaign_summary.md").exists()
    assert (out / "metric_availability.md").exists()
    assert "case_a" in (out / "top_cases.md").read_text(encoding="utf-8")
