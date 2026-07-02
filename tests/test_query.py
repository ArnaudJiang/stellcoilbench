from data_twin.core.models import CaseRecord, MetricRecord, RunRecord
from data_twin.query.api import DataTwin
from data_twin.storage.jsonl_store import JsonlStore


def test_query_api_filters_and_ranks_cases(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.append("cases.jsonl", CaseRecord(case_id="case_a", campaign_id="camp", tags=["n4"]).to_dict())
    store.append("runs.jsonl", RunRecord(run_id="run_a", case_id="case_a", campaign_id="camp", status="completed").to_dict())
    store.append(
        "metrics.jsonl",
        MetricRecord(
            metric_id="metric_a",
            campaign_id="camp",
            case_id="case_a",
            run_id="run_a",
            metric_name="final_squared_flux",
            metric_value=0.2,
            available=True,
        ).to_dict(),
    )

    dt = DataTwin.open(tmp_path)

    assert len(dt.cases(tags="n4")) == 1
    assert len(dt.runs(status="completed")) == 1
    assert dt.top_cases("final_squared_flux", ascending=True, n=1)[0]["case_id"] == "case_a"
