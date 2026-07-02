from data_twin.core.hashing import parameter_hash
from data_twin.core.models import CaseRecord, RunRecord


def test_core_records_round_trip() -> None:
    case = CaseRecord(
        case_id="case_a",
        campaign_id="camp",
        parameter_hash=parameter_hash({"n_coils": 4}),
        parameters={"n_coils": 4},
        tags=["n4"],
    )

    restored = CaseRecord.from_dict(case.to_dict())

    assert restored.case_id == "case_a"
    assert restored.parameters == {"n_coils": 4}
    assert restored.tags == ["n4"]


def test_run_record_preserves_failed_attempt() -> None:
    run = RunRecord(run_id="run_a", case_id="case_a", campaign_id="camp", status="failed", failure_reason="runtime_error")

    assert run.to_dict()["status"] == "failed"
    assert run.to_dict()["failure_reason"] == "runtime_error"
