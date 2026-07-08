from data_twin.core.hashing import parameter_hash
from data_twin.core.models import BriefRecord, CaseRecord, ReviewRecord, RunRecord


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


def test_collaboration_records_round_trip() -> None:
    brief = BriefRecord(
        brief_id="brief_a",
        campaign_id="camp",
        owner="alice",
        collaborators=["bob"],
        hypothesis="length variance helps",
    )
    review = ReviewRecord(
        review_id="review_a",
        campaign_id="camp",
        reviewer="bob",
        status="approved",
        note="preflight checked",
    )

    assert BriefRecord.from_dict(brief.to_dict()).collaborators == ["bob"]
    assert ReviewRecord.from_dict(review.to_dict()).status == "approved"
