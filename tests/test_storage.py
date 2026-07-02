from data_twin.core.models import CaseRecord
from data_twin.storage.jsonl_store import JsonlStore


def test_jsonl_store_is_append_only(tmp_path) -> None:
    store = JsonlStore(tmp_path)

    store.append("cases.jsonl", CaseRecord(case_id="case_1", campaign_id="camp").to_dict())
    store.append("cases.jsonl", CaseRecord(case_id="case_2", campaign_id="camp").to_dict())

    rows = store.read("cases.jsonl")
    assert [row["case_id"] for row in rows] == ["case_1", "case_2"]
