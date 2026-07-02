from data_twin.core.models import CaseRecord
from data_twin.query.lineage import lineage_for


def test_lineage_includes_parents_and_children() -> None:
    cases = [
        CaseRecord(case_id="parent", campaign_id="camp").to_dict(),
        CaseRecord(case_id="child", campaign_id="camp", parent_case_ids=["parent"]).to_dict(),
        CaseRecord(case_id="grandchild", campaign_id="camp", parent_case_ids=["child"]).to_dict(),
    ]

    lineage = lineage_for(cases, "child")

    assert lineage["case_id"] == "child"
    assert lineage["parents"] == ["parent"]
    assert lineage["children"] == ["grandchild"]
