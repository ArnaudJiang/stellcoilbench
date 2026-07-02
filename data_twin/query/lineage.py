"""Lineage utilities."""

from __future__ import annotations

from typing import Any


def lineage_for(cases: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    by_id = {case["case_id"]: case for case in cases}
    direct_parents = list(by_id.get(case_id, {}).get("parent_case_ids") or [])
    ancestors = []

    def visit(current: str) -> None:
        case = by_id.get(current)
        if not case:
            return
        for parent in case.get("parent_case_ids") or []:
            ancestors.append(parent)
            visit(parent)

    visit(case_id)
    children = [case["case_id"] for case in cases if case_id in (case.get("parent_case_ids") or [])]
    return {"case_id": case_id, "parents": direct_parents, "ancestors": ancestors, "children": children}
