#!/usr/bin/env python3
"""Repair Data Twin lifecycle state after a workflow ordering violation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_twin.core.ids import make_id
from data_twin.core.models import EventRecord, now_iso
from data_twin.storage.jsonl_store import JsonlStore


def _write_lifecycle(root: Path, state: str, metadata: dict[str, Any]) -> None:
    (root / "lifecycle.json").write_text(
        json.dumps(
            {
                "campaign": root.name,
                "state": state,
                "metadata": metadata,
                "updated_at": now_iso(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--state", default="registered_late_results_ready")
    args = parser.parse_args()

    root = REPO / "experiments/data_twin" / args.campaign
    if not root.exists():
        raise SystemExit(f"Campaign does not exist: {root}")
    records = list(args.results_dir.glob("runs/*/record.json"))
    metadata = {
        "reason": args.reason,
        "results_dir": str(args.results_dir),
        "record_json_count": len(records),
        "late_registration": True,
    }
    store = JsonlStore(root)
    store.append(
        "events.jsonl",
        EventRecord(
            event_id=make_id(
                "event",
                {"campaign": args.campaign, "event_type": "workflow_repaired", "time": now_iso()},
            ),
            timestamp=now_iso(),
            campaign_id=args.campaign,
            object_type="campaign",
            object_id=args.campaign,
            event_type="workflow_repaired",
            message="Workflow state reconciled after launch gate violation.",
            metadata=metadata,
        ).to_dict(),
    )
    _write_lifecycle(root, args.state, metadata)
    print(json.dumps({"ok": True, "state": args.state, **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
