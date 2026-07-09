#!/usr/bin/env python3
"""Mark a generic workflow campaign as screened and attach screen artifacts."""

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
from data_twin.storage.artifact_store import attach_artifact
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


def _append_event(
    store: JsonlStore,
    campaign_id: str,
    event_type: str,
    message: str,
    metadata: dict[str, Any],
) -> None:
    store.append(
        "events.jsonl",
        EventRecord(
            event_id=make_id(
                "event",
                {"campaign": campaign_id, "event_type": event_type, "time": now_iso()},
            ),
            timestamp=now_iso(),
            campaign_id=campaign_id,
            object_type="campaign",
            object_id=campaign_id,
            event_type=event_type,
            message=message,
            metadata=metadata,
        ).to_dict(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    args = parser.parse_args()

    root = REPO / "experiments/data_twin" / args.campaign
    if not root.exists():
        raise SystemExit(f"Campaign does not exist: {root}")
    if not args.report.exists():
        raise SystemExit(f"Report does not exist: {args.report}")

    store = JsonlStore(root)
    attached = []
    for artifact_type, path in [("screen_report", args.report), *[("screen_artifact", p) for p in args.artifact]]:
        if not path.exists():
            continue
        record = attach_artifact(
            root,
            campaign_id=args.campaign,
            case_id="campaign",
            run_id="campaign",
            artifact_path=path,
            artifact_type=artifact_type,
            description=f"Workflow {artifact_type}.",
        )
        attached.append(record.path)
    metadata = {
        "results_dir": str(args.results_dir),
        "report": str(args.report),
        "artifacts": attached,
    }
    _append_event(store, args.campaign, "screened", "Workflow screening completed.", metadata)
    _write_lifecycle(root, "screened", metadata)
    print(json.dumps({"ok": True, "state": "screened", **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
