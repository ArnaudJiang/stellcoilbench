#!/usr/bin/env python3
"""Unified entry point for StellCoilBench optimization workflows.

This script is the user-facing workflow interface. It routes generic
policy/manifest experiments through the Data Twin gated launcher, and routes
legacy eval000030 board workflows through their adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_twin.core.ids import make_id
from data_twin.core.models import DecisionRecord, EventRecord, ReviewRecord, now_iso
from data_twin.storage.index import DEFAULT_ROOT
from data_twin.storage.jsonl_store import JsonlStore
from data_twin.storage.sqlite_index import (
    DEFAULT_INDEX_PATH,
    campaign_status,
    compare_campaigns,
    rebuild_index,
)

GENERIC_LAUNCH = REPO / "scripts/workflow_launch.py"
GENERIC_RECONCILE = REPO / "scripts/workflow_reconcile.py"
GENERIC_INGEST = REPO / "scripts/workflow_ingest_results.py"
GENERIC_SCREEN = REPO / "scripts/workflow_screen_results.py"
EVAL000030_ADAPTER = REPO / "experiments/wout_squid_eval_000030/workflow/experiment.py"

GENERIC_ACTIONS = {"prepare", "launch", "reconcile", "ingest", "screen"}
DATA_TWIN_ACTIONS = {"index", "status", "review", "decide", "compare"}
BOARD_ACTIONS = {
    "plan",
    "generate",
    "prepare",
    "register",
    "preflight",
    "status",
    "monitor",
    "sync",
    "ingest",
    "screen",
    "report",
    "close",
    "launch",
}


def _print_json(data: dict[str, Any]) -> int:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))
    return 0


def _campaign_root(campaign_id: str, root: Path = DEFAULT_ROOT) -> Path:
    return root / campaign_id


def _run(command: list[str], *, print_command: bool) -> int:
    if print_command:
        print(" ".join(command))
        return 0
    env = os.environ.copy()
    env["STELLCOILBENCH_WORKFLOW_ENTRYPOINT"] = "scripts/optimization_workflow.py"
    return subprocess.run(command, cwd=REPO, env=env, check=False).returncode


def _board_command(args: argparse.Namespace, unknown: list[str]) -> list[str]:
    if args.workflow not in {"auto", "eval000030"}:
        raise SystemExit(f"Unsupported board workflow: {args.workflow}")
    if not EVAL000030_ADAPTER.exists():
        raise SystemExit(f"Board workflow adapter does not exist: {EVAL000030_ADAPTER}")
    if args.action not in BOARD_ACTIONS:
        raise SystemExit(f"Action {args.action!r} is not supported for board workflows")
    command = [
        sys.executable,
        str(EVAL000030_ADAPTER.relative_to(REPO)),
        args.action,
        "--board",
        str(args.board),
    ]
    if args.yes:
        command.append("--yes")
    return command + unknown


def _generic_launch_command(args: argparse.Namespace, unknown: list[str]) -> list[str]:
    missing = [
        name
        for name in ("campaign", "policy", "results_dir", "expected")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(
            "Generic prepare/launch requires: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    command = [
        sys.executable,
        str(GENERIC_LAUNCH.relative_to(REPO)),
        "--campaign",
        args.campaign,
        "--policy",
        str(args.policy),
        "--results-dir",
        str(args.results_dir),
        "--backend",
        args.backend,
        "--expected",
        str(args.expected),
        "--surface-resolution",
        str(args.surface_resolution),
        "--max-parallel-simsopt",
        str(args.max_parallel_simsopt),
    ]
    if args.campaign_config:
        command.extend(["--campaign-config", str(args.campaign_config)])
    if args.action == "prepare":
        command.append("--prepare-only")
    if args.action == "launch":
        if args.tmux_session:
            command.extend(["--tmux-session", args.tmux_session])
        if args.yes:
            command.append("--yes")
    return command + unknown


def _generic_reconcile_command(args: argparse.Namespace, unknown: list[str]) -> list[str]:
    missing = [
        name
        for name in ("campaign", "results_dir", "reason")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(
            "Generic reconcile requires: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    command = [
        sys.executable,
        str(GENERIC_RECONCILE.relative_to(REPO)),
        "--campaign",
        args.campaign,
        "--results-dir",
        str(args.results_dir),
        "--reason",
        args.reason,
        "--state",
        args.state,
    ]
    return command + unknown


def _generic_ingest_command(args: argparse.Namespace, unknown: list[str]) -> list[str]:
    missing = [
        name
        for name in ("campaign", "results_dir")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(
            "Generic ingest requires: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    command = [
        sys.executable,
        str(GENERIC_INGEST.relative_to(REPO)),
        "--campaign",
        args.campaign,
        "--results-dir",
        str(args.results_dir),
    ]
    if args.expected is not None:
        command.extend(["--expected", str(args.expected)])
    return command + unknown


def _generic_screen_command(args: argparse.Namespace, unknown: list[str]) -> list[str]:
    missing = [
        name
        for name in ("campaign", "results_dir", "report")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(
            "Generic screen requires: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    command = [
        sys.executable,
        str(GENERIC_SCREEN.relative_to(REPO)),
        "--campaign",
        args.campaign,
        "--results-dir",
        str(args.results_dir),
        "--report",
        str(args.report),
    ]
    for artifact in args.artifact or []:
        command.extend(["--artifact", str(artifact)])
    return command + unknown


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


def _require_campaign(args: argparse.Namespace) -> str:
    if not args.campaign:
        raise SystemExit("--campaign is required")
    campaign = args.campaign[0] if isinstance(args.campaign, list) else args.campaign
    if not isinstance(campaign, str):
        raise SystemExit("--campaign is required")
    return campaign


def _run_data_twin_action(args: argparse.Namespace) -> int:
    root = args.data_twin_root
    index_path = args.index_path
    if args.action == "index":
        if args.subaction != "rebuild":
            raise SystemExit("index requires subaction: rebuild")
        return _print_json(
            {
                "ok": True,
                "index": str(index_path),
                "counts": rebuild_index(root=root, index_path=index_path),
            }
        )
    if args.action == "status":
        campaign_id = _require_campaign(args)
        return _print_json(campaign_status(campaign_id, root=root, index_path=index_path))
    if args.action == "compare":
        campaign_ids = args.campaign or []
        if len(campaign_ids) < 2:
            raise SystemExit("compare requires at least two --campaign values")
        return _print_json(compare_campaigns(campaign_ids, root=root, index_path=index_path))
    if args.action == "review":
        campaign_id = _require_campaign(args)
        if not args.note:
            raise SystemExit("review requires --note")
        campaign_root = _campaign_root(campaign_id, root)
        if not campaign_root.exists():
            raise SystemExit(f"Campaign does not exist: {campaign_root}")
        store = JsonlStore(campaign_root)
        review = ReviewRecord(
            review_id=make_id(
                "review",
                {"campaign": campaign_id, "reviewer": args.by or "", "time": now_iso()},
            ),
            campaign_id=campaign_id,
            reviewer=args.by or "",
            status=args.review_status,
            note=args.note,
            metadata={"source": "optimization_workflow"},
        )
        store.append("reviews.jsonl", review.to_dict())
        _append_event(
            store,
            campaign_id,
            "campaign_reviewed",
            "Campaign review recorded.",
            {"review_id": review.review_id, "status": review.status},
        )
        rebuild_index(root=root, index_path=index_path)
        return _print_json({"ok": True, "review": review.to_dict()})
    if args.action == "decide":
        campaign_id = _require_campaign(args)
        if not args.decision:
            raise SystemExit("decide requires --decision")
        if not args.reason:
            raise SystemExit("decide requires --reason")
        campaign_root = _campaign_root(campaign_id, root)
        if not campaign_root.exists():
            raise SystemExit(f"Campaign does not exist: {campaign_root}")
        selected = args.selected_run or []
        rejected = args.rejected_run or []
        store = JsonlStore(campaign_root)
        decision = DecisionRecord(
            decision_id=make_id(
                "decision",
                {"campaign": campaign_id, "decision": args.decision, "time": now_iso()},
            ),
            campaign_id=campaign_id,
            case_id=args.case_id or "campaign",
            run_id=args.run_id or "",
            decision=args.decision,
            decision_type=args.decision,
            reason=args.reason,
            selected_runs=selected,
            rejected_runs=rejected,
            next_action=args.next_action or "",
            next_policy_hint=args.next_policy_hint or "",
            decided_by=args.by or "",
            notes=args.note or "",
        )
        store.append("decisions.jsonl", decision.to_dict())
        _append_event(
            store,
            campaign_id,
            "campaign_decision_recorded",
            "Campaign decision recorded.",
            {"decision_id": decision.decision_id, "decision": decision.decision},
        )
        rebuild_index(root=root, index_path=index_path)
        return _print_json({"ok": True, "decision": decision.to_dict()})
    raise SystemExit(f"Unknown Data Twin action: {args.action}")


def build_command(args: argparse.Namespace, unknown: list[str]) -> list[str]:
    if args.board is not None:
        return _board_command(args, unknown)
    if args.action not in GENERIC_ACTIONS:
        raise SystemExit(
            f"Action {args.action!r} needs --board, or use one of: {', '.join(sorted(GENERIC_ACTIONS))}"
        )
    if args.action == "reconcile":
        return _generic_reconcile_command(args, unknown)
    if args.action == "ingest":
        return _generic_ingest_command(args, unknown)
    if args.action == "screen":
        return _generic_screen_command(args, unknown)
    return _generic_launch_command(args, unknown)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=sorted(BOARD_ACTIONS | GENERIC_ACTIONS | DATA_TWIN_ACTIONS),
        help="Workflow action. Board actions require --board.",
    )
    parser.add_argument("subaction", nargs="?", help="Subaction, e.g. `index rebuild`.")
    parser.add_argument(
        "--workflow",
        choices=["auto", "eval000030"],
        default="auto",
        help="Board workflow adapter. Currently eval000030 is the only board adapter.",
    )
    parser.add_argument("--board", type=Path, help="Board YAML for board-driven workflows.")
    parser.add_argument("--campaign", action="append")
    parser.add_argument("--campaign-config", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--artifact", type=Path, action="append")
    parser.add_argument("--backend", choices=["simsopt", "focus", "both"], default="simsopt")
    parser.add_argument("--expected", type=int)
    parser.add_argument("--surface-resolution", type=int, default=64)
    parser.add_argument("--max-parallel-simsopt", type=int, default=1)
    parser.add_argument("--tmux-session")
    parser.add_argument("--reason", help="Reconcile reason.")
    parser.add_argument("--state", default="registered_late_results_ready")
    parser.add_argument("--data-twin-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--note")
    parser.add_argument("--by", help="Reviewer or decision author.")
    parser.add_argument(
        "--review-status",
        choices=["comment", "approved", "needs_changes", "blocked"],
        default="comment",
    )
    parser.add_argument(
        "--decision",
        choices=["continue", "stop", "refine", "promote", "abandon", "manual_review"],
    )
    parser.add_argument("--case-id")
    parser.add_argument("--run-id")
    parser.add_argument("--selected-run", action="append")
    parser.add_argument("--rejected-run", action="append")
    parser.add_argument("--next-action")
    parser.add_argument("--next-policy-hint")
    parser.add_argument("--yes", action="store_true", help="Required by launch actions.")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the routed command without executing it.",
    )
    args, unknown = parser.parse_known_args()
    if args.action in DATA_TWIN_ACTIONS:
        return _run_data_twin_action(args)
    if isinstance(args.campaign, list):
        if len(args.campaign) > 1:
            raise SystemExit(f"{args.action} accepts one --campaign")
        args.campaign = args.campaign[0]
    command = build_command(args, unknown)
    return _run(command, print_command=args.print_command)


if __name__ == "__main__":
    raise SystemExit(main())
