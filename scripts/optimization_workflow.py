#!/usr/bin/env python3
"""Unified entry point for StellCoilBench optimization workflows.

This script is the user-facing workflow interface. It routes generic
policy/manifest experiments through the Data Twin gated launcher, and routes
legacy eval000030 board workflows through their adapter.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
GENERIC_LAUNCH = REPO / "scripts/workflow_launch.py"
GENERIC_RECONCILE = REPO / "scripts/workflow_reconcile.py"
EVAL000030_ADAPTER = REPO / "experiments/wout_squid_eval_000030/workflow/experiment.py"

GENERIC_ACTIONS = {"prepare", "launch", "reconcile"}
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


def _run(command: list[str], *, print_command: bool) -> int:
    if print_command:
        print(" ".join(command))
        return 0
    return subprocess.run(command, cwd=REPO, check=False).returncode


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


def build_command(args: argparse.Namespace, unknown: list[str]) -> list[str]:
    if args.board is not None:
        return _board_command(args, unknown)
    if args.action not in GENERIC_ACTIONS:
        raise SystemExit(
            f"Action {args.action!r} needs --board, or use one of: {', '.join(sorted(GENERIC_ACTIONS))}"
        )
    if args.action == "reconcile":
        return _generic_reconcile_command(args, unknown)
    return _generic_launch_command(args, unknown)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=sorted(BOARD_ACTIONS | GENERIC_ACTIONS),
        help="Workflow action. Board actions require --board.",
    )
    parser.add_argument(
        "--workflow",
        choices=["auto", "eval000030"],
        default="auto",
        help="Board workflow adapter. Currently eval000030 is the only board adapter.",
    )
    parser.add_argument("--board", type=Path, help="Board YAML for board-driven workflows.")
    parser.add_argument("--campaign")
    parser.add_argument("--campaign-config", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--backend", choices=["simsopt", "focus", "both"], default="simsopt")
    parser.add_argument("--expected", type=int)
    parser.add_argument("--surface-resolution", type=int, default=64)
    parser.add_argument("--max-parallel-simsopt", type=int, default=1)
    parser.add_argument("--tmux-session")
    parser.add_argument("--reason", help="Reconcile reason.")
    parser.add_argument("--state", default="registered_late_results_ready")
    parser.add_argument("--yes", action="store_true", help="Required by launch actions.")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the routed command without executing it.",
    )
    args, unknown = parser.parse_known_args()
    command = build_command(args, unknown)
    return _run(command, print_command=args.print_command)


if __name__ == "__main__":
    raise SystemExit(main())
