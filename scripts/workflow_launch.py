#!/usr/bin/env python3
"""Data Twin gated launcher for generic optimization policy runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_twin.core.hashing import parameter_hash
from data_twin.core.ids import make_id
from data_twin.core.models import ArtifactRecord, CaseRecord, EventRecord, RunRecord, now_iso
from data_twin.core.state import init_campaign
from data_twin.storage.jsonl_store import JsonlStore


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def _ensure_anchor(store: JsonlStore, campaign_id: str) -> None:
    if not any(row.get("case_id") == "campaign" for row in store.read("cases.jsonl")):
        parameters = {"campaign_anchor": campaign_id}
        constraints: dict[str, Any] = {}
        store.append(
            "cases.jsonl",
            CaseRecord(
                case_id="campaign",
                campaign_id=campaign_id,
                proposal_source="workflow_launch",
                proposal_reason="Anchor record for campaign-level artifacts.",
                parameter_hash=parameter_hash(parameters, constraints),
                parameters=parameters,
                constraints=constraints,
                tags=["campaign_anchor"],
                status="proposed",
            ).to_dict(),
        )
    if not any(row.get("run_id") == "campaign" for row in store.read("runs.jsonl")):
        store.append(
            "runs.jsonl",
            RunRecord(
                run_id="campaign",
                case_id="campaign",
                campaign_id=campaign_id,
                backend="metadata",
                workdir=str(REPO),
                status="completed",
                notes="Anchor run for campaign-level artifacts.",
            ).to_dict(),
        )


def _register_manifest(
    *,
    campaign_id: str,
    root: Path,
    manifest_path: Path,
    policy_path: Path,
    results_dir: Path,
    command: str,
    max_parallel_simsopt: int,
    surface_resolution: int,
) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    jobs = manifest["jobs"]
    store = JsonlStore(root)
    _ensure_anchor(store, campaign_id)

    existing_cases = {row.get("case_id") for row in store.read("cases.jsonl")}
    existing_runs = {row.get("run_id") for row in store.read("runs.jsonl")}
    constraints = dict(manifest.get("targets") or {})
    constraints.update(policy.get("common", {}).get("thresholds", {}) or {})
    input_refs = {
        "manifest": str(manifest_path),
        "policy": str(policy_path),
        "results_dir": str(results_dir),
    }
    added_cases = 0
    added_runs = 0
    for job in jobs:
        run_id = job["run_id"]
        parameters = {
            "backend": job.get("backend"),
            "surface": job.get("surface"),
            "ncoils": job.get("ncoils"),
            "order": job.get("order"),
            "random_seed": job.get("random_seed"),
            "length_variance_weight": job.get("length_variance_weight"),
            "queue": job.get("queue"),
            "policy_label": job.get("policy_label"),
            "surface_resolution": surface_resolution,
            "max_parallel_simsopt": max_parallel_simsopt,
        }
        if run_id not in existing_cases:
            store.append(
                "cases.jsonl",
                CaseRecord(
                    case_id=run_id,
                    campaign_id=campaign_id,
                    proposal_source="workflow_launch",
                    proposal_reason="Registered from runner dry-run manifest before launch.",
                    parameter_hash=parameter_hash(parameters, constraints),
                    parameters=parameters,
                    constraints=constraints,
                    input_refs=input_refs,
                    tags=["workflow_launch", str(job.get("backend", ""))],
                    status="proposed",
                ).to_dict(),
            )
            added_cases += 1
        if run_id not in existing_runs:
            store.append(
                "runs.jsonl",
                RunRecord(
                    run_id=run_id,
                    case_id=run_id,
                    campaign_id=campaign_id,
                    backend=str(job.get("backend") or ""),
                    command=command,
                    workdir=str(REPO),
                    status="pending",
                    config_snapshot_path=str(policy_path),
                    environment_snapshot={
                        "conda_env": "stellcoilbench_vmec",
                        "surface_resolution": surface_resolution,
                        "max_parallel_simsopt": max_parallel_simsopt,
                    },
                ).to_dict(),
            )
            added_runs += 1

    artifact_specs = [
        (policy_path, "policy", "Runner policy used by workflow launch."),
        (manifest_path, "manifest", "Dry-run manifest used by workflow launch."),
    ]
    existing_artifacts = {row.get("artifact_id") for row in store.read("artifacts.jsonl")}
    for path, artifact_type, description in artifact_specs:
        artifact_id = make_id(
            "artifact",
            {"campaign": campaign_id, "path": str(path), "type": artifact_type},
        )
        if artifact_id in existing_artifacts:
            continue
        store.append(
            "artifacts.jsonl",
            ArtifactRecord(
                artifact_id=artifact_id,
                campaign_id=campaign_id,
                case_id="campaign",
                run_id="campaign",
                artifact_type=artifact_type,
                path=str(path.resolve()) if path.exists() else str(path),
                relative_path=str(path),
                description=description,
                metadata={"planned_cases": len(jobs)},
            ).to_dict(),
        )

    _append_event(
        store,
        campaign_id,
        "board_registered",
        f"Registered {len(jobs)} planned runs from runner dry-run manifest.",
        {
            "added_cases": added_cases,
            "added_runs": added_runs,
            "manifest": str(manifest_path),
            "policy": str(policy_path),
            "results_dir": str(results_dir),
        },
    )
    return len(jobs)


def _write_lifecycle(root: Path, state: str, metadata: dict[str, Any]) -> None:
    campaign_id = root.name
    (root / "lifecycle.json").write_text(
        json.dumps(
            {
                "campaign": campaign_id,
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
    parser.add_argument("--campaign-config", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["simsopt", "focus", "both"], default="simsopt")
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--surface-resolution", type=int, default=64)
    parser.add_argument("--max-parallel-simsopt", type=int, default=1)
    parser.add_argument("--tmux-session")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Required to actually launch.")
    args = parser.parse_args()

    root = REPO / "experiments/data_twin" / args.campaign
    if args.campaign_config and not root.exists():
        init_campaign(args.campaign_config)
    if not root.exists():
        raise SystemExit(f"Campaign does not exist: {root}")

    runner = [
        "conda",
        "run",
        "-n",
        "stellcoilbench_vmec",
        "python",
        "scripts/run_simsopt_batch.py",
        "--policy",
        str(args.policy),
        "--backend",
        args.backend,
        "--results-dir",
        str(args.results_dir),
        "--surface-resolution",
        str(args.surface_resolution),
        "--max-parallel-simsopt",
        str(args.max_parallel_simsopt),
        "--dry-run",
    ]
    dry_run = subprocess.run(runner, cwd=REPO, check=False, text=True, capture_output=True)
    if dry_run.returncode != 0:
        raise SystemExit(dry_run.stderr or dry_run.stdout)
    manifest_path = args.results_dir / "batch_manifest.json"
    if not manifest_path.exists():
        manifest_path = args.results_dir / "round1_manifest.json"
    planned = len(json.loads(manifest_path.read_text(encoding="utf-8"))["jobs"])
    if planned != args.expected:
        raise SystemExit(f"Dry-run planned {planned} jobs, expected {args.expected}")

    launch_cmd = (
        "MPLCONFIGDIR=/tmp/stellcoilbench_mplconfig "
        "conda run -n stellcoilbench_vmec python scripts/run_simsopt_batch.py "
        f"--policy {args.policy} --backend {args.backend} --results-dir {args.results_dir} "
        f"--surface-resolution {args.surface_resolution} "
        f"--max-parallel-simsopt {args.max_parallel_simsopt} "
        f"--data-twin-campaign {args.campaign}"
    )
    planned = _register_manifest(
        campaign_id=args.campaign,
        root=root,
        manifest_path=manifest_path,
        policy_path=args.policy,
        results_dir=args.results_dir,
        command=launch_cmd,
        max_parallel_simsopt=args.max_parallel_simsopt,
        surface_resolution=args.surface_resolution,
    )
    store = JsonlStore(root)
    _append_event(
        store,
        args.campaign,
        "preflight_passed",
        "Workflow preflight passed.",
        {"planned": planned, "expected": args.expected, "results_dir": str(args.results_dir)},
    )
    _write_lifecycle(root, "registered", {"planned": planned, "results_dir": str(args.results_dir)})
    if args.prepare_only:
        print(yaml.safe_dump({"ok": True, "state": "registered", "planned": planned}, sort_keys=False))
        return 0
    if not args.yes:
        print(yaml.safe_dump({"ok": True, "would_run": launch_cmd, "note": "pass --yes to launch"}, sort_keys=False))
        return 0

    _append_event(store, args.campaign, "launch_started", "Workflow launch started.", {"command": launch_cmd})
    _write_lifecycle(root, "running", {"command": launch_cmd})
    if args.tmux_session:
        tmux_cmd = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            args.tmux_session,
            f"cd {REPO} && {launch_cmd}",
        ]
        result = subprocess.run(tmux_cmd, cwd=REPO, check=False)
    else:
        result = subprocess.run(launch_cmd, cwd=REPO, shell=True, check=False)
        _append_event(
            store,
            args.campaign,
            "launch_finished",
            "Workflow launch finished.",
            {"command": launch_cmd, "returncode": result.returncode},
        )
        _write_lifecycle(root, "launch_finished", {"returncode": result.returncode})
    print(yaml.safe_dump({"ok": result.returncode == 0, "returncode": result.returncode}, sort_keys=False))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
