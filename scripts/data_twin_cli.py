#!/usr/bin/env python3
"""CLI for the Data Twin core."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))

from data_twin.core.hashing import parameter_hash
from data_twin.core.ids import make_id
from data_twin.core.models import CaseRecord, EventRecord, RunRecord, now_iso
from data_twin.core.state import campaign_root, init_campaign, load_config
from data_twin.core.validation import validate_campaign
from data_twin.evaluation.scoring import evaluate_campaign
from data_twin.ingest.existing_csv import ingest_csv
from data_twin.metrics.extractors import extract_metrics
from data_twin.query.api import DataTwin
from data_twin.report.campaign_report import write_campaign_reports
from data_twin.storage.artifact_store import attach_artifact
from data_twin.storage.csv_export import export_campaign
from data_twin.storage.jsonl_store import JsonlStore


def _campaign_id_from_root(root: Path) -> str:
    return root.name


def _root(args: argparse.Namespace) -> Path:
    return campaign_root(args.campaign)


def cmd_init(args: argparse.Namespace) -> int:
    root = init_campaign(args.config)
    print(f"Initialized campaign at {root}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    errors = validate_campaign(_root(args), allow_duplicate_parameter_hash=args.allow_duplicate_parameter_hash)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Campaign {args.campaign} is valid")
    return 0


def cmd_add_case(args: argparse.Namespace) -> int:
    root = _root(args)
    data = load_config(args.case_file)
    campaign_id = args.campaign
    parameters = data.get("parameters", data)
    constraints = data.get("constraints", {})
    case = CaseRecord(
        case_id=data.get("case_id") or make_id("case", parameters),
        campaign_id=campaign_id,
        generation_index=int(data.get("generation_index", 0)),
        parent_case_ids=data.get("parent_case_ids", []),
        proposal_source=data.get("proposal_source", "manual"),
        proposal_reason=data.get("proposal_reason", ""),
        parameter_hash=data.get("parameter_hash") or parameter_hash(parameters, constraints),
        parameters=parameters,
        constraints=constraints,
        input_refs=data.get("input_refs", {}),
        tags=data.get("tags", []),
        status=data.get("status", "proposed"),
        notes=data.get("notes", ""),
    )
    store = JsonlStore(root)
    store.append("cases.jsonl", case.to_dict())
    store.append("events.jsonl", EventRecord(make_id("event", case.to_dict()), now_iso(), campaign_id, "case", case.case_id, "case_added").to_dict())
    print(case.case_id)
    return 0


def cmd_add_run(args: argparse.Namespace) -> int:
    root = _root(args)
    data = load_config(args.run_file)
    campaign_id = args.campaign
    run = RunRecord(
        run_id=data.get("run_id") or make_id("run", data),
        case_id=data["case_id"],
        campaign_id=campaign_id,
        generation_index=int(data.get("generation_index", 0)),
        backend=data.get("backend", ""),
        backend_version=data.get("backend_version", ""),
        command=data.get("command", ""),
        workdir=data.get("workdir", ""),
        status=data.get("status", "pending"),
        failure_reason=data.get("failure_reason", ""),
        start_time=data.get("start_time", ""),
        end_time=data.get("end_time", ""),
        runtime_seconds=data.get("runtime_seconds"),
        stdout_path=data.get("stdout_path", ""),
        stderr_path=data.get("stderr_path", ""),
        config_snapshot_path=data.get("config_snapshot_path", ""),
        environment_snapshot=data.get("environment_snapshot", {}),
        git_commit=data.get("git_commit", ""),
        notes=data.get("notes", ""),
    )
    store = JsonlStore(root)
    store.append("runs.jsonl", run.to_dict())
    store.append("events.jsonl", EventRecord(make_id("event", run.to_dict()), now_iso(), campaign_id, "run", run.run_id, "run_added").to_dict())
    print(run.run_id)
    return 0


def cmd_attach_artifact(args: argparse.Namespace) -> int:
    root = _root(args)
    store = JsonlStore(root)
    runs = {run["run_id"]: run for run in store.read("runs.jsonl")}
    run = runs[args.run_id]
    artifact = attach_artifact(
        root,
        campaign_id=args.campaign,
        case_id=run["case_id"],
        run_id=args.run_id,
        artifact_path=args.path,
        artifact_type=args.type,
        copy=args.copy,
    )
    print(artifact.artifact_id)
    return 0


def cmd_ingest_csv(args: argparse.Namespace) -> int:
    counts = ingest_csv(_root(args), args.campaign, args.input)
    print(counts)
    return 0


def cmd_extract_metrics(args: argparse.Namespace) -> int:
    count = extract_metrics(_root(args), args.campaign)
    print(f"Extracted {count} metrics")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    count = evaluate_campaign(_root(args), args.campaign)
    print(f"Created {count} evaluations")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    out = export_campaign(_root(args))
    print(f"Exported CSVs to {out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    out = write_campaign_reports(_root(args))
    print(f"Wrote reports to {out}")
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    print(DataTwin.open(_root(args)).lineage(args.case_id))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("init")
    p.add_argument("--config", type=Path, required=True)
    p.set_defaults(func=cmd_init)
    p = sub.add_parser("validate")
    p.add_argument("--campaign", required=True)
    p.add_argument("--allow-duplicate-parameter-hash", action="store_true")
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser("add-case")
    p.add_argument("--campaign", required=True)
    p.add_argument("--case-file", type=Path, required=True)
    p.set_defaults(func=cmd_add_case)
    p = sub.add_parser("add-run")
    p.add_argument("--campaign", required=True)
    p.add_argument("--run-file", type=Path, required=True)
    p.set_defaults(func=cmd_add_run)
    p = sub.add_parser("attach-artifact")
    p.add_argument("--campaign", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--path", type=Path, required=True)
    p.add_argument("--type", required=True)
    p.add_argument("--copy", action="store_true")
    p.set_defaults(func=cmd_attach_artifact)
    p = sub.add_parser("ingest-csv")
    p.add_argument("--campaign", required=True)
    p.add_argument("--input", type=Path, required=True)
    p.set_defaults(func=cmd_ingest_csv)
    p = sub.add_parser("extract-metrics")
    p.add_argument("--campaign", required=True)
    p.set_defaults(func=cmd_extract_metrics)
    p = sub.add_parser("evaluate")
    p.add_argument("--campaign", required=True)
    p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("export")
    p.add_argument("--campaign", required=True)
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("report")
    p.add_argument("--campaign", required=True)
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("lineage")
    p.add_argument("--campaign", required=True)
    p.add_argument("--case-id", required=True)
    p.set_defaults(func=cmd_lineage)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
