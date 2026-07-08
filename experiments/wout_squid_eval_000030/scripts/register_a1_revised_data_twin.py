#!/usr/bin/env python3
"""Register A1-revised planned cases/runs in the Data Twin campaign."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from data_twin.core.hashing import parameter_hash
from data_twin.core.ids import make_id
from data_twin.core.models import ArtifactRecord, CaseRecord, EventRecord, RunRecord, now_iso
from data_twin.core.state import init_campaign
from data_twin.storage.jsonl_store import JsonlStore


CAMPAIGN = "eval000030_a1_revised_dt"
CONFIG = REPO / "configs/data_twin/eval000030_a1_revised_campaign.yaml"
MANIFEST = (
    REPO
    / "experiments/wout_squid_eval_000030/policies/"
    / "squid_eval000030_industrial_round1_stageA_a1_revised_manifest.csv"
)
POLICY = (
    REPO
    / "experiments/wout_squid_eval_000030/policies/"
    / "squid_eval000030_industrial_round1_stageA_a1_revised_dt_policy.yaml"
)
RESULTS_DIR = (
    "experiments/wout_squid_eval_000030/raw/results/"
    "industrial_round1_stageA_a1_revised_dt_res64_q128"
)
COMMAND = (
    "MPLCONFIGDIR=/tmp/stellcoilbench_mplconfig "
    "conda run -n stellcoilbench_vmec python scripts/run_simsopt_batch.py "
    "--backend simsopt "
    "--policy experiments/wout_squid_eval_000030/policies/"
    "squid_eval000030_industrial_round1_stageA_a1_revised_dt_policy.yaml "
    "--surface-resolution 64 --max-parallel-simsopt 64 "
    "--data-twin-campaign eval000030_a1_revised_dt"
)


def _load_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_id(row: dict[str, str], idx: int) -> str:
    return (
        f"eval000030_industrial_stageA_a1_revised_{row['queue']}_simsopt_{idx:04d}_"
        f"lbfgsb_o{int(float(row['order']))}_it{int(float(row['max_iterations']))}"
    )


def _write_dt_policy() -> None:
    data = json.loads(
        (
            REPO
            / "experiments/wout_squid_eval_000030/policies/"
            / "squid_eval000030_industrial_round1_stageA_a1_revised_policy.yaml"
        ).read_text(encoding="utf-8")
    )
    data["results_dir"] = RESULTS_DIR
    data.setdefault("resources", {})["max_parallel_simsopt"] = 64
    POLICY.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _append_unique(store: JsonlStore, filename: str, key: str, record: dict[str, Any]) -> bool:
    existing = {row.get(key) for row in store.read(filename)}
    if record.get(key) in existing:
        return False
    store.append(filename, record)
    return True


def main() -> None:
    _write_dt_policy()
    root = init_campaign(CONFIG)
    store = JsonlStore(root)
    rows = _load_rows()
    added_cases = 0
    added_runs = 0
    for idx, row in enumerate(rows):
        run_id = _run_id(row, idx)
        case_id = run_id
        parameters = dict(row)
        parameters.update(
            {
                "surface": "plasma_surfaces/wout_squid_eval_000030.nc",
                "backend": "simsopt",
                "algorithm": "L-BFGS-B",
                "result_dir": RESULTS_DIR,
                "policy": str(POLICY.relative_to(REPO)),
            }
        )
        constraints = {
            "cc_threshold": row.get("cc_threshold"),
            "cs_threshold": row.get("cs_threshold"),
            "curvature_threshold": row.get("curvature_threshold"),
            "torsion_threshold": row.get("torsion_threshold"),
            "msc_threshold": row.get("msc_threshold"),
            "arclength_variation_threshold": row.get("arclength_variation_threshold"),
            "link_guard": True,
        }
        case = CaseRecord(
            case_id=case_id,
            campaign_id=CAMPAIGN,
            generation_index=1,
            parent_case_ids=[],
            proposal_source="industrial_round1_a1_revised_policy",
            proposal_reason="A1 revised scan after A0 showed cs/torsion/curvature failures.",
            parameter_hash=parameter_hash(parameters, constraints),
            parameters=parameters,
            constraints=constraints,
            input_refs={
                "policy": str(POLICY.relative_to(REPO)),
                "manifest": str(MANIFEST.relative_to(REPO)),
            },
            tags=["eval000030", "a1_revised", row["family"], row["queue"]],
            status="proposed",
        )
        if _append_unique(store, "cases.jsonl", "case_id", case.to_dict()):
            added_cases += 1
        run = RunRecord(
            run_id=run_id,
            case_id=case_id,
            campaign_id=CAMPAIGN,
            generation_index=1,
            backend="simsopt",
            command=COMMAND,
            workdir=str(REPO),
            status="pending",
            config_snapshot_path=str(POLICY.relative_to(REPO)),
            environment_snapshot={"conda_env": "stellcoilbench_vmec", "max_parallel_simsopt": 64},
            notes="planned_before_launch",
        )
        if _append_unique(store, "runs.jsonl", "run_id", run.to_dict()):
            added_runs += 1
    artifact = ArtifactRecord(
        artifact_id=make_id("artifact", {"campaign": CAMPAIGN, "policy": str(POLICY)}),
        campaign_id=CAMPAIGN,
        case_id="campaign",
        run_id="campaign",
        generation_index=1,
        artifact_type="policy",
        path=str(POLICY),
        relative_path=str(POLICY.relative_to(REPO)),
        description="A1-revised Data Twin launch policy.",
        metadata={"planned_cases": len(rows), "max_parallel_simsopt": 64},
    )
    _append_unique(store, "artifacts.jsonl", "artifact_id", artifact.to_dict())
    store.append(
        "events.jsonl",
        EventRecord(
            event_id=make_id("event", {"campaign": CAMPAIGN, "event": "a1_revised_registered", "rows": len(rows)}),
            timestamp=now_iso(),
            campaign_id=CAMPAIGN,
            object_type="campaign",
            object_id=CAMPAIGN,
            event_type="planned_cases_registered",
            message=f"Registered {len(rows)} A1-revised planned runs with max_parallel_simsopt=64.",
            metadata={"added_cases": added_cases, "added_runs": added_runs, "results_dir": RESULTS_DIR},
        ).to_dict(),
    )
    print(yaml.safe_dump({"campaign_root": str(root), "added_cases": added_cases, "added_runs": added_runs, "policy": str(POLICY), "results_dir": RESULTS_DIR}, sort_keys=False))


if __name__ == "__main__":
    main()
