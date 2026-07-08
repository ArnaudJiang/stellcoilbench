#!/usr/bin/env python3
"""Screen industrial Stage-A Round1 results for eval000030."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = (
    ROOT
    / "experiments/wout_squid_eval_000030/raw/results/"
    / "industrial_round1_stageA_a0_res128_q256"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments/wout_squid_eval_000030/reports/"
    / "industrial_round1_stageA_screen"
)


FIELDS = [
    "tier",
    "rank_score",
    "run_id",
    "status",
    "success",
    "order",
    "family",
    "policy_label",
    "random_seed",
    "avg_BdotN_over_B",
    "final_min_cc_separation",
    "final_min_cs_separation",
    "final_max_curvature",
    "final_mean_squared_curvature",
    "final_arclength_variation",
    "final_max_torsion",
    "final_linking_number",
    "final_length_ratio",
    "topology_status",
    "link_guard_topology_change",
    "link_guard_restored",
    "coil_coil_link",
    "run_dir",
    "failure_reason",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _num(value: Any, default: float = math.inf) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in {0, 1}:
        return bool(value)
    return None


def _case_meta(run_dir: Path) -> dict[str, Any]:
    case = _load_json(run_dir / "case.yaml")
    metadata = case.get("experiment_metadata", {}) if isinstance(case, dict) else {}
    return {
        "family": metadata.get("family", ""),
        "policy_label": metadata.get("policy_label", ""),
        "random_seed": case.get("random_seed", "") if isinstance(case, dict) else "",
    }


def _topology(run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    guard = _load_json(run_dir / "link_guard_final.json")
    audit = _load_json(run_dir / "link_audit.json")
    guard_change = _bool_or_none(guard.get("has_topology_change"))
    guard_restored = _bool_or_none(guard.get("restored_last_safe"))
    coil_link = _bool_or_none(audit.get("has_coil_coil_link"))
    final_link = _num(record.get("final_linking_number"), math.inf)

    if guard_change is False and coil_link is False:
        status = "clean"
    elif guard_change is False and coil_link is None:
        status = "clean_guard_only"
    elif guard_change is True or coil_link is True:
        status = "linked_or_changed"
    elif not guard and not audit and final_link != math.inf:
        status = "clean_record_link_only" if abs(final_link) < 0.5 else "linked_or_changed"
    elif guard or audit:
        status = "unknown_partial"
    else:
        status = "unknown_missing"
    return {
        "topology_status": status,
        "link_guard_topology_change": guard_change,
        "link_guard_restored": guard_restored,
        "coil_coil_link": coil_link,
    }


def _tier(row: dict[str, Any]) -> str:
    if row["status"] != "completed":
        return "incomplete_or_failed"
    if row["topology_status"] not in {"clean", "clean_guard_only", "clean_record_link_only"}:
        return "reject_topology"

    avg = _num(row["avg_BdotN_over_B"])
    cc = _num(row["final_min_cc_separation"], -math.inf)
    cs = _num(row["final_min_cs_separation"], -math.inf)
    curv = _num(row["final_max_curvature"])
    msc = _num(row["final_mean_squared_curvature"])
    arc = _num(row["final_arclength_variation"])
    tors = _num(row["final_max_torsion"])
    ratio = _num(row["final_length_ratio"])
    link = _num(row["final_linking_number"])

    if (
        cc >= 0.25
        and cs >= 0.25
        and curv <= 5.0
        and msc <= 5.0
        and arc <= 0.5
        and tors <= 7.0
        and ratio <= 1.25
        and abs(link) < 0.5
        and avg <= 0.08
    ):
        return "tier0_strict"
    if (
        cc >= 0.23
        and cs >= 0.23
        and curv <= 5.5
        and tors <= 8.0
        and ratio <= 1.35
        and abs(link) < 0.5
        and avg <= 0.10
    ):
        return "tier1_watch"
    if (
        cc >= 0.20
        and cs >= 0.20
        and curv <= 6.0
        and tors <= 10.0
        and ratio <= 1.50
        and abs(link) < 0.5
        and avg <= 0.12
    ):
        return "tier2_repair_seed"
    return "reject_geometry_or_bn"


def _rank_score(row: dict[str, Any]) -> float:
    avg = _num(row["avg_BdotN_over_B"])
    cc_margin = max(0.0, 0.25 - _num(row["final_min_cc_separation"], -math.inf))
    cs_margin = max(0.0, 0.25 - _num(row["final_min_cs_separation"], -math.inf))
    curv_margin = max(0.0, _num(row["final_max_curvature"]) - 5.0)
    tors_margin = max(0.0, _num(row["final_max_torsion"]) - 7.0)
    ratio_margin = max(0.0, _num(row["final_length_ratio"]) - 1.25)
    link_penalty = 0.1 * abs(_num(row["final_linking_number"], 999))
    return (
        avg
        + 0.2 * cc_margin
        + 0.2 * cs_margin
        + 0.02 * curv_margin
        + 0.01 * tors_margin
        + 0.03 * ratio_margin
        + link_penalty
    )


def _row(run_dir: Path) -> dict[str, Any]:
    record = _load_json(run_dir / "record.json")
    meta = _case_meta(run_dir)
    topology = _topology(run_dir, record)
    row = {
        "run_id": record.get("run_id", run_dir.name),
        "status": record.get("status", "missing_record"),
        "success": bool(record.get("success", False)),
        "order": record.get("order", ""),
        "avg_BdotN_over_B": record.get("avg_BdotN_over_B", ""),
        "final_min_cc_separation": record.get("final_min_cc_separation", ""),
        "final_min_cs_separation": record.get("final_min_cs_separation", ""),
        "final_max_curvature": record.get("final_max_curvature", ""),
        "final_mean_squared_curvature": record.get("final_mean_squared_curvature", ""),
        "final_arclength_variation": record.get("final_arclength_variation", ""),
        "final_max_torsion": record.get("final_max_torsion", ""),
        "final_linking_number": record.get("final_linking_number", ""),
        "final_length_ratio": record.get("final_length_ratio", ""),
        "run_dir": str(run_dir),
        "failure_reason": record.get("failure_reason", ""),
        **meta,
        **topology,
    }
    row["tier"] = _tier(row)
    row["rank_score"] = _rank_score(row)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["tier"]] = counts.get(row["tier"], 0) + 1
    ranked = [r for r in rows if r["tier"].startswith("tier")]
    ranked.sort(key=lambda r: (r["tier"], _num(r["rank_score"])))
    lines = [
        "# Industrial Round1 Stage-A Screening",
        "",
        f"- Results: `{rows[0]['run_dir'].split('/runs/')[0] if rows else ''}`",
        f"- Total records: {len(rows)}",
        "",
        "## Tier Counts",
        "",
    ]
    for tier in sorted(counts):
        lines.append(f"- {tier}: {counts[tier]}")
    lines.extend(["", "## Top Candidates", ""])
    for row in ranked[:30]:
        lines.append(
            "- "
            f"{row['tier']} `{row['run_id']}` "
            f"avgBn={row['avg_BdotN_over_B']} "
            f"cc={row['final_min_cc_separation']} "
            f"cs={row['final_min_cs_separation']} "
            f"curv={row['final_max_curvature']} "
            f"tors={row['final_max_torsion']} "
            f"ratio={row['final_length_ratio']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    run_dirs = sorted((args.results_dir / "runs").glob("*"))
    rows = [_row(run_dir) for run_dir in run_dirs if run_dir.is_dir()]
    rows.sort(key=lambda r: (r["tier"], _num(r["rank_score"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "screen_all.csv", rows)
    for tier in ["tier0_strict", "tier1_watch", "tier2_repair_seed"]:
        _write_csv(args.output_dir / f"{tier}.csv", [r for r in rows if r["tier"] == tier])
    _write_report(args.output_dir / "screen_report.md", rows)
    print(f"Screened {len(rows)} runs from {args.results_dir}")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
