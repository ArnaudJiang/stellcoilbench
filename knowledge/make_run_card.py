#!/usr/bin/env python3
"""
Generate a short text "run card" from a CI run summary for embedding and semantic search.

Turns summary.json into a 10–25 line human-readable card suitable for vector search.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from stellcoilbench.path_utils import get_surface_filename
except ImportError:
    _repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repo / "src"))
    from stellcoilbench.path_utils import get_surface_filename

_THRESHOLD_UNITS: dict[str, str] = {
    "length": "m",
    "cc": "m",
    "cs": "m",
    "curvature": "1/m",
    "msc": "1/m²",
    "force": "N/m",
    "torque": "N·m",
    "flux": "",
    "arclength_variation": "m²",
}


def _fmt(val: float) -> str:
    """Format a number to 2 significant figures."""
    return f"{val:.2g}"


def make_run_card(summary: dict) -> str:
    """Generate a run card string from a summary dict.

    Includes threshold values (with units) and objective terms from the
    case config so the LLM proposer can see which thresholds led to
    which outcomes.  Numbers are formatted to 2 significant figures for
    readability.

    Parameters
    ----------
    summary : dict
        A CI run summary dict (from ``summary.json``).

    Returns
    -------
    str
        Human-readable run card (10–25 lines).
    """
    lines: list[str] = []
    cid = summary.get("case_id", "?")
    success = summary.get("success", False)
    score = summary.get("total_score", float("inf"))
    iters = summary.get("iterations_used", 0)
    wall = summary.get("walltime_sec", 0)
    tags = summary.get("tags", [])
    parents = summary.get("parent_ids", [])

    status = "SUCCESS" if success else "FAILED"
    lines.append(f"Run {cid}: {status}")
    lines.append(f"Score: {score:.2e} | Iterations: {iters} | Walltime: {wall:.0f}s")
    if tags:
        lines.append(f"Tags: {', '.join(tags)}")
    if parents:
        lines.append(f"Parents: {', '.join(parents[:3])}{'...' if len(parents) > 3 else ''}")

    cfg = summary.get("case_config", {})
    surface = get_surface_filename(cfg) or "?"
    cp = cfg.get("coils_params", {})
    ncoils = cp.get("ncoils", "?") if isinstance(cp, dict) else "?"
    order = cp.get("order", "?") if isinstance(cp, dict) else "?"
    lines.append(f"Surface: {surface} | ncoils={ncoils} order={order}")

    obj = cfg.get("coil_objective_terms", {})
    if obj and isinstance(obj, dict):
        threshold_parts: list[str] = []
        for key in sorted(obj):
            val = obj[key]
            if key.endswith("_threshold") and isinstance(val, (int, float)):
                short = key.replace("_threshold", "")
                unit = _THRESHOLD_UNITS.get(short, "")
                unit_str = f" {unit}" if unit else ""
                threshold_parts.append(f"{short}={_fmt(val)}{unit_str}")
        if threshold_parts:
            lines.append(f"Thresholds (reactor-scale): {', '.join(threshold_parts)}")

        term_types: list[str] = []
        for key in sorted(obj):
            val = obj[key]
            if isinstance(val, str) and val:
                term_types.append(f"{key}={val}")
        if term_types:
            lines.append(f"Objective terms: {', '.join(term_types)}")

    metrics = summary.get("metrics", {})
    if metrics:
        if "final_min_cc_separation" in metrics:
            lines.append(f"CC separation: {_fmt(metrics['final_min_cc_separation'])} m")
        if "final_min_cs_separation" in metrics:
            lines.append(f"CS separation: {_fmt(metrics['final_min_cs_separation'])} m")
        if "final_max_curvature" in metrics:
            lines.append(f"Max curvature: {_fmt(metrics['final_max_curvature'])} 1/m")
        if "final_mean_squared_curvature" in metrics:
            lines.append(f"MSC: {_fmt(metrics['final_mean_squared_curvature'])} 1/m²")
        if "final_total_length" in metrics:
            lines.append(f"Total length: {_fmt(metrics['final_total_length'])} m")
        if "final_max_max_coil_force" in metrics:
            lines.append(f"Max force: {_fmt(metrics['final_max_max_coil_force'])} N/m")
        if "BdotN_over_B" in metrics:
            lines.append(f"B·n/B: {metrics['BdotN_over_B']:.2e}")

    margins = summary.get("margins", {})
    if margins:
        tight = [k for k, v in margins.items() if isinstance(v, (int, float)) and v < 0.1]
        if tight:
            lines.append(f"Tight margins: {', '.join(tight)}")
        violated = [
            f"{k}={_fmt(v)}"
            for k, v in margins.items()
            if isinstance(v, (int, float)) and v < 0
        ]
        if violated:
            lines.append(f"Violated: {', '.join(violated)}")

    if not success:
        fc = summary.get("failure_class", "")
        fr = summary.get("failure_reason", "")[:80]
        lines.append(f"Failure: {fc} — {fr}")

    return "\n".join(lines)


def main() -> int:
    """Read summary from stdin or file, print run card to stdout."""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        summary = json.loads(path.read_text())
    else:
        summary = json.load(sys.stdin)
    print(make_run_card(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
