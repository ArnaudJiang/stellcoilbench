#!/usr/bin/env python3
"""
Build a compact context payload from recent CI results.

Used by the proposer (and eventually an LLM) to decide what to run next.
Output is a JSON dict written to stdout or a file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


from stellcoilbench.path_utils import get_surface_filename, load_yaml, load_yaml_safe
from stellcoilbench.ci_autopilot import _compute_margins

logger = logging.getLogger(__name__)


def _load_policy(policy_path: Path) -> Dict[str, Any]:
    """Load proposer_policy.yaml."""
    return load_yaml(path=policy_path)


def _load_summaries(done_dir: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    """Load completed case summaries from cases/done/*/summary.json.

    Returns summaries sorted newest-first (by case_id which embeds a date).
    If *limit* is given, return at most that many.
    """
    summaries: List[Dict[str, Any]] = []
    if not done_dir.is_dir():
        return summaries

    for summary_file in sorted(done_dir.glob("*/summary.json"), reverse=True):
        try:
            data = json.loads(summary_file.read_text())
            summaries.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    if limit is not None:
        summaries = summaries[:limit]
    return summaries


def _load_summaries_from_submissions(
    submissions_root: Path,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Load successful run summaries from submissions/<surface>/auto/<case_id>/.

    Each submission has results.json (metrics, metadata) and case.yaml.
    Builds summary-like dicts for build_context compatibility.

    Parameters
    ----------
    submissions_root : Path
        Root submissions directory (e.g. submissions/).
    limit : int | None
        Max number of summaries to return (default None = no limit).

    Returns
    -------
    list[dict]
        Summary-like dicts sorted by case_id newest-first.
    """
    summaries: List[Dict[str, Any]] = []
    if not submissions_root.is_dir():
        return summaries

    for surface_dir in submissions_root.iterdir():
        if not surface_dir.is_dir():
            continue
        auto_dir = surface_dir / "auto"
        if not auto_dir.is_dir():
            continue
        for case_dir in auto_dir.iterdir():
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            results_path = case_dir / "results.json"
            case_yaml_path = case_dir / "case.yaml"
            if not results_path.exists() or not case_yaml_path.exists():
                continue
            try:
                results = json.loads(results_path.read_text())
                case_config = load_yaml(path=case_yaml_path) or {}
            except (json.JSONDecodeError, OSError):
                continue

            metrics = results.get("metrics", {})
            meta = results.get("metadata", {})
            total_score = float(metrics.get("final_squared_flux", float("inf")))
            iterations_used = int(meta.get("iterations_used", 0))
            walltime_sec = float(meta.get("walltime_sec", 0.0))

            cfg_hash = _config_hash(case_config)
            margins = _compute_margins(metrics)

            summary: Dict[str, Any] = {
                "case_id": case_id,
                "success": True,
                "total_score": total_score,
                "iterations_used": iterations_used,
                "walltime_sec": walltime_sec,
                "metrics": {
                    k: v for k, v in metrics.items() if isinstance(v, (int, float))
                },
                "case_config": case_config,
                "config_hash": cfg_hash,
                "margins": margins,
                "tags": [],
                "parent_ids": [],
            }
            summaries.append(summary)

    summaries.sort(key=lambda s: s.get("case_id", ""), reverse=True)
    if limit is not None:
        summaries = summaries[:limit]
    return summaries


def _load_failures_from_file(
    failures_path: Path,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Load failure summaries from policy/autopilot_failures.json.

    Parameters
    ----------
    failures_path : Path
        Path to autopilot_failures.json.
    limit : int | None
        Max number of failures to return (default None = no limit).

    Returns
    -------
    list[dict]
        Failure summary-like dicts (success=False).
    """
    if not failures_path.exists():
        return []
    try:
        data = json.loads(failures_path.read_text())
        failures = (
            data
            if isinstance(data, list)
            else (data.get("failures", []) if isinstance(data, dict) else [])
        )
        if not isinstance(failures, list):
            return []
        if limit is not None:
            failures = failures[-limit:]
        return list(reversed(failures))
    except (json.JSONDecodeError, OSError):
        return []


def _config_hash(cfg: Dict[str, Any]) -> str:
    """Deterministic hash of a case_config dict for novelty checking."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def compute_failure_stats(
    summaries: List[Dict[str, Any]],
    window: int = 30,
) -> Dict[str, Any]:
    """Compute failure statistics over a sliding window of recent summaries."""
    recent = summaries[:window]
    if not recent:
        return {
            "window_size": 0,
            "fail_count": 0,
            "fail_rate": 0.0,
            "failure_reasons": {},
            "failure_classes": {},
            "most_common_reason": None,
            "most_common_reason_count": 0,
        }

    fail_count = sum(1 for s in recent if not s.get("success", True))
    fail_rate = fail_count / len(recent) if recent else 0.0

    reasons: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    for s in recent:
        if not s.get("success", True):
            reason = s.get("failure_reason", "unknown")
            cls = s.get("failure_class", "unknown")
            if reason:
                reasons[reason] += 1
            if cls:
                classes[cls] += 1

    most_common = reasons.most_common(1)
    return {
        "window_size": len(recent),
        "fail_count": fail_count,
        "fail_rate": round(fail_rate, 4),
        "failure_reasons": dict(reasons.most_common(10)),
        "failure_classes": dict(classes.most_common(10)),
        "most_common_reason": most_common[0][0] if most_common else None,
        "most_common_reason_count": most_common[0][1] if most_common else 0,
    }


def get_top_parents(
    summaries: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Return the top-K feasible (success=True) parents sorted by total_score ascending."""
    feasible = [s for s in summaries if s.get("success", False)]
    # Lower total_score is better (it's squared flux)
    feasible.sort(key=lambda s: s.get("total_score", float("inf")))
    parents = []
    for s in feasible[:top_k]:
        parents.append(
            {
                "case_id": s.get("case_id", ""),
                "total_score": s.get("total_score"),
                "iterations_used": s.get("iterations_used"),
                "walltime_sec": s.get("walltime_sec"),
                "metrics": {
                    k: v
                    for k, v in s.get("metrics", {}).items()
                    if isinstance(v, (int, float))
                },
                "case_config": s.get("case_config", {}),
            }
        )
    return parents


def get_recent_config_hashes(
    summaries: List[Dict[str, Any]],
    last_n: int = 50,
) -> List[str]:
    """Return config hashes of the last *last_n* runs for novelty checking."""
    hashes = []
    for s in summaries[:last_n]:
        cfg = s.get("case_config", {})
        if cfg:
            hashes.append(_config_hash(cfg))
    return hashes


def _summary_to_parent(s: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a summary dict to top_parents format."""
    return {
        "case_id": s.get("case_id", ""),
        "total_score": s.get("total_score"),
        "iterations_used": s.get("iterations_used"),
        "walltime_sec": s.get("walltime_sec"),
        "metrics": {
            k: v for k, v in s.get("metrics", {}).items() if isinstance(v, (int, float))
        },
        "case_config": s.get("case_config", {}),
    }


def _generate_run_cards(
    summaries: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[str]:
    """Generate run card texts for the top feasible summaries.

    Parameters
    ----------
    summaries : list[dict]
        Summaries sorted newest-first.
    top_k : int
        Max number of run cards to generate.

    Returns
    -------
    list[str]
        Run card texts (one per feasible run, best first).
    """
    try:
        from knowledge.make_run_card import make_run_card
    except ImportError:
        return []

    feasible = [s for s in summaries if s.get("success", False)]
    feasible.sort(key=lambda s: s.get("total_score", float("inf")))
    cards: List[str] = []
    for s in feasible[:top_k]:
        try:
            card = make_run_card(s)
            if card:
                cards.append(card)
        except Exception:
            continue
    return cards


def _generate_postmortems(
    summaries: List[Dict[str, Any]],
    max_pm: int = 5,
) -> List[str]:
    """Generate postmortem texts for recent failures.

    Parameters
    ----------
    summaries : list[dict]
        Summaries sorted newest-first.
    max_pm : int
        Max number of postmortems to generate.

    Returns
    -------
    list[str]
        Postmortem texts (one per failed run).
    """
    try:
        from knowledge.make_postmortem import make_postmortem
    except ImportError:
        return []

    pms: List[str] = []
    for s in summaries:
        if not s.get("success", True):
            try:
                pm = make_postmortem(s)
                if pm:
                    pms.append(pm)
                    if len(pms) >= max_pm:
                        break
            except Exception:
                continue
    return pms


def _load_baseline_cases(
    cases_dir: Path,
    allowed_surfaces: List[str],
    *,
    max_per_surface: int = 3,
) -> List[str]:
    """Load baseline case configs from cases/*.yaml for LLM context.

    Returns formatted strings describing curated configs that work, filtered
    to only surfaces in the policy's allowed list. Skips dipole cases.
    Use as reference for reasonable thresholds when proposing explore actions
    or when there are few/no done runs for a surface.

    Parameters
    ----------
    cases_dir : Path
        Directory containing case YAML files (e.g. cases/).
    allowed_surfaces : list[str]
        Surfaces from policy.exploration.surfaces; only include cases for these.
    max_per_surface : int, optional
        Max baseline cases to include per surface (default 3).

    Returns
    -------
    list[str]
        Formatted baseline case descriptions, one per case.
    """
    if not cases_dir.is_dir() or not allowed_surfaces:
        return []

    allowed = set(allowed_surfaces)
    by_surface: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}

    for yaml_path in sorted(cases_dir.glob("*.yaml")):
        data = load_yaml_safe(path=yaml_path)
        if data is None:
            continue
        if not isinstance(data, dict):
            continue
        surface = get_surface_filename(data)
        if surface not in allowed:
            continue
        cp = data.get("coils_params", {}) or {}
        if cp.get("coil_type") == "dipole":
            continue

        name = yaml_path.stem
        by_surface.setdefault(surface, []).append((name, data))

    result: List[str] = []
    for surface in allowed_surfaces:
        cases = by_surface.get(surface, [])
        for name, data in cases[:max_per_surface]:
            lines = _format_baseline_case(name, data)
            if lines:
                result.append("\n".join(lines))
    return result


def _format_baseline_case(name: str, data: Dict[str, Any]) -> List[str]:
    """Format a single baseline case for the LLM prompt."""
    surface = get_surface_filename(data) or "?"
    cp = data.get("coils_params", {}) or {}
    ncoils = cp.get("ncoils", "?")
    order = cp.get("order", "?")
    opt = data.get("optimizer_params", {}) or {}
    algo = opt.get("algorithm", "?")
    obj = data.get("coil_objective_terms", {}) or {}

    parts = [
        f"Baseline case {name}: surface={surface}, ncoils={ncoils}, order={order}, algorithm={algo}"
    ]

    fc = data.get("fourier_continuation", {})
    if fc and fc.get("enabled") and fc.get("orders"):
        parts[0] += f", fourier_continuation={fc['orders']}"

    thresholds = []
    for k in (
        "length_threshold",
        "cc_threshold",
        "cs_threshold",
        "curvature_threshold",
        "msc_threshold",
        "force_threshold",
        "torque_threshold",
        "torsion_threshold",
    ):
        v = obj.get(k)
        if isinstance(v, (int, float)):
            thresholds.append(f"{k}={v}")
    if thresholds:
        parts.append("  thresholds (reactor scale): " + ", ".join(thresholds))

    return parts


def _compute_margin_summary(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate constraint margin statistics across successful runs.

    Returns a dict mapping constraint names to their violation rates and
    median margins, giving the LLM visibility into which constraints are
    hardest to satisfy.

    Parameters
    ----------
    summaries : list[dict]
        Summaries sorted newest-first.

    Returns
    -------
    dict
        ``{constraint: {violation_rate, median_margin, n}}`` for each
        constraint found in the margins dicts.
    """
    from statistics import median as _median

    margin_vals: Dict[str, List[float]] = {}
    for s in summaries:
        if not s.get("success"):
            continue
        for k, v in s.get("margins", {}).items():
            if isinstance(v, (int, float)):
                margin_vals.setdefault(k, []).append(v)

    result: Dict[str, Any] = {}
    for k, vals in margin_vals.items():
        violated = sum(1 for v in vals if v < 0)
        result[k] = {
            "violation_rate": round(violated / len(vals), 3) if vals else 0.0,
            "median_margin": round(_median(vals), 6) if vals else 0.0,
            "n": len(vals),
        }
    return result


def _compute_score_trend(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute score trend and Pareto analysis across recent runs.

    Splits summaries into an older half and a newer half and compares
    median scores to indicate whether the system is improving, plateaued,
    or regressing.  Also reports the overall best score and the best
    score among runs that satisfy all constraints.

    Parameters
    ----------
    summaries : list[dict]
        Summaries sorted newest-first.

    Returns
    -------
    dict
        Trend information: best score, best feasible score, older/newer
        median comparison, and total run count.
    """
    from statistics import median as _median

    successful = [s for s in summaries if s.get("success")]
    if not successful:
        return {
            "best_score": None,
            "best_feasible_score": None,
            "trend": "no_data",
            "total_runs": 0,
        }

    scores = [s.get("total_score", float("inf")) for s in successful]
    best = min(scores)

    feasible = []
    for s in successful:
        margins = s.get("margins", {})
        all_satisfied = all(
            v >= 0 for v in margins.values() if isinstance(v, (int, float))
        )
        if all_satisfied:
            feasible.append(s.get("total_score", float("inf")))
    best_feasible = min(feasible) if feasible else None

    mid = len(successful) // 2
    newer = successful[:mid] if mid > 0 else successful
    older = successful[mid:] if mid > 0 else []

    newer_scores = [s.get("total_score", float("inf")) for s in newer]
    older_scores = [s.get("total_score", float("inf")) for s in older]

    newer_med = _median(newer_scores) if newer_scores else None
    older_med = _median(older_scores) if older_scores else None

    if newer_med is not None and older_med is not None:
        if newer_med < older_med * 0.9:
            trend = "improving"
        elif newer_med > older_med * 1.1:
            trend = "regressing"
        else:
            trend = "plateaued"
    else:
        trend = "insufficient_data"

    return {
        "best_score": best,
        "best_feasible_score": best_feasible,
        "feasible_count": len(feasible),
        "total_successful": len(successful),
        "newer_median": newer_med,
        "older_median": older_med,
        "trend": trend,
    }


def build_context(
    done_dir: Path,
    policy_path: Path,
    *,
    max_summaries: int = 200,
    submissions_root: Path | None = None,
    failures_path: Path | None = None,
) -> Dict[str, Any]:
    """Build the full context payload for the proposer.

    Loads summaries from cases/done (or from submissions + failures file when
    both submissions_root and failures_path are provided), computes failure
    stats and top parents, generates run cards and postmortems for the LLM
    proposer.

    Parameters
    ----------
    done_dir : Path
        Directory containing completed case summaries (cases/done/*/summary.json).
        Used only when submissions_root and failures_path are not both provided.
    policy_path : Path
        Path to proposer_policy.yaml.
    max_summaries : int, optional
        Maximum number of summaries to load (default 200).
    submissions_root : Path | None, optional
        Root submissions directory (e.g. submissions/). When provided together
        with failures_path, loads successful runs from submissions and
        failures from the file instead of done_dir.
    failures_path : Path | None, optional
        Path to policy/autopilot_failures.json. When provided together with
        submissions_root, loads failures from this file.

    Returns
    -------
    dict
        Context with policy, failure_stats, top_parents, recent_config_hashes,
        surface_exploration_counts, run_cards, postmortems, total_completed.
    """
    # --- Policy load ---
    policy = _load_policy(policy_path)

    # --- Summaries ---
    if submissions_root is not None and failures_path is not None:
        success_summaries = _load_summaries_from_submissions(
            submissions_root, limit=max_summaries
        )
        failure_summaries = _load_failures_from_file(failures_path, limit=max_summaries)
        summaries = success_summaries + failure_summaries
        summaries.sort(key=lambda s: s.get("case_id", ""), reverse=True)
        summaries = summaries[:max_summaries]
        cases_dir = submissions_root.parent / "cases"
    else:
        summaries = _load_summaries(done_dir, limit=max_summaries)
        cases_dir = done_dir.parent

    # --- Failure stats ---
    window = policy.get("guardrails", {}).get("sliding_window", 30)
    top_k = policy.get("top_k_parents", 10)

    failure_stats = compute_failure_stats(summaries, window=window)
    top_parents = get_top_parents(summaries, top_k=top_k)
    config_hashes = get_recent_config_hashes(summaries)

    # LLM proposer limits from policy
    llm_cfg = policy.get("llm_proposer", {})
    max_cards = int(llm_cfg.get("max_run_cards", 10))
    max_pm = int(llm_cfg.get("max_postmortems", 5))

    # Generate run cards and postmortems for LLM context
    run_cards = _generate_run_cards(summaries, top_k=max_cards)
    postmortems = _generate_postmortems(summaries, max_pm=max_pm)

    # Surfaces explored so far (count per surface)
    surface_counts: Counter[str] = Counter()
    for s in summaries:
        cfg = s.get("case_config", {})
        surface = get_surface_filename(cfg) or "unknown"
        surface_counts[surface] += 1

    # Constraint margin summary and score trend for LLM context
    margin_summary = _compute_margin_summary(summaries)
    score_trend = _compute_score_trend(summaries)

    # Baseline reference cases from cases/*.yaml (curated configs that work)
    # cases_dir already set above (submissions_root.parent / "cases" or done_dir.parent)
    allowed_surfaces = policy.get("exploration", {}).get("surfaces", [])
    baseline_cases = _load_baseline_cases(
        cases_dir, allowed_surfaces, max_per_surface=3
    )

    # --- Output ---
    ctx: Dict[str, Any] = {
        "policy": {
            "batch_size": policy.get("batch_size", 8),
            "exploit_fraction": policy.get("exploit_fraction", 0.5),
            "resource_caps": policy.get("resource_caps", {}),
        },
        "failure_stats": failure_stats,
        "top_parents": top_parents,
        "recent_config_hashes": config_hashes,
        "surface_exploration_counts": dict(surface_counts.most_common()),
        "run_cards": run_cards,
        "postmortems": postmortems,
        "total_completed": len(summaries),
        "margin_summary": margin_summary,
        "score_trend": score_trend,
        "baseline_cases": baseline_cases,
    }
    return ctx


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--done-dir",
        type=Path,
        default=Path("cases/done"),
        help="Directory containing completed case summaries.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("policy/proposer_policy.yaml"),
        help="Path to proposer_policy.yaml.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write context JSON to this file (default: stdout).",
    )
    args = parser.parse_args()

    ctx = build_context(args.done_dir, args.policy)
    text = json.dumps(ctx, indent=2)

    if args.out:
        args.out.write_text(text)
        print(f"Wrote context to {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
