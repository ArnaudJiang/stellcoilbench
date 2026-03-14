"""LLM proposer: apply LLM actions, KB /propose, and direct LLM calls."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List

from stellcoilbench.validate_config import validate_ci_case

from .ga import _config_hash_short, _new_case_id, _rng
from .proposer import propose_batch
from .reasoning import _append_reasoning_to_history, _load_prior_reasoning

__all__ = ["apply_llm_action", "propose_batch_llm_direct"]


def apply_llm_action(
    action: Dict[str, Any],
    ctx: Dict[str, Any],
    policy: Dict[str, Any],
    rng,
) -> Dict[str, Any] | None:
    """Convert one LLM mutation/exploration action to a CI case dict."""
    action_type = action.get("type", "")
    if action_type == "mutate":
        parent_id = action.get("parent_id", "")
        overrides = action.get("overrides", {})
        parents = {p.get("case_id"): p for p in ctx.get("top_parents", [])}
        parent = parents.get(parent_id)
        if not parent:
            return None
        child_cfg = copy.deepcopy(parent.get("case_config", {}))
        if "surface" in overrides:
            sp = child_cfg.setdefault("surface_params", {})
            sp["surface"] = overrides["surface"]
        if "ncoils" in overrides:
            cp = child_cfg.setdefault("coils_params", {})
            cp["ncoils"] = int(overrides["ncoils"])
        if "order" in overrides:
            cp = child_cfg.setdefault("coils_params", {})
            cp["order"] = int(overrides["order"])
        obj = child_cfg.get("coil_objective_terms") or {}
        for k in [
            "cc_threshold",
            "cs_threshold",
            "curvature_threshold",
            "msc_threshold",
            "length_threshold",
            "flux_threshold",
            "force_threshold",
            "torque_threshold",
            "torsion_threshold",
        ]:
            if k in overrides and isinstance(overrides[k], (int, float)):
                obj[k] = overrides[k]
        child_cfg["coil_objective_terms"] = obj
        child_cfg["description"] = f"LLM mutate: {parent_id}"
        opt = child_cfg.get("optimizer_params", {})
        opt["algorithm"] = "augmented_lagrangian"
        opt["verbose"] = True
        mut = policy.get("mutation", {})
        opt["max_iterations"] = mut.get("max_iterations", 500)
        child_cfg["optimizer_params"] = opt
        fc = policy.get("fourier_continuation", {})
        if fc and fc.get("enabled") and fc.get("orders"):
            child_cfg["fourier_continuation"] = {
                "enabled": True,
                "orders": list(fc["orders"]),
            }
        caps = policy.get("resource_caps", {})
        case = {
            "case_id": _new_case_id(),
            "parent_ids": [parent_id],
            "tags": ["exploit", "llm"],
            "proposer_mode": "llm",
            "resource": {
                "max_total_iterations": min(
                    opt.get("max_iterations", 500),
                    caps.get("max_total_iterations", 10000),
                ),
                "timeout_minutes": caps.get("timeout_minutes_max", 60),
            },
            "case_config": child_cfg,
            "random_seed": rng.randint(0, 2**31 - 1),
        }
        return case

    if action_type == "explore":
        surface = action.get("surface")
        ncoils = action.get("ncoils", 4)
        order = action.get("order", 8)
        thresholds = action.get("thresholds", {})
        expl = policy.get("exploration", {})
        surfaces = expl.get("surfaces", ["input.LandremanPaul2021_QA"])
        if surface not in surfaces:
            surface = surfaces[0] if surfaces else "input.LandremanPaul2021_QA"
        coil_objective_terms: Dict[str, Any] = {
            "total_length": "l2_threshold",
            "coil_curvature": "lp_threshold",
            "coil_curvature_p": 2,
            "coil_mean_squared_curvature": "l2_threshold",
            "coil_arclength_variation": "l2_threshold",
            "linking_number": "",
        }
        for k, v in thresholds.items():
            if isinstance(v, (int, float)):
                coil_objective_terms[k] = v
        # Add objective terms when torque/torsion thresholds are proposed
        if "torque_threshold" in thresholds:
            coil_objective_terms["coil_coil_torque"] = "lp_threshold"
        if "torsion_threshold" in thresholds:
            coil_objective_terms["coil_torsion"] = "lp_threshold"
            coil_objective_terms["coil_torsion_p"] = 2
        max_iterations = expl.get("max_iterations", 500)
        case_config: Dict[str, Any] = {
            "description": f"LLM explore: {surface} ncoils={ncoils} order={order}",
            "surface_params": {"surface": surface, "range": "half period"},
            "coils_params": {"ncoils": int(ncoils), "order": int(order)},
            "optimizer_params": {
                "algorithm": "augmented_lagrangian",
                "max_iterations": max_iterations,
                "verbose": True,
            },
            "coil_objective_terms": coil_objective_terms,
        }
        fc = policy.get("fourier_continuation", {})
        if fc and fc.get("enabled") and fc.get("orders"):
            case_config["fourier_continuation"] = {
                "enabled": True,
                "orders": list(fc["orders"]),
            }
        caps = policy.get("resource_caps", {})
        case = {
            "case_id": _new_case_id(),
            "parent_ids": [],
            "tags": ["explore", "llm"],
            "proposer_mode": "llm",
            "resource": {
                "max_total_iterations": min(
                    max_iterations, caps.get("max_total_iterations", 10000)
                ),
                "timeout_minutes": caps.get("timeout_minutes_max", 60),
            },
            "case_config": case_config,
            "random_seed": rng.randint(0, 2**31 - 1),
        }
        return case

    return None


def propose_batch_llm_direct(
    ctx: Dict[str, Any],
    policy: Dict[str, Any],
    batch_size: int = 8,
    seed: int | None = None,
    *,
    reasoning_history_path: Path | None = None,
) -> List[Dict[str, Any]]:
    """Propose a batch using the LLM directly, without a KB server."""
    try:
        from knowledge.llm_endpoints import call_propose
    except ImportError as e:
        print(f"[LLM] IMPORT ERROR: {e}. Falling back to GA.", file=sys.stderr)
        return propose_batch(ctx, policy, batch_size=batch_size, seed=seed)

    run_cards = ctx.get("run_cards") or []
    postmortems = ctx.get("postmortems") or []
    print(
        f"[LLM] Context: run_cards={len(run_cards)}, postmortems={len(postmortems)}",
        file=sys.stderr,
    )
    print(f"[LLM] Calling LLM proposer (batch_size={batch_size})...", file=sys.stderr)

    prior_reasoning: List[str] = []
    if reasoning_history_path:
        prior_reasoning = _load_prior_reasoning(reasoning_history_path, policy)
        if prior_reasoning:
            print(
                f"[LLM] Loaded {len(prior_reasoning)} prior reasoning block(s).",
                file=sys.stderr,
            )

    try:
        resp = call_propose(
            ctx,
            policy,
            batch_size=batch_size,
            run_cards=run_cards,
            postmortems=postmortems,
            surface_counts=ctx.get("surface_exploration_counts"),
            prior_reasoning=prior_reasoning if prior_reasoning else None,
        )
    except Exception as e:
        print(f"[LLM] API CALL FAILED: {e}. Falling back to GA.", file=sys.stderr)
        return propose_batch(ctx, policy, batch_size=batch_size, seed=seed)

    actions = resp.get("actions", [])
    if resp.get("error"):
        print(
            f"[LLM] PROPOSER ERROR: {resp['error']}. Falling back to GA.",
            file=sys.stderr,
        )
        return propose_batch(ctx, policy, batch_size=batch_size, seed=seed)

    if not actions:
        print(
            "[LLM] EMPTY RESPONSE: no actions returned. Falling back to GA.",
            file=sys.stderr,
        )
        return propose_batch(ctx, policy, batch_size=batch_size, seed=seed)

    print(f"[LLM] Received {len(actions)} actions from LLM.", file=sys.stderr)
    for i, a in enumerate(actions):
        reasoning_preview = (a.get("reasoning") or "")[:180]
        if len(a.get("reasoning") or "") > 180:
            reasoning_preview += "..."
        print(
            f"[LLM]   action[{i}]: type={a.get('type')}, "
            f"parent_id={a.get('parent_id', 'N/A')}, "
            f"surface={a.get('surface', a.get('overrides', {}).get('surface', 'N/A'))}",
            file=sys.stderr,
        )
        if reasoning_preview:
            print(f"[LLM]      reasoning: {reasoning_preview}", file=sys.stderr)

    rng = _rng(seed)
    recent_hashes = set(ctx.get("recent_config_hashes", []))
    cases: List[Dict[str, Any]] = []
    seen_hashes: set = set()
    reasoning_entries: List[Dict[str, Any]] = []

    for action in actions[:batch_size]:
        case = apply_llm_action(action, ctx, policy, rng)
        if not case:
            print(
                f"[LLM]   Skipped invalid action: {action.get('type')}", file=sys.stderr
            )
            continue
        h = _config_hash_short(case.get("case_config", {}))
        if h in recent_hashes or h in seen_hashes:
            print(
                f"[LLM]   Skipped duplicate config: {case.get('case_id')}",
                file=sys.stderr,
            )
            continue
        errors = validate_ci_case(case, policy=policy)
        if errors:
            print(
                f"[LLM]   Validation failed for {case.get('case_id')}: {errors}",
                file=sys.stderr,
            )
            continue
        reasoning = action.get("reasoning", "")
        case["llm_reasoning"] = reasoning
        reasoning_entries.append(
            {
                "case_id": case["case_id"],
                "type": action.get("type", "?"),
                "reasoning": reasoning,
            }
        )
        seen_hashes.add(h)
        cases.append(case)

    llm_count = len(cases)
    if len(cases) < batch_size:
        ga_needed = batch_size - len(cases)
        print(
            f"[LLM] {llm_count} valid LLM cases, filling {ga_needed} with GA.",
            file=sys.stderr,
        )
        extra = propose_batch(ctx, policy, batch_size=ga_needed, seed=seed)
        for c in extra:
            h = _config_hash_short(c.get("case_config", {}))
            if h not in seen_hashes:
                seen_hashes.add(h)
                cases.append(c)
                if len(cases) >= batch_size:
                    break
    else:
        print(
            f"[LLM] All {llm_count} cases from LLM (no GA fill needed).",
            file=sys.stderr,
        )

    if reasoning_history_path and reasoning_entries:
        _append_reasoning_to_history(reasoning_history_path, reasoning_entries)
        print(
            f"[LLM] Appended {len(reasoning_entries)} reasoning entries to {reasoning_history_path}",
            file=sys.stderr,
        )

    return cases[:batch_size]
