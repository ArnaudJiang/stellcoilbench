#!/usr/bin/env python3
"""
LLM-powered endpoints for the StellCoilBench case proposer.

Provides **call_propose**: generates a batch of mutation/exploration actions (JSON)
for the CI autopilot. Uses top parent runs, failure stats, and policy constraints
to propose new cases.  The system prompt is enriched with domain context from
``knowledge/`` (surface catalog, optimization guide, threshold scaling).

Requires the LLM to be configured via KB_LLM_* environment variables
(see knowledge.llm_client).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from stellcoilbench.path_utils import get_surface_filename
except ImportError:
    import sys
    _repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repo / "src"))
    from stellcoilbench.path_utils import get_surface_filename

_KNOWLEDGE_DIR = Path(__file__).resolve().parent


def _load_context_doc(path: Path) -> str:
    """Load a context document, returning empty string on any error.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the document.

    Returns
    -------
    str
        File contents, or ``""`` if the file is missing or unreadable.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _load_surface_catalog_text() -> str:
    """Load ``surface_catalog.json`` and format it for the system prompt.

    Returns
    -------
    str
        Human-readable surface catalog text, or ``""`` on failure.
    """
    raw = _load_context_doc(_KNOWLEDGE_DIR / "surface_catalog.json")
    if not raw:
        return ""
    try:
        catalog = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    lines = ["## Plasma Surface Catalog"]
    ref = catalog.get("reference", {})
    if ref:
        lines.append(
            f"Reference: ARIES-CS a={ref.get('ARIES_CS_minor_radius_m', 1.7)} m, "
            f"R={ref.get('ARIES_CS_major_radius_m', 7.75)} m"
        )
    for s in catalog.get("surfaces", []):
        lines.append(
            f"- {s['file']} ({s.get('short_name', '?')}): "
            f"a={s.get('minor_radius_m', '?')} m, a0={s.get('a0', '?')}, "
            f"nfp={s.get('nfp', '?')}, stellsym={s.get('stellsym', '?')}, "
            f"typical ncoils={s.get('typical_ncoils', '?')}, "
            f"orders={s.get('typical_orders', '?')}. "
            f"{s.get('notes', '')}"
        )
    return "\n".join(lines)


_SURFACE_CATALOG_TEXT: str = _load_surface_catalog_text()
_LLM_CONTEXT_TEXT: str = _load_context_doc(_KNOWLEDGE_DIR / "llm_context.md")


_PROPOSE_SYSTEM_BASE = """You are an expert at proposing stellarator coil optimization cases for StellCoilBench.
Given context (top runs, failure stats, constraint margins, score trends, policy constraints), output a JSON array of mutation actions.
Each action is either:
- {{"type": "mutate", "parent_id": "<case_id>", "overrides": {{"ncoils": 5, "cc_threshold": 1.2, "surface": "...", ...}}}, "reasoning": "Brief justification (1-3 sentences)."}}
- {{"type": "explore", "surface": "...", "ncoils": 4, "order": 8, "thresholds": {{"cc_threshold": 1.0, "cs_threshold": 2.0, "curvature_threshold": 2.0, "msc_threshold": 2.0, "length_threshold": 150}}}, "reasoning": "Brief justification (1-3 sentences)."}}

The "reasoning" field is REQUIRED: 1-3 sentences per action. Explain which run/parent you're building on, which failure you're avoiding, or why you're exploring this region. Cite run IDs or failure modes when relevant. Short justifications are fine.

Rules:
- Only use surfaces, ncoils, order from the policy's allowed lists.
- For mutate: parent_id must be from top_parents. Overrides are optional (ncoils, order, surface, cc_threshold, cs_threshold, curvature_threshold, msc_threshold, length_threshold, force_threshold, torque_threshold, torsion_threshold).
- For explore: surface, ncoils, order required; thresholds optional (use policy ranges).
- All thresholds must be specified at ARIES-CS reactor scale (a=1.7 m). The optimizer auto-rescales to device scale.
- Use the surface catalog below to understand each surface's a0, nfp, and typical coil configurations.
- Use failure postmortems and run cards to avoid repeating failed configurations.
- Prioritize under-explored surfaces when surface exploration counts are provided.

Strategy guidance:
- When the score trend is "plateaued", prefer exploration over mutation to discover new regions.
- Use constraint margin data to choose intelligent mutations: if a parent has tight margins on a constraint (e.g. cc or msc), relax that threshold; if it has ample margins, tighten the threshold to push for better solutions.
- Prioritize finding FEASIBLE solutions (all constraints satisfied) over just minimizing score.
- When mutating a parent with score=0, focus on adjusting thresholds to improve constraint satisfaction rather than score.
- Diversify across different ncoils values and threshold combinations to map the Pareto frontier.
- When baseline reference cases are provided, use their thresholds as a starting point for explore actions; they are curated configs known to work.
- Output ONLY a valid JSON array of actions. Each action must include a "reasoning" string (1-3 sentences). No other text."""


def _build_propose_system() -> str:
    """Build the full PROPOSE system prompt with domain context documents.

    Includes the surface catalog, threshold scaling guide, optimization
    guide, and curated literature context so the LLM has domain knowledge
    even without a running Qdrant RAG pipeline.

    Returns
    -------
    str
        System prompt string with domain docs appended.
    """
    parts = [_PROPOSE_SYSTEM_BASE]
    if _SURFACE_CATALOG_TEXT:
        parts.append(f"\n\n{_SURFACE_CATALOG_TEXT}")
    if _LLM_CONTEXT_TEXT:
        parts.append(f"\n\n{_LLM_CONTEXT_TEXT}")
    return "\n".join(parts)


PROPOSE_SYSTEM: str = _build_propose_system()


def call_propose(
    context: dict,
    policy: dict,
    batch_size: int = 8,
    *,
    run_cards: list[str] | None = None,
    postmortems: list[str] | None = None,
    surface_counts: dict[str, int] | None = None,
    prior_reasoning: list[str] | None = None,
) -> dict[str, Any]:
    """Generate mutation/exploration actions for the next CI batch.

    Parameters
    ----------
    context : dict
        Build context with top_parents, failure_stats, recent_config_hashes.
    policy : dict
        Proposer policy (exploration surfaces, ncoils, order, mutation ranges).
        The ``llm_proposer.temperature`` key is used for LLM sampling
        (default 0.3).
    batch_size : int, optional
        Number of actions to propose (default 8).
    run_cards : list[str] | None, optional
        Pre-formatted run card texts for top parents (from ``make_run_card``).
        When provided, these replace the one-line parent summaries.
    postmortems : list[str] | None, optional
        Pre-formatted postmortem texts for recent failures (from
        ``make_postmortem``).  Helps the LLM avoid repeating failed configs.
    surface_counts : dict[str, int] | None, optional
        Map of surface name to number of completed runs.  Helps the LLM
        prioritise under-explored surfaces.
    prior_reasoning : list[str] | None, optional
        Explanations from previous proposal batches. Each string is a
        past run's reasoning block. Helps the LLM build on its own prior
        choices and avoid repeating or contradicting past reasoning.

    Returns
    -------
    dict
        {"actions": list[dict]} or {"error": str, "actions": []} if LLM fails.
        Each action is {"type": "mutate"|"explore", ...} with overrides/surface/etc.
        Actions may include a "reasoning" field when the LLM provides it.
    """
    try:
        from knowledge.llm_client import complete_json, is_available
    except ImportError:
        return {"error": "LLM not available", "actions": []}

    if not is_available():
        return {"error": "LLM not configured (set KB_LLM_* env vars)", "actions": []}

    top_parents = context.get("top_parents", [])
    failure_stats = context.get("failure_stats", {})

    # Build allowed values from policy
    expl = policy.get("exploration", {})
    surfaces = expl.get("surfaces", ["input.LandremanPaul2021_QA"])
    ncoils_choices = expl.get("ncoils_choices", [3, 4, 5, 6, 7])
    order_choices = expl.get("order_choices", [4, 6, 8])
    mut = policy.get("mutation", {})
    if not ncoils_choices:
        ncoils_choices = mut.get("ncoils_choices", [3, 4, 5, 6, 7])
    if not order_choices:
        order_choices = mut.get("order_choices", [4, 6, 8])

    # Threshold ranges from policy (for LLM guidance)
    threshold_ranges: dict[str, list[float]] = {}
    for key in (
        "cc_threshold_range", "cs_threshold_range", "curvature_threshold_range",
        "msc_threshold_range", "length_threshold_range", "force_threshold_range",
        "torque_threshold_range", "torsion_threshold_range",
    ):
        val = expl.get(key)
        if val and isinstance(val, list) and len(val) == 2:
            threshold_ranges[key.replace("_range", "")] = val

    # --- Build user prompt sections ---
    sections: list[str] = []

    sections.append(f"Propose {batch_size} cases for the next optimization batch.")

    # Prior reasoning from previous batches — build on your past choices
    if prior_reasoning:
        llm_cfg = policy.get("llm_proposer", {})
        max_prior = int(llm_cfg.get("max_prior_reasoning_batches", 5))
        prior_text = "\n\n".join(prior_reasoning[:max_prior])
        sections.append(
            "Your reasoning from previous proposal batches (build on this, avoid repeating):\n"
            f"{prior_text}"
        )

    # Policy constraints
    policy_block = (
        f"Policy constraints:\n"
        f"- Allowed surfaces: {surfaces}\n"
        f"- Allowed ncoils: {ncoils_choices}\n"
        f"- Allowed order: {order_choices}"
    )
    if threshold_ranges:
        policy_block += f"\n- Threshold ranges (reactor scale): {json.dumps(threshold_ranges)}"
    sections.append(policy_block)

    # Top parent runs — use run cards if provided, else compact summaries
    if run_cards:
        llm_cfg = policy.get("llm_proposer", {})
        max_cards = int(llm_cfg.get("max_run_cards", 10))
        cards_text = "\n---\n".join(run_cards[:max_cards])
        sections.append(f"Top parent runs (detailed run cards):\n{cards_text}")
    else:
        parent_summaries = []
        for p in top_parents[:10]:
            cid = p.get("case_id", "?")
            score = p.get("total_score", "?")
            cfg = p.get("case_config", {})
            surface = get_surface_filename(cfg) or "?"
            cp = cfg.get("coils_params", {})
            ncoils = cp.get("ncoils", "?") if isinstance(cp, dict) else "?"
            order = cp.get("order", "?") if isinstance(cp, dict) else "?"
            parent_summaries.append(
                f"  - {cid}: score={score}, surface={surface}, "
                f"ncoils={ncoils}, order={order}"
            )
        sections.append(
            f"Top parent runs (for mutate):\n"
            f"{chr(10).join(parent_summaries) if parent_summaries else '  (none)'}"
        )

    # Failure stats + postmortems
    stats_line = (
        f"Failure stats: fail_rate={failure_stats.get('fail_rate', 0):.2f}, "
        f"failure_classes={failure_stats.get('failure_classes', {})}"
    )
    sections.append(stats_line)

    if postmortems:
        llm_cfg = policy.get("llm_proposer", {})
        max_pm = int(llm_cfg.get("max_postmortems", 5))
        non_empty = [pm for pm in postmortems if pm.strip()][:max_pm]
        if non_empty:
            pm_text = "\n---\n".join(non_empty)
            sections.append(
                f"Recent failure postmortems (avoid these patterns):\n{pm_text}"
            )

    # Surface exploration counts
    sc = surface_counts or context.get("surface_exploration_counts")
    if sc:
        sc_lines = [f"  {surf}: {cnt} runs" for surf, cnt in sc.items()]
        sections.append(
            "Surface exploration coverage:\n" + "\n".join(sc_lines)
        )

    # Constraint margin summary — tells LLM which constraints are hardest
    margin_summary = context.get("margin_summary", {})
    if margin_summary:
        margin_lines = []
        for cname, info in sorted(margin_summary.items(),
                                   key=lambda x: -x[1].get("violation_rate", 0)):
            vr = info.get("violation_rate", 0)
            med = info.get("median_margin", 0)
            margin_lines.append(
                f"  {cname}: violated in {vr:.0%} of runs, median margin={med:.4f}"
            )
        sections.append(
            "Constraint satisfaction analysis (higher violation = harder constraint):\n"
            + "\n".join(margin_lines)
        )

    # Score trend and Pareto analysis
    score_trend = context.get("score_trend", {})
    if score_trend and score_trend.get("trend") != "no_data":
        trend_lines = [
            f"Score trend: {score_trend.get('trend', '?')}",
            f"  Best score overall: {score_trend.get('best_score', '?'):.2e}" if isinstance(score_trend.get("best_score"), (int, float)) else "",
            f"  Best feasible score (all constraints satisfied): {score_trend.get('best_feasible_score', '?'):.2e}" if isinstance(score_trend.get("best_feasible_score"), (int, float)) else "  Best feasible score: none found",
            f"  Feasible runs: {score_trend.get('feasible_count', 0)}/{score_trend.get('total_successful', 0)}",
        ]
        newer_med = score_trend.get("newer_median")
        older_med = score_trend.get("older_median")
        if newer_med is not None and older_med is not None:
            trend_lines.append(
                f"  Recent median score: {newer_med:.2e}, older median: {older_med:.2e}"
            )
        sections.append("\n".join(line for line in trend_lines if line))

    # Baseline reference cases (curated configs from cases/*.yaml)
    baseline_cases = context.get("baseline_cases", [])
    if baseline_cases:
        sections.append(
            "Baseline reference cases (curated configs from cases/*.yaml):\n"
            "Use these as reference for reasonable thresholds and ncoils/order "
            "when proposing explore actions or when there are few/no done runs "
            "for a surface. Prefer thresholds close to baseline when exploring new regions.\n\n"
            + "\n\n".join(baseline_cases)
        )

    sections.append(
        f"Output a JSON array of exactly {batch_size} actions. "
        f"Mix mutate and explore. For mutate use parent_id from the run cards above.\n"
        f"Output ONLY the JSON array, no other text."
    )

    user_content = "\n\n".join(sections)

    messages = [
        {"role": "system", "content": PROPOSE_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    llm_cfg = policy.get("llm_proposer", {})
    temperature = float(llm_cfg.get("temperature", 0.3))

    try:
        result = complete_json(messages, max_tokens=4096, temperature=temperature)
        if isinstance(result, list):
            actions = result
        elif isinstance(result, dict) and "actions" in result:
            actions = result["actions"]
        else:
            actions = []
        return {"actions": actions}
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": str(e), "actions": []}
