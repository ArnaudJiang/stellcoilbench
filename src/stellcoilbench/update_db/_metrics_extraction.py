"""Metrics extraction, normalization, and scoring for submissions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from ._constraints import (
    check_reactor_constraints,
    compute_composite_score,
)
from ._path_parsing import (
    load_case_yaml_from_submission,
    parse_submission_path,
)
from ._backfill import backfill_reactor_scale_metrics
from ._recompute import _recompute_coils_linked_to_surface

logger = logging.getLogger(__name__)


def _numeric_fields(values: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric (int/float) entries from a dict, casting to float.

    Parameters
    ----------
    values : dict
        Arbitrary key-value mapping.

    Returns
    -------
    dict[str, float]
        Subset of *values* where the value is ``int`` or ``float``,
        with all values cast to ``float``.
    """
    return {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, (int, float))
    }


def _extract_primary_score(
    metrics: Dict[str, Any],
    metrics_numeric: Dict[str, Any],
    path: Path,
) -> float | None:
    """Extract primary score from metrics with fallback chain.

    Tries score_primary, then final_squared_flux, final_flux,
    final_normalized_squared_flux. Updates metrics_numeric when a fallback
    is used. Logs a warning if a fallback key exists but value is not numeric.

    Parameters
    ----------
    metrics : dict
        Raw metrics from submission.
    metrics_numeric : dict
        Numeric metrics dict; updated with score_primary when fallback used.
    path : Path
        Submission path (for warning messages).

    Returns
    -------
    float | None
        Primary score, or None if not found.
    """
    primary_score = metrics_numeric.get("score_primary")
    if primary_score is not None:
        return float(primary_score)
    for key in ["final_squared_flux", "final_flux", "final_normalized_squared_flux"]:
        fallback = metrics.get(key)
        if isinstance(fallback, (int, float)):
            primary_score = float(fallback)
            metrics_numeric["score_primary"] = primary_score
            return primary_score
        if fallback is not None:
            logger.warning(
                "fallback score '%s' (type %s) is not numeric for %s",
                fallback,
                type(fallback).__name__,
                path,
            )
    return None


def _extract_coil_params_from_case(
    case_yaml_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract coil parameters from case.yaml for leaderboard metrics.

    Reads ``coils_params`` and ``fourier_continuation`` from the case config.

    Parameters
    ----------
    case_yaml_data : dict | None
        Parsed case.yaml content. Returns empty dict if None.

    Returns
    -------
    dict[str, Any]
        Metric keys: ``coil_order``, ``num_coils``, ``fourier_continuation_orders``.
        Only includes keys for which values were found. Caller merges into
        ``metrics_numeric``.
    """
    out: dict[str, Any] = {}
    if not case_yaml_data:
        return out
    coils_params = case_yaml_data.get("coils_params", {})
    if "order" in coils_params:
        out["coil_order"] = float(coils_params["order"])
    if "ncoils" in coils_params:
        out["num_coils"] = float(coils_params["ncoils"])
    fourier_continuation = case_yaml_data.get("fourier_continuation", {})
    if fourier_continuation and fourier_continuation.get("enabled", False):
        orders = fourier_continuation.get("orders", [])
        if orders:
            out["fourier_continuation_orders"] = ",".join(str(o) for o in orders)
    return out


def _normalize_entry_metrics(
    data: Dict[str, Any],
    path: Path,
    repo_root: Path,
    submissions_root: Path,
) -> Dict[str, Any]:
    """Normalise raw submission data into leaderboard-ready metrics.

    Performs the following transformations on a single submission entry:

    1. Extracts numeric metrics.
    2. Loads the companion ``case.yaml`` to resolve coil parameters
       (order, count, Fourier-continuation orders).
    3. Falls back to ``coils.json`` when case.yaml is incomplete.
    4. Computes the primary score (with fallback chain).
    5. Resolves the submission path to a repo-relative string.
    6. Determines the GitHub username from path structure.

    Parameters
    ----------
    data : dict
        Parsed submission JSON (must contain ``"metadata"`` and
        ``"metrics"`` keys, as produced by
        :func:`normalize_submission_metrics`).
    path : Path
        Path to ``results.json`` or the containing zip file.
    repo_root : Path
        Repository root for resolving relative paths and plasma surfaces.
    submissions_root : Path
        Root of the ``submissions/`` tree (for path parsing).

    Returns
    -------
    dict[str, Any]
        Keys: ``metrics_numeric``, ``primary_score``, ``rel_path``,
        ``github_username``.
    """
    from ._load_submissions import _extract_coil_params_from_coils_json

    meta = data.get("metadata") or {}
    metrics = data.get("metrics") or {}

    metrics_numeric = _numeric_fields(metrics)

    case_yaml_data = load_case_yaml_from_submission(path)

    coil_params = _extract_coil_params_from_case(case_yaml_data)
    coil_params.update(
        _extract_coil_params_from_coils_json(
            path, case_yaml_data, repo_root, submissions_root, coil_params
        )
    )
    for k, v in coil_params.items():
        metrics_numeric[k] = v

    primary_score = _extract_primary_score(metrics, metrics_numeric, path)

    abs_path = path if path.is_absolute() else (repo_root / path).resolve()
    try:
        rel_path = str(abs_path.relative_to(repo_root.resolve()))
    except ValueError:
        rel_path = str(abs_path)

    parsed_path = parse_submission_path(path, submissions_root)
    github_username = (
        parsed_path["user"]
        if parsed_path["user"] != "unknown"
        else meta.get("contact", "")
    )

    return {
        "metrics_numeric": metrics_numeric,
        "primary_score": primary_score,
        "rel_path": rel_path,
        "github_username": github_username,
    }


def _compute_submission_score(
    method_key: str,
    path: Path,
    metrics: Dict[str, Any],
    metrics_numeric: Dict[str, Any],
    reactor_scale: Dict[str, Any],
    repo_root: Path,
) -> Dict[str, Any]:
    """Backfill, recompute linkage, check constraints, and score a submission.

    Orchestrates the reactor-scale evaluation pipeline for a single
    submission:

    1. Back-fills missing reactor-scale metrics from device-scale data.
    2. Recomputes ``coils_linked_to_surface`` using a per-phi-slice
       check (more accurate than the legacy global R-range check).
    3. Evaluates hard and soft reactor-scale engineering constraints.
    4. Computes the composite feasibility/quality score.

    Parameters
    ----------
    method_key : str
        ``"contact:surface:user:version"`` identifier — the surface name
        is extracted from position 1 to drive the linkage recomputation.
    path : Path
        Path to the submission ``results.json`` or zip.
    metrics : dict
        Device-scale metrics (mutable — ``coils_linked_to_surface`` may
        be updated in-place when recomputed).
    metrics_numeric : dict
        Numeric metrics dict (mutable — same key updated in-place).
    reactor_scale : dict
        Reactor-scale metrics (mutable — back-filled in-place).
    repo_root : Path
        Repository root for plasma-surface resolution.

    Returns
    -------
    dict[str, Any]
        Keys: ``passes_constraints`` (bool), ``violations`` (list),
        ``composite_score`` (float | None), ``score_details`` (dict).
    """
    backfill_reactor_scale_metrics(reactor_scale, metrics)

    key_parts = method_key.split(":")
    surface_from_key = key_parts[1] if len(key_parts) >= 2 else ""
    if surface_from_key and surface_from_key != "unknown":
        corrected = _recompute_coils_linked_to_surface(
            path, surface_from_key, repo_root
        )
        if corrected is not None:
            old_val = metrics_numeric.get("coils_linked_to_surface")
            new_val = float(corrected)
            metrics_numeric["coils_linked_to_surface"] = new_val
            metrics["coils_linked_to_surface"] = corrected
            if old_val is not None and old_val != new_val:
                logger.info(
                    "Recomputed coils_linked_to_surface: %s → %s for %s",
                    old_val,
                    new_val,
                    path,
                )

    passes_constraints, violations = check_reactor_constraints(metrics, reactor_scale)
    if not passes_constraints:
        logger.warning("%s fails reactor-scale constraints:", path)
        for v in violations:
            op = "≤" if v["direction"] == "max" else "≥"
            if v["direction"] == "eq":
                op = "=="
            logger.warning(
                "  %s: %s (bound %s %s %s)%s",
                v["label"],
                v["value"],
                op,
                v["bound"],
                v["units"],
                " [HARD]" if v.get("hard") else "",
            )

    composite_score, score_details = compute_composite_score(metrics, reactor_scale)

    return {
        "passes_constraints": passes_constraints,
        "violations": violations,
        "composite_score": composite_score,
        "score_details": score_details,
    }
