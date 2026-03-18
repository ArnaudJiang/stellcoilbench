"""Load submissions and build in-memory leaderboard data structures."""

from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from ..constants import CONSTRAINT_VIOLATIONS_KEY


def _flatten_continuation_metrics_for_reactor(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract final-step metrics from continuation_results when top-level keys are missing.

    Reactor-scale compute needs target_B_field, _cached_thresholds (major/minor_radius),
    and device metrics (final_min_cc_separation, etc.). Some submissions only have
    continuation_results; use the last step's dict when top-level lacks these keys.

    Parameters
    ----------
    metrics : dict
        Raw metrics dict (may have continuation_results).

    Returns
    -------
    dict
        Metrics suitable for compute_reactor_scale_metrics (flattened when needed).
    """
    if not metrics:
        return metrics
    # Already have required keys at top level
    if metrics.get("target_B_field") is not None and (
        metrics.get("_cached_thresholds") or metrics.get("final_min_cc_separation") is not None
    ):
        return metrics
    continuation = metrics.get("continuation_results")
    if not continuation or not isinstance(continuation, list):
        return metrics
    last = continuation[-1]
    if not isinstance(last, dict):
        return metrics
    # Merge last step into metrics; last step overrides when both exist
    out = dict(metrics)
    for k, v in last.items():
        if out.get(k) is None and v is not None:
            out[k] = v
    return out
from ._path_parsing import (
    _extract_surface_from_submission_path,
    load_case_yaml_from_submission,
    parse_submission_path,
)

from ._load_submissions import (
    _extract_coil_params_from_coils_json,  # noqa: F401 (re-exported)
    _load_submissions,
)
from ._metrics_extraction import (
    _compute_submission_score,
    _extract_coil_params_from_case,  # noqa: F401 (re-exported)
    _extract_primary_score,  # noqa: F401 (re-exported)
    _normalize_entry_metrics,
)

logger = logging.getLogger(__name__)

_REACTOR_SCALE_NON_METRIC_KEYS = frozenset(
    {"reference", "error", "scaling_factors", "jc_model"}
)


def _reactor_scale_completeness(rs: Dict[str, Any]) -> int:
    """Count reactor-scale metric keys (exclude reference, error, metadata)."""
    if not rs:
        return 0
    return sum(
        1
        for k in rs
        if k not in _REACTOR_SCALE_NON_METRIC_KEYS and rs[k] is not None
    )


def _load_case_yaml_fallback(
    path: Path, submissions_root: Path, repo_root: Path
) -> Dict[str, Any] | None:
    """Load case.yaml from submission; if missing, try cases/ in repo by surface match."""
    case_cfg = load_case_yaml_from_submission(path)
    if case_cfg is not None:
        return case_cfg
    parsed = parse_submission_path(path, submissions_root)
    surface = parsed.get("surface", "")
    if not surface or surface == "unknown":
        return None
    cases_dir = repo_root / "cases"
    if not cases_dir.is_dir():
        return None
    try:
        from ..path_utils import get_surface_filename, load_yaml, surface_stem_from_filename

        for yaml_path in cases_dir.glob("*.yaml"):
            try:
                data = load_yaml(path=yaml_path)
                if not isinstance(data, dict):
                    continue
                surf_file = get_surface_filename(data)
                if surf_file:
                    stem = surface_stem_from_filename(surf_file)
                    if stem == surface:
                        return data
            except Exception:
                continue
    except Exception as e:
        logger.debug("Case fallback from cases/ failed for %s: %s", path, e)
    return None


_PER_COIL_KEYS = frozenset({
    "final_max_force_per_coil",
    "final_max_torque_per_coil",
    "final_length_per_coil",
    "final_current_per_coil",
})


def _extract_coils_path_from_submission(
    path: Path,
    repo_root: Path,
) -> Tuple[Path | None, Callable[[], None]]:
    """Return (coils_path, cleanup_fn) for loading coils from a submission.

    For zip files: extracts coils JSON to a temp file; cleanup_fn removes it.
    For directory submissions: returns existing path; cleanup_fn is no-op.

    Parameters
    ----------
    path : Path
        Path to results.json or all_files.zip.
    repo_root : Path
        Repository root (unused for extraction; for future path resolution).

    Returns
    -------
    tuple
        (Path to coils JSON, or None; callable that performs cleanup).
    """

    def noop() -> None:
        pass

    if path.suffix == ".zip":
        try:
            with zipfile.ZipFile(path, "r") as zf:
                # Prefer biot_savart_optimized.json (highest order first), then coils.json
                bs_files = [
                    n for n in zf.namelist()
                    if n.endswith("biot_savart_optimized.json")
                ]
                coils_files = [n for n in zf.namelist() if n.endswith("coils.json")]
                if bs_files:
                    bs_files.sort(reverse=True)
                    chosen = bs_files[0]
                elif coils_files:
                    coils_files.sort(reverse=True)
                    chosen = coils_files[0]
                else:
                    return None, noop
                coils_bytes = zf.read(chosen)
        except (zipfile.BadZipFile, OSError) as e:
            logger.debug("Could not read coils from zip %s: %s", path, e)
            return None, noop

        fd, tmp_path_str = tempfile.mkstemp(suffix=".json")
        try:
            os.write(fd, coils_bytes)
            os.close(fd)
            tmp_path = Path(tmp_path_str)

            def cleanup() -> None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.debug("Could not remove temp coils file %s: %s", tmp_path, exc)

            return tmp_path, cleanup
        except OSError as e:
            os.close(fd)
            try:
                os.unlink(tmp_path_str)
            except OSError:
                pass
            logger.debug("Could not write temp coils file: %s", e)
            return None, noop

    # Directory submission
    from ..path_utils import coils_json_path_from_dir

    coils_path = coils_json_path_from_dir(path.parent)
    return coils_path, noop


def build_methods_json(
    submissions_root: Path,
    repo_root: Path,
) -> Dict[str, Any]:
    """Build the per-method summary dictionary from submissions.

    Scans submissions, normalizes metrics, recomputes coils_linked_to_surface,
    checks reactor-scale constraints, and computes composite scores.

    Parameters
    ----------
    submissions_root : Path
        Root directory containing submission results.json files.
    repo_root : Path
        Repository root for resolving plasma surfaces and relative paths.

    Returns
    -------
    dict[str, Any]
        Keys are ``"contact:surface:user:version"``; values hold metadata,
        metrics, reactor_scale_metrics, composite_score, passes_constraints.
    """
    methods: Dict[str, Any] = {}

    loaded_count = 0
    skipped_no_metrics = 0
    skipped_no_score = 0
    duplicate_keys = {}  # Track duplicate method_keys

    skipped_constraints = 0

    for method_key, path, data in _load_submissions(submissions_root):
        loaded_count += 1
        meta = data.get("metadata") or {}
        metrics = data.get("metrics") or {}
        reactor_scale = data.get("reactor_scale_metrics") or {}

        if not metrics:
            skipped_no_metrics += 1
            logger.warning("Skipping %s - no metrics found", path)
            continue

        # Track duplicate method_keys (prefer entry with more reactor-scale data)
        if method_key in methods:
            if method_key not in duplicate_keys:
                duplicate_keys[method_key] = [methods[method_key].get("path")]
            duplicate_keys[method_key].append(str(path))
            logger.warning(
                "Duplicate method_key '%s'. Previous: %s, New: %s (may overwrite if more complete)",
                method_key,
                methods[method_key].get("path"),
                path,
            )

        normalized = _normalize_entry_metrics(data, path, repo_root, submissions_root)
        metrics_numeric = normalized["metrics_numeric"]
        primary_score = normalized["primary_score"]
        rel_path = normalized["rel_path"]
        github_username = normalized["github_username"]

        # Compute reactor_scale from device metrics when missing (older submissions)
        needs_recompute = (
            len(reactor_scale) < 3 or reactor_scale.get("error") is not None
        )
        if needs_recompute:
            from .._optional_imports import get_reactor_scale_compute
            from ..path_utils import get_surface_filename

            compute_fn = get_reactor_scale_compute()
            case_cfg = _load_case_yaml_fallback(path, submissions_root, repo_root)
            flat_metrics = _flatten_continuation_metrics_for_reactor(metrics)
            if compute_fn is not None:
                try:
                    reactor_scale = compute_fn(flat_metrics, case_cfg)
                except Exception as e:
                    logger.info(
                        "Could not compute reactor_scale for %s: %s", path, e
                    )
            surface_file = get_surface_filename(case_cfg) if case_cfg else None
            coils_path, coils_cleanup = _extract_coils_path_from_submission(
                path, repo_root
            )
            try:
                # Fallback: recompute device metrics from coils when compute failed
                if (
                    (reactor_scale.get("error") or len(reactor_scale) < 3)
                    and case_cfg is not None
                    and coils_path is not None
                    and surface_file
                ):
                    try:
                        from ..coil_optimization import evaluate_external_coils

                        device_metrics = evaluate_external_coils(
                            coils_path,
                            surface_file,
                            plasma_surfaces_dir=repo_root / "plasma_surfaces",
                        )
                        if compute_fn is not None:
                            reactor_scale = compute_fn(device_metrics, case_cfg)
                            logger.info(
                                "Recomputed reactor_scale from coils for %s",
                                path,
                            )
                    except Exception as e:
                        logger.info(
                            "Coil-based reactor_scale recompute failed for %s: %s",
                            path,
                            e,
                        )
                # Backfill turn metrics (L_SC, N_turns, F_turn, etc.) when missing
                elif (
                    reactor_scale.get("error") is None
                    and len(reactor_scale) >= 3
                    and (
                        "N_turns_per_coil" not in reactor_scale
                        or "total_superconductor_length_km" not in reactor_scale
                    )
                    and case_cfg is not None
                    and coils_path is not None
                    and surface_file
                ):
                    try:
                        from ..coil_optimization import evaluate_external_coils

                        device_metrics = evaluate_external_coils(
                            coils_path,
                            surface_file,
                            plasma_surfaces_dir=repo_root / "plasma_surfaces",
                        )
                        merged = dict(flat_metrics)
                        for k in _PER_COIL_KEYS:
                            v = device_metrics.get(k)
                            if v is not None:
                                merged[k] = v
                        reactor_scale = compute_fn(merged, case_cfg)
                        logger.info(
                            "Backfilled turn metrics from coils for %s", path
                        )
                    except Exception as e:
                        logger.info(
                            "Coil-based turn-metrics backfill failed for %s: %s",
                            path,
                            e,
                        )
            finally:
                coils_cleanup()

        if primary_score is None:
            skipped_no_score += 1

        score_result = _compute_submission_score(
            method_key,
            path,
            metrics,
            metrics_numeric,
            reactor_scale,
            repo_root,
        )
        passes_constraints = score_result["passes_constraints"]
        violations = score_result["violations"]
        composite_score = score_result["composite_score"]
        score_details = score_result["score_details"]

        if not passes_constraints:
            skipped_constraints += 1

        new_entry = {
            "method_version": meta.get(
                "method_version",
                path.stem if path.suffix == ".zip" else path.parent.name,
            ),
            "contact": github_username,
            "hardware": meta.get("hardware", ""),
            "run_date": meta.get("run_date", ""),
            "path": rel_path,
            "score_primary": primary_score,
            "composite_score": composite_score,
            "score_details": score_details,
            "metrics": metrics_numeric,
            "reactor_scale_metrics": reactor_scale,
            "passes_constraints": passes_constraints,
            CONSTRAINT_VIOLATIONS_KEY: violations,
        }
        # Prefer entry with more complete reactor-scale data when duplicates exist
        if method_key in methods:
            existing_comp = _reactor_scale_completeness(
                methods[method_key].get("reactor_scale_metrics") or {}
            )
            new_comp = _reactor_scale_completeness(reactor_scale)
            if new_comp > existing_comp:
                methods[method_key] = new_entry
        else:
            methods[method_key] = new_entry

    # Log summary
    total_duplicates = sum(len(paths) - 1 for paths in duplicate_keys.values())
    logger.info(
        "Loaded %d submissions, skipped %d (no metrics), %d will be filtered (no score), %d fail constraints",
        loaded_count,
        skipped_no_metrics,
        skipped_no_score,
        skipped_constraints,
    )
    if duplicate_keys:
        logger.info(
            "Found %d duplicate method_keys (%d overwrites):",
            len(duplicate_keys),
            total_duplicates,
        )
        for key, paths in duplicate_keys.items():
            logger.info(
                "  %s: %d total (first kept, %d overwritten)",
                key,
                len(paths),
                len(paths) - 1,
            )
    expected_entries = loaded_count - skipped_no_metrics - total_duplicates
    logger.info(
        "Methods dict has %d entries (expected: %d, loaded: %d, skipped: %d, duplicates: %d)",
        len(methods),
        expected_entries,
        loaded_count,
        skipped_no_metrics,
        total_duplicates,
    )

    return methods


def build_leaderboard_json(methods: Dict[str, Any]) -> Dict[str, Any]:
    """Build a leaderboard summary from methods.json-style data.

    Ranking uses the composite score (higher is better). Entries that fail
    hard reactor-scale constraints receive composite_score=0 and are moved
    to excluded_entries. Entries without any usable score are filtered out.

    Parameters
    ----------
    methods : dict[str, Any]
        Per-method data from :func:`build_methods_json`.

    Returns
    -------
    dict[str, Any]
        Keys: ``entries`` (ranked list), ``excluded_entries`` (failed
        constraints). Each entry has rank, composite_score, metrics, etc.
    """
    entries = []
    excluded_entries = []  # entries that fail constraints (kept for documentation)

    for method_key, md in methods.items():
        metrics = md.get("metrics", {})
        path = md.get("path", "")

        # Determine the composite score (preferred) or fall back to score_primary
        composite_score = md.get("composite_score")
        score_primary = md.get("score_primary")

        # If composite_score is missing, try to derive one from score_primary
        if composite_score is None:
            if "score_primary" in md:
                if score_primary is None:
                    logger.warning("Entry %s has score_primary=None, skipping", path)
                    continue
            else:
                # score_primary key doesn't exist, try fallback
                score_primary = metrics.get("final_squared_flux")
                if score_primary is None:
                    score_primary = metrics.get("final_normalized_squared_flux")
                if score_primary is None or not isinstance(score_primary, (int, float)):
                    logger.warning(
                        "Entry %s has no composite_score or score_primary (metrics keys: %s), skipping",
                        path,
                        list(metrics.keys())[:5],
                    )
                    continue

        entry = {
            "method_key": method_key,
            "method_version": md.get("method_version", ""),
            "composite_score": float(composite_score)
            if composite_score is not None
            else None,
            "score_primary": float(score_primary)
            if score_primary is not None
            else None,
            "run_date": md.get("run_date", ""),
            "contact": md.get("contact", ""),
            "hardware": md.get("hardware", ""),
            "path": md.get("path", ""),
            "metrics": metrics,
            "reactor_scale_metrics": md.get("reactor_scale_metrics", {}),
        }

        # Exclude entries that fail *hard* reactor-scale constraints
        all_violations = md.get(CONSTRAINT_VIOLATIONS_KEY, [])
        if not md.get("passes_constraints", True):
            entry[CONSTRAINT_VIOLATIONS_KEY] = all_violations
            excluded_entries.append(entry)
            continue
        if composite_score is not None and composite_score == 0.0:
            entry[CONSTRAINT_VIOLATIONS_KEY] = all_violations
            excluded_entries.append(entry)
            continue
        if all_violations:
            entry[CONSTRAINT_VIOLATIONS_KEY] = all_violations

        entries.append(entry)

    # Sort by composite_score descending (higher = better engineering margin).
    # Fall back to score_primary ascending for entries without composite_score.
    def _sort_key(e: Dict[str, Any]) -> Tuple[int, float]:
        cs = e.get("composite_score")
        if cs is not None:
            return (1, cs)  # group 1: has composite_score, higher is better
        sp = e.get("score_primary")
        if sp is not None:
            return (0, -sp)  # group 0: lower squared flux is better
        return (-1, 0)

    entries.sort(key=_sort_key, reverse=True)
    for i, e in enumerate(entries, start=1):
        e["rank"] = i

    logger.info(
        "Leaderboard: %d entries included, %d excluded (failed constraints)",
        len(entries),
        len(excluded_entries),
    )

    return {"entries": entries, "excluded_entries": excluded_entries}


# Metrics that should never appear in device-scale leaderboard tables.
_DEVICE_LEADERBOARD_EXCLUDE: set[str] = {
    "coil_order",
    "score_primary",
    "composite_score",
    "initial_B_field",
    "final_B_field",
    "target_B_field",
    "flux_threshold",
    "cc_threshold",
    "cs_threshold",
    "msc_threshold",
    "curvature_threshold",
    "force_threshold",
    "torque_threshold",
    "arclength_variation_threshold",
    "BdotN",
    "BdotN_over_B",
    "final_normalized_squared_flux",
    "coils_linked_to_surface",
    "arclength_variation",
    "final_order",
    "continuation_step",
    "fourier_continuation",
    "fourier_order",
    "N_turns_required",
    "iterations_used",
    "walltime_sec",
    "optimization_nfev",
    "optimization_njev",
    "optimization_success",
    "total_current_after",
    "total_current_before",
    "final_avg_max_coil_force",
    "final_avg_max_coil_torque",
    "quasisymmetry_average",
    "loss_fraction",
    "final_average_curvature",
}

# Default metric display order for surface leaderboards
_DEVICE_LEADERBOARD_DESIRED_ORDER: list[str] = [
    "num_coils",
    "fourier_continuation_orders",
    "final_squared_flux",
    "final_normalized_squared_flux",
    "avg_BdotN_over_B",
    "max_BdotN_over_B",
    "final_total_length",
    "final_arclength_variation",
    "final_min_cc_separation",
    "final_min_cs_separation",
    "final_mean_squared_curvature",
    "final_max_max_coil_force",
    "final_max_max_coil_torque",
    "final_linking_number",
    "optimization_time",
]
_DEVICE_LEADERBOARD_ALWAYS_INCLUDE: list[str] = [
    "num_coils",
    "fourier_continuation_orders",
]


def _collect_metric_keys_from_entries(
    entries: list[Dict[str, Any]],
    exclude: set[str] | None = None,
) -> set[str]:
    """Collect unique metric keys from entries, excluding specified keys.

    Parameters
    ----------
    entries : list[dict]
        Leaderboard entries (each with ``metrics`` dict).
    exclude : set[str], optional
        Keys to exclude. Defaults to _DEVICE_LEADERBOARD_EXCLUDE.

    Returns
    -------
    set[str]
        Unique metric keys present in entries.
    """
    exclude = exclude or _DEVICE_LEADERBOARD_EXCLUDE
    all_keys: set[str] = set()
    for entry in entries:
        for key in entry.get("metrics", {}).keys():
            if key not in exclude:
                all_keys.add(key)
    return all_keys


def _get_ordered_metrics_for_entries(
    entries: list[Dict[str, Any]],
    desired_order: list[str] | None = None,
    always_include: list[str] | None = None,
) -> list[str]:
    """Extract unique metric keys from entries and return in display order.

    Excludes keys in _DEVICE_LEADERBOARD_EXCLUDE. Orders by desired_order
    first, then appends remaining keys sorted alphabetically.

    Parameters
    ----------
    entries : list[dict]
        Leaderboard entries (each with ``metrics`` dict).
    desired_order : list[str], optional
        Preferred column order. Defaults to _DEVICE_LEADERBOARD_DESIRED_ORDER.
    always_include : list[str], optional
        Keys to always include even if no entry has them.

    Returns
    -------
    list[str]
        Ordered metric keys for display.
    """
    desired_order = desired_order or _DEVICE_LEADERBOARD_DESIRED_ORDER
    always_include = always_include or []

    all_keys = _collect_metric_keys_from_entries(entries)

    ordered: list[str] = []
    for key in desired_order:
        if key in all_keys or key in always_include:
            ordered.append(key)
    remaining = sorted(all_keys - set(ordered))
    ordered.extend(remaining)
    return ordered


def _ensure_flux_first(keys: list[str]) -> None:
    """Reorder keys in-place so flux metric is first (for leaderboard display).

    Prefers final_squared_flux, falls back to final_normalized_squared_flux.
    """
    if "final_squared_flux" in keys:
        keys.remove("final_squared_flux")
        keys.insert(0, "final_squared_flux")
    elif "final_normalized_squared_flux" in keys:
        keys.remove("final_normalized_squared_flux")
        keys.insert(0, "final_normalized_squared_flux")


def _get_all_metrics_from_entries(entries: list[Dict[str, Any]]) -> list[str]:
    """Get all unique metric keys from overall leaderboard entries.

    Excludes _DEVICE_LEADERBOARD_EXCLUDE keys. Puts final_squared_flux first.

    Parameters
    ----------
    entries : list[dict]
        Leaderboard entries (each with ``metrics`` dict).

    Returns
    -------
    list[str]
        Sorted metric keys with final_squared_flux first.
    """
    all_keys = _collect_metric_keys_from_entries(entries)
    sorted_keys = sorted(all_keys)
    _ensure_flux_first(sorted_keys)
    return sorted_keys


def build_surface_leaderboards(
    leaderboard: Dict[str, Any],
    submissions_root: Path,
    plasma_surfaces_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Group entries by plasma surface extracted from case.yaml or path structure.

    Expects path structure: submissions/surface/user/timestamp/results.json
    or all_files.zip. Entries within each surface are ranked by composite_score.

    Parameters
    ----------
    leaderboard : dict
        Leaderboard with ``entries`` list.
    submissions_root : Path
        Root of submissions directory (for path parsing).
    plasma_surfaces_dir : Path
        Unused; kept for API compatibility.

    Returns
    -------
    dict[str, dict]
        Mapping surface_name -> ``{"entries": [...]}`` with ranked entries.
    """
    entries = leaderboard.get("entries") or []
    surface_leaderboards: Dict[str, Dict[str, Any]] = {}

    for entry in entries:
        path_str = entry.get("path", "")
        if not path_str:
            continue

        if path_str.startswith("submissions/"):
            path_obj = (
                submissions_root.parent / path_str
                if submissions_root.parent
                else Path(path_str)
            )
        else:
            path_obj = Path(path_str)

        surface_name = _extract_surface_from_submission_path(path_obj, submissions_root)
        if surface_name == "unknown":
            continue

        if surface_name not in surface_leaderboards:
            surface_leaderboards[surface_name] = {"entries": []}

        surface_leaderboards[surface_name]["entries"].append(entry)

    def _surface_sort_key(e: Dict[str, Any]) -> Tuple[int, float]:
        cs = e.get("composite_score")
        if cs is not None:
            return (1, cs)
        sp = e.get("score_primary")
        if sp is not None:
            return (0, -sp)
        return (-1, 0)

    for surface, surf_data in surface_leaderboards.items():
        entries = surf_data["entries"]
        entries.sort(key=_surface_sort_key, reverse=True)
        for i, entry in enumerate(entries, start=1):
            entry["rank"] = i

    return surface_leaderboards
