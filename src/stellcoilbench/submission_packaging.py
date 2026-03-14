"""Submission packaging utilities for StellCoilBench.

Builds submission metadata, creates directory structure, writes results.json,
copies case YAML, and zips submission directories.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import typer

from .cli_helpers import _fmt_scalar, _write_json, _zip_submission_directory
from .constants import SUBMISSION_DATETIME_FMT
from .path_utils import (
    dump_yaml,
    get_surface_filename,
    load_yaml,
    surface_stem_from_filename,
)
from .version_utils import _get_version_info

from ._optional_imports import require_reactor_scale_compute

_compute_reactor_scale_metrics = require_reactor_scale_compute()


def _extract_surface_name(case_cfg: Any) -> str:
    """Extract the canonical surface name from a loaded CaseConfig or raw dict.

    Parameters
    ----------
    case_cfg : CaseConfig | dict
        Loaded case configuration object (with ``.surface_params`` attribute)
        or a raw configuration dictionary with a ``surface_params`` key.

    Returns
    -------
    str
        Normalised surface stem name.

    Raises
    ------
    ValueError
        If ``surface_params.surface`` is missing.
    """
    surface_file = get_surface_filename(case_cfg)
    if not surface_file:
        raise ValueError("case.yaml must specify surface_params.surface")
    return surface_stem_from_filename(surface_file)


def _prepare_submission_dir(
    submissions_dir: Path,
    surface_name: str,
    username: str,
    case_path: Path,
) -> tuple[Path, str]:
    """Create the submission directory and return (submission_dir, datetime_str).

    Parameters
    ----------
    submissions_dir : Path
        Root submissions directory.
    surface_name : str
        Plasma surface canonical name.
    username : str
        GitHub username (or fallback).
    case_path : Path
        Path to the case YAML file.

    Returns
    -------
    tuple[Path, str]
        ``(submission_dir, datetime_str)``
    """
    now = datetime.now()
    datetime_str = now.strftime(SUBMISSION_DATETIME_FMT)
    case_name = case_path.stem if case_path.suffix == ".yaml" else case_path.name
    submission_dir = (
        submissions_dir / surface_name / username / case_name / datetime_str
    )
    submission_dir.mkdir(parents=True, exist_ok=True)
    return submission_dir, datetime_str


def _build_submission_dict(
    metrics: Dict[str, Any],
    case_cfg: Any,
    *,
    run_date: str | None = None,
    contact: str | None = None,
    hardware: str | None = None,
    sensitivity_results: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a submission results dictionary.

    Parameters
    ----------
    metrics : dict
        Optimisation result metrics.
    case_cfg : CaseConfig
        Loaded case configuration (used for reactor scaling).
    run_date : str, optional
        ISO-formatted run date; defaults to now.
    contact : str, optional
        Contact/user name.
    hardware : str, optional
        Hardware description.
    sensitivity_results : dict, optional
        Sensitivity analysis results.

    Returns
    -------
    dict
        Submission dictionary ready for JSON serialisation.
    """
    if run_date is None:
        run_date = datetime.now().isoformat()
    version_info = _get_version_info()
    reactor_scale_metrics = _compute_reactor_scale_metrics(metrics, case_cfg)

    metadata: Dict[str, Any] = {
        "run_date": run_date,
    }
    if contact is not None:
        metadata["contact"] = contact
    if hardware is not None:
        metadata["hardware"] = hardware

    submission: Dict[str, Any] = {
        "metadata": metadata,
        "version_info": version_info,
        "metrics": metrics,
        "reactor_scale_metrics": reactor_scale_metrics,
    }
    if sensitivity_results is not None:
        submission["sensitivity"] = sensitivity_results
    return submission


# --- Submission summary printing (used by _package_submission) ---

_METRICS_SUPPRESS: set[str] = {
    "timing",
    "_cached_thresholds",
    "continuation_results",
    "total_current_after",
    "total_current_before",
    "optimization_message",
    "optimization_nfev",
    "optimization_njev",
    "optimization_success",
    "optimization_time",
    "initial_B_field",
    "final_B_field",
    "fourier_order",
    "continuation_step",
    "final_order",
}
"""Metric keys hidden from the console submission summary."""

_PER_COIL_SUPPRESS: set[str] = {
    "final_max_torque_per_coil",
    "final_max_force_per_coil",
    "final_current_per_coil",
}
"""Per-coil array keys hidden from the main summary (shown in sub-sections)."""

_SUBSET_PER_COIL_SUPPRESS: set[str] = _PER_COIL_SUPPRESS | {"max_force", "max_torque"}

_THRESHOLD_SUFFIX: str = "_threshold"

_COMBINED_GROUPS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (
        ("BdotN", "BdotN_over_B", "avg_BdotN_over_B", "max_BdotN_over_B"),
        ("B·n", "B·n/B", "avg(B·n/B)", "max(B·n/B)"),
    ),
    (
        (
            "final_max_curvature",
            "final_average_curvature",
            "final_mean_squared_curvature",
        ),
        ("κ_max", "κ̄", "MSC"),
    ),
    (
        ("final_max_max_coil_force", "final_avg_max_coil_force"),
        ("F_max", "F̄"),
    ),
    (
        ("final_max_max_coil_torque", "final_avg_max_coil_torque"),
        ("τ_max", "τ̄"),
    ),
    (
        (
            "final_arclength_variation",
            "final_min_cc_separation",
            "final_min_cs_separation",
        ),
        ("√Var", "d_cc", "d_cs"),
    ),
    (
        ("final_squared_flux",),
        ("f_B",),
    ),
]
"""Metric groups printed as combined single-line rows in the summary."""

_ALL_COMBINED_KEYS: set[str] = {k for keys, _ in _COMBINED_GROUPS for k in keys} | {
    "final_length_per_coil",
    "final_total_length",
}


def _should_suppress(k: str) -> bool:
    """Return True if metric *k* should be hidden from the main summary."""
    return (
        k in _METRICS_SUPPRESS
        or k in _PER_COIL_SUPPRESS
        or k.endswith(_THRESHOLD_SUFFIX)
    )


def _print_timing_summary(metrics: Dict[str, Any], submission: Dict[str, Any]) -> None:
    """Print the coil optimization timing breakdown."""
    timing = metrics.get("timing") or submission.get("timing")
    if not timing:
        return
    coil_opt_keys = [
        "coil_initialization",
        "biotsavart_setup",
        "objective_setup",
        "coil_optimization",
        "save_and_metrics",
    ]
    total_coil_opt = sum(timing.get(k, 0) for k in coil_opt_keys)
    if total_coil_opt > 0:
        typer.echo("  Timing:")
        for key in coil_opt_keys:
            if key in timing:
                typer.echo(f"    {key}: {timing[key]:.2f}s")
        typer.echo(f"    {'─' * 30}")
        typer.echo(f"    Total coil optimization: {total_coil_opt:.2f}s")
        typer.echo("")


def _print_combined_metrics_group(
    metrics_to_print: Dict[str, Any],
    groups: list[tuple[tuple[str, ...], tuple[str, ...]]],
) -> None:
    """Print groups of related metrics as compact one-liner rows."""
    for keys, labels in groups:
        parts: list[str] = []
        for key, label in zip(keys, labels):
            if key in metrics_to_print and isinstance(
                metrics_to_print[key], (int, float)
            ):
                parts.append(f"{label}: {_fmt_scalar(metrics_to_print[key])}")
        if parts:
            typer.echo("    " + " | ".join(parts))


def _print_metrics_dict(d: dict, per_coil_suppress: set) -> None:
    """Print a dict of metrics, skipping suppressed per-coil keys."""
    for k, v in sorted(d.items()):
        if k in per_coil_suppress:
            continue
        if isinstance(v, (int, float)):
            typer.echo(f"    {k}: {_fmt_scalar(v)}")
        else:
            typer.echo(f"    {k}: {v}")


def _print_submission_summary(submission: Dict[str, Any]) -> None:
    """Print a clearly formatted summary of the submission results.

    Parameters
    ----------
    submission : dict[str, Any]
        Full submission dictionary containing ``metadata``, ``metrics``,
        and optionally ``timing``.
    """
    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("  OPTIMIZATION RESULTS SUMMARY")
    typer.echo("=" * 60)
    typer.echo("")
    meta = submission.get("metadata", {})
    typer.echo("  Metadata:")
    for k, v in meta.items():
        typer.echo(f"    {k}: {v}")
    typer.echo("")
    metrics = submission.get("metrics", {})

    metrics_to_print = {k: v for k, v in metrics.items() if not _should_suppress(k)}

    if metrics_to_print:
        typer.echo("  Metrics:")
        _print_combined_metrics_group(metrics_to_print, _COMBINED_GROUPS)
        l_total = metrics.get("final_total_length")
        l_per_coil = metrics.get("final_length_per_coil")
        length_parts: list[str] = []
        if isinstance(l_total, (int, float)):
            length_parts.append(f"L: {_fmt_scalar(l_total)}")
        if isinstance(l_per_coil, (list, tuple)) and l_per_coil:
            pc = [_fmt_scalar(x) for x in l_per_coil if isinstance(x, (int, float))]
            if pc:
                length_parts.append("L_per_coil: " + ", ".join(pc))
        if length_parts:
            typer.echo("    " + " | ".join(length_parts))
        for k, v in sorted(metrics_to_print.items()):
            if k in _ALL_COMBINED_KEYS:
                continue
            if k == "lagrange_multipliers" and v is None:
                continue
            if isinstance(v, (int, float)):
                typer.echo(f"    {k}: {_fmt_scalar(v)}")
            else:
                typer.echo(f"    {k}: {v}")
        typer.echo("")

    tf_metrics = metrics.get("tf_metrics")
    if isinstance(tf_metrics, dict) and tf_metrics:
        typer.echo("  TF metrics:")
        _print_metrics_dict(tf_metrics, _SUBSET_PER_COIL_SUPPRESS)
        typer.echo("")
    iters = metrics.get("iterations_used")
    if isinstance(iters, (int, float)) and iters <= 2:
        msg = metrics.get("optimization_message")
        nfev = metrics.get("optimization_nfev")
        njev = metrics.get("optimization_njev")
        if msg or nfev is not None or njev is not None:
            typer.echo("  Optimization diagnostics (early exit):")
            if msg:
                typer.echo(f"    message: {msg}")
            if nfev is not None:
                typer.echo(f"    nfev: {nfev}")
            if njev is not None:
                typer.echo(f"    njev: {njev}")
            typer.echo("")
    _print_timing_summary(metrics, submission)
    typer.echo("=" * 60)
    typer.echo("")


def _remove_pre_zip_artifacts(submission_dir: Path) -> None:
    """Remove VMEC and QFM outputs from submission dir before creating all_files.zip.

    Deletes wout*, input.* (VMEC), and qfm_surface.vts (QFM) to reduce zip size.

    Parameters
    ----------
    submission_dir : Path
        Submission directory to clean.
    """
    patterns = [
        "wout*",
        "input.*",
        "qfm_surface.vts",
    ]
    for pattern in patterns:
        for p in submission_dir.rglob(pattern):
            if p.is_file():
                try:
                    p.unlink()
                except OSError:
                    pass


def _package_submission(
    *,
    submission: Dict[str, Any],
    submission_dir: Path,
    case_path: Path,
) -> None:
    """Write results.json, copy case.yaml, and zip the submission directory.

    Performs four steps:

    1. Writes ``results.json`` and prints the submission summary.
    2. Copies the case YAML into the submission directory, annotating it
       with a ``source_case_file`` field for traceability.
    3. Removes VMEC (wout, input) and QFM (qfm_surface.vts) artifacts.
    4. Zips the directory via :func:`_zip_submission_directory`.

    Parameters
    ----------
    submission : dict[str, Any]
        Full submission dictionary (metadata + metrics).
    submission_dir : Path
        Target directory for all submission artefacts.
    case_path : Path
        Original case YAML path (file or directory).
    """
    submission_path = submission_dir / "results.json"
    _write_json(submission_path, submission)
    _print_submission_summary(submission)

    case_yaml_path = case_path if case_path.is_file() else (case_path / "case.yaml")
    if case_yaml_path.exists() and case_yaml_path.is_file():
        submission_case_yaml = submission_dir / "case.yaml"
        case_data = load_yaml(path=case_yaml_path)
        repo_root = Path.cwd()
        try:
            source_case_file = str(
                case_yaml_path.resolve().relative_to(repo_root.resolve())
            )
        except ValueError:
            source_case_file = str(case_yaml_path.resolve())
        case_data["source_case_file"] = source_case_file
        dump_yaml(case_data, path=submission_case_yaml)

    _remove_pre_zip_artifacts(submission_dir)
    _zip_submission_directory(submission_dir)
