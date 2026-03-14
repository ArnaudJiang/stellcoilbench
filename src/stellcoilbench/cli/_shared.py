"""Shared CLI helpers used by multiple subcommands.

Provides run_sensitivity_if_configured, apply_all_post_processing_flags,
and other utilities that coordinate between submit-case, post-process,
and sensitivity flows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import typer


def run_sensitivity_if_configured(
    *,
    run_sensitivity: bool,
    coils_out_path: Path,
    case_path: Path,
    correlation_length: float,
    n_samples: int,
    output_dir: Path,
    n_vtk: int,
    metrics: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Optionally run coil sensitivity analysis and return the results."""
    if run_sensitivity is not True:
        return None

    import time as _time_sens

    typer.echo("Running coil sensitivity analysis...")
    t0 = _time_sens.time()
    try:
        from ..sensitivity import run_sensitivity_analysis

        sens_result = run_sensitivity_analysis(
            coils_json_path=coils_out_path,
            case_yaml_path=case_path
            if case_path.is_file()
            else (case_path / "case.yaml"),
            correlation_length_m=correlation_length,
            n_samples=n_samples,
            output_dir=output_dir,
            make_plot=True,
            n_vtk_samples=n_vtk,
        )
        elapsed = _time_sens.time() - t0
        sensitivity_results = sens_result.to_dict()
        metrics["critical_sigma_mm"] = sens_result.critical_sigma_m * 1e3
        typer.echo(
            "  sigma* is the coil positional tolerance: perturbations below "
            "this level keep f_B within the specified degradation factor."
        )
        typer.echo(f"  Critical sigma*: {sens_result.critical_sigma_m * 1e3:.2f} mm")
        typer.echo(f"  Sensitivity analysis took {elapsed:.1f} s")
        return sensitivity_results
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Warning: Sensitivity analysis failed: {exc}", err=True)
        return None


def apply_all_post_processing_flags(
    all_post_processing: bool,
    *,
    run_vmec: bool,
    run_simple: bool,
    plot_poincare: bool,
    plot_boozer: bool,
    plot_finite_build: bool,
    run_structural: bool,
    compute_shape_gradient: bool = False,
) -> tuple[bool, bool, bool, bool, bool, bool, bool]:
    """Enable all post-processing flags when *all_post_processing* is set."""
    if all_post_processing:
        return True, True, True, True, True, True, True
    return (
        run_vmec,
        run_simple,
        plot_poincare,
        plot_boozer,
        plot_finite_build,
        run_structural,
        compute_shape_gradient,
    )
