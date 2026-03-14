"""sensitivity CLI command implementation.

Runs stochastic coil sensitivity analysis on optimized coils. Perturbs
coil positions with spatially correlated noise, evaluates B·n degradation,
and reports the critical sigma* (positional tolerance) below which
performance stays within a specified degradation factor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..cli_helpers import _cli_error


def sensitivity_cmd(
    coils_json: Path = typer.Argument(
        ...,
        help="Path to coils JSON file (e.g., coils.json or biot_savart_optimized.json).",
    ),
    case_yaml: Optional[Path] = typer.Option(
        None, "--case-yaml", help="Path to case.yaml. Auto-detected if not provided."
    ),
    plasma_surfaces_dir: Optional[Path] = typer.Option(
        None, "--plasma-surfaces-dir", help="Directory containing plasma surface files."
    ),
    output_dir: Path = typer.Option(
        Path("sensitivity_output"),
        "--output-dir",
        "-o",
        help="Directory where results will be saved.",
    ),
    correlation_length: float = typer.Option(1.0, "--correlation-length"),
    n_samples: int = typer.Option(100, "--n-samples", "-n"),
    factor: float = typer.Option(2.0, "--factor"),
    percentile: float = typer.Option(95.0, "--percentile"),
    sigma_min: float = typer.Option(1e-5, "--sigma-min"),
    sigma_max: float = typer.Option(0.05, "--sigma-max"),
    seed: int = typer.Option(42, "--seed"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
    n_sweep: int = typer.Option(8, "--n-sweep"),
    n_vtk_samples: int = typer.Option(0, "--n-vtk-samples"),
) -> None:
    """Run coil sensitivity analysis via stochastic perturbation."""
    import time as _time_sens

    from ..sensitivity import run_sensitivity_analysis

    typer.echo(f"Running sensitivity analysis on {coils_json}")
    typer.echo(f"  correlation length = {correlation_length:.2f} m")
    typer.echo(
        f"  n_samples = {n_samples}, factor = {factor}, percentile = {percentile}"
    )

    try:
        t0 = _time_sens.time()
        result = run_sensitivity_analysis(
            coils_json_path=coils_json,
            case_yaml_path=case_yaml,
            plasma_surfaces_dir=plasma_surfaces_dir,
            correlation_length_m=correlation_length,
            n_samples=n_samples,
            factor=factor,
            percentile=percentile,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            seed=seed,
            output_dir=output_dir,
            make_plot=plot,
            n_sweep=n_sweep,
            n_vtk_samples=n_vtk_samples,
        )
        elapsed = _time_sens.time() - t0
        typer.echo("\nSensitivity analysis complete!")
        typer.echo(f"  Nominal f_B:    {result.nominal_fb:.6e}")
        typer.echo(
            "  sigma* is the coil positional tolerance: perturbations below "
            "this level keep f_B within the specified degradation factor."
        )
        typer.echo(f"  Critical sigma*: {result.critical_sigma_m * 1e3:.2f} mm")
        typer.echo(f"  Time elapsed:   {elapsed:.1f} s")
        typer.echo(f"  Results saved to: {output_dir}")
    except (OSError, RuntimeError, ValueError) as e:
        _cli_error(f"Sensitivity analysis failed: {e}")


def register(app: typer.Typer) -> None:
    app.command("sensitivity")(sensitivity_cmd)
