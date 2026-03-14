"""post-process CLI command implementation.

Runs post-processing on previously optimized coils (coils.json or
biot_savart_optimized.json). Supports VMEC equilibrium, Poincaré plots,
Boozer surface analysis, finite-build VTK, structural stress, and SIMPLE
particle tracing. Options can be enabled individually or via
--all-post-processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..cli_helpers import _cli_error

from ._post_processing_options import (
    ALL_POST_PROCESSING_OPTION,
    COMPUTE_SHAPE_GRADIENT_OPTION,
    EXPORT_STRUCTURAL_FULL_COIL_SET_OPTION,
    FINITE_BUILD_HEIGHT_OPTION,
    FINITE_BUILD_WIDTH_OPTION,
    HELICITY_M_OPTION,
    HELICITY_N_OPTION,
    NFIELDLINES_OPTION,
    NS_OPTION,
    PLOT_BOOZER_OPTION,
    PLOT_FINITE_BUILD_OPTION,
    PLOT_POINCARE_OPTION,
    RUN_SIMPLE_OPTION,
    RUN_STRUCTURAL_OPTION,
    RUN_VMEC_OPTION,
    STRUCTURAL_E_OPTION,
    STRUCTURAL_NU_OPTION,
    build_post_processing_config,
)


def post_process(
    coils_json: Path = typer.Argument(
        ...,
        help="Path to coils JSON file (e.g., biot_savart_optimized.json or coils.json).",
    ),
    output_dir: Path = typer.Option(
        Path("post_processing_output"),
        "--output-dir",
        "-o",
        help="Directory where post-processing results will be saved.",
    ),
    case_yaml: Optional[Path] = typer.Option(
        None, "--case-yaml", help="Path to case.yaml file."
    ),
    plasma_surfaces_dir: Optional[Path] = typer.Option(
        None, "--plasma-surfaces-dir", help="Directory containing plasma surface files."
    ),
    all_post_processing: bool = ALL_POST_PROCESSING_OPTION,
    run_vmec: bool = RUN_VMEC_OPTION,
    helicity_m: int = HELICITY_M_OPTION,
    helicity_n: int = HELICITY_N_OPTION,
    ns: int = NS_OPTION,
    plot_boozer: bool = PLOT_BOOZER_OPTION,
    plot_poincare: bool = PLOT_POINCARE_OPTION,
    nfieldlines: int = NFIELDLINES_OPTION,
    run_simple: bool = RUN_SIMPLE_OPTION,
    plot_finite_build: bool = PLOT_FINITE_BUILD_OPTION,
    finite_build_width: Optional[float] = FINITE_BUILD_WIDTH_OPTION,
    finite_build_height: Optional[float] = FINITE_BUILD_HEIGHT_OPTION,
    run_structural: bool = RUN_STRUCTURAL_OPTION,
    structural_E: Optional[float] = STRUCTURAL_E_OPTION,
    structural_nu: Optional[float] = STRUCTURAL_NU_OPTION,
    compute_shape_gradient: bool = COMPUTE_SHAPE_GRADIENT_OPTION,
    export_structural_full_coil_set: bool = EXPORT_STRUCTURAL_FULL_COIL_SET_OPTION,
) -> None:
    """Run post-processing on optimized coil results."""
    from ..post_processing import run_post_processing

    cfg = build_post_processing_config(
        all_post_processing,
        run_vmec,
        run_simple,
        plot_poincare,
        plot_boozer,
        plot_finite_build,
        finite_build_width,
        finite_build_height,
        run_structural,
        structural_E,
        structural_nu,
        compute_shape_gradient,
        export_structural_full_coil_set=export_structural_full_coil_set,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ns=ns,
        nfieldlines=nfieldlines,
    )

    typer.echo(f"Running post-processing on {coils_json}")
    typer.echo(f"Output directory: {output_dir}")

    try:
        pp_kwargs = cfg.to_run_post_processing_kwargs(
            coils_json_path=coils_json,
            output_dir=output_dir,
            case_yaml_path=case_yaml,
            plasma_surfaces_dir=plasma_surfaces_dir,
        )
        results = run_post_processing(**pp_kwargs)
        typer.echo("\nPost-processing complete!")
        typer.echo(f"Results saved to: {output_dir}")
        if "quasisymmetry_average" in results:
            typer.echo(
                f"Average quasisymmetry error: {results['quasisymmetry_average']:.2e}"
            )
    except (OSError, RuntimeError, ValueError) as e:
        _cli_error(f"Post-processing failed: {e}")


def register(app: typer.Typer) -> None:
    app.command("post-process")(post_process)
