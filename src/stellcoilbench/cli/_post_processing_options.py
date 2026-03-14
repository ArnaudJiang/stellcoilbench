"""Shared post-processing CLI options for submit-case and post-process.

Provides reusable typer.Option objects and a builder function to reduce
duplication between submit_run and post_process commands.
"""

from __future__ import annotations

from typing import Optional

import typer

from ..config_scheme import PostProcessingConfig

from . import _shared

# Reusable Option objects - use as default values in command parameters
ALL_POST_PROCESSING_OPTION = typer.Option(
    False,
    "--all-post-processing",
    help="Enable all post-processing (VMEC, SIMPLE, Poincaré, Boozer, finite-build, structural, shape gradient).",
)
RUN_VMEC_OPTION = typer.Option(
    False,
    "--run-vmec/--no-vmec",
    help="Run VMEC equilibrium and quasisymmetry analysis.",
)
RUN_SIMPLE_OPTION = typer.Option(
    False,
    "--run-simple/--no-simple",
    help="Run SIMPLE fast particle tracing (requires VMEC).",
)
PLOT_POINCARE_OPTION = typer.Option(
    False, "--plot-poincare/--no-plot-poincare", help="Generate Poincaré plot."
)
PLOT_BOOZER_OPTION = typer.Option(
    False,
    "--plot-boozer/--no-plot-boozer",
    help="Generate Boozer surface plot (requires VMEC).",
)
PLOT_FINITE_BUILD_OPTION = typer.Option(
    False,
    "--plot-finite-build/--no-plot-finite-build",
    help="Generate finite-build coil geometry VTK.",
)
FINITE_BUILD_WIDTH_OPTION = typer.Option(
    None, "--finite-build-width", help="Cross-section width [m] for finite-build coils."
)
FINITE_BUILD_HEIGHT_OPTION = typer.Option(
    None,
    "--finite-build-height",
    help="Cross-section height [m] for finite-build coils.",
)
RUN_STRUCTURAL_OPTION = typer.Option(
    False,
    "--run-structural/--no-structural",
    help="Run FEM structural analysis on finite-build coils.",
)
EXPORT_STRUCTURAL_FULL_COIL_SET_OPTION = typer.Option(
    False,
    "--export-structural-full-coil-set/--no-export-structural-full-coil-set",
    help="Export structural_results_full.vtk with full coil set (symmetry copies).",
)
STRUCTURAL_E_OPTION = typer.Option(
    None, "--structural-E", help="Young's modulus [Pa] for structural analysis."
)
STRUCTURAL_NU_OPTION = typer.Option(
    None, "--structural-nu", help="Poisson ratio for structural analysis."
)
COMPUTE_SHAPE_GRADIENT_OPTION = typer.Option(
    False,
    "--compute-shape-gradient/--no-shape-gradient",
    help="Compute per-coil shape gradients.",
)
HELICITY_M_OPTION = typer.Option(
    1, "--helicity-m", help="Poloidal mode number for quasisymmetry."
)
HELICITY_N_OPTION = typer.Option(
    0, "--helicity-n", help="Toroidal mode number for quasisymmetry."
)
NS_OPTION = typer.Option(
    50, "--ns", help="Number of radial surfaces for VMEC/quasisymmetry."
)
NFIELDLINES_OPTION = typer.Option(
    20, "--nfieldlines", help="Number of fieldlines for Poincaré plot."
)


def build_post_processing_config(
    all_post_processing: bool,
    run_vmec: bool,
    run_simple: bool,
    plot_poincare: bool,
    plot_boozer: bool,
    plot_finite_build: bool,
    finite_build_width: Optional[float],
    finite_build_height: Optional[float],
    run_structural: bool,
    structural_E: Optional[float],
    structural_nu: Optional[float],
    compute_shape_gradient: bool,
    export_structural_full_coil_set: bool = False,
    helicity_m: int = 1,
    helicity_n: int = 0,
    ns: int = 50,
    nfieldlines: int = 20,
) -> PostProcessingConfig:
    """Apply --all-post-processing logic and build PostProcessingConfig from CLI options."""
    (
        run_vmec,
        run_simple,
        plot_poincare,
        plot_boozer,
        plot_finite_build,
        run_structural,
        compute_shape_gradient,
    ) = _shared.apply_all_post_processing_flags(
        all_post_processing,
        run_vmec=run_vmec,
        run_simple=run_simple,
        plot_poincare=plot_poincare,
        plot_boozer=plot_boozer,
        plot_finite_build=plot_finite_build,
        run_structural=run_structural,
        compute_shape_gradient=compute_shape_gradient,
    )
    return PostProcessingConfig.from_cli_options(
        run_vmec=run_vmec,
        run_simple=run_simple,
        plot_poincare=plot_poincare,
        plot_boozer=plot_boozer,
        plot_finite_build=plot_finite_build,
        finite_build_width=finite_build_width,
        finite_build_height=finite_build_height,
        run_structural=run_structural,
        structural_E=structural_E,
        structural_nu=structural_nu,
        export_structural_full_coil_set=export_structural_full_coil_set,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ns=ns,
        nfieldlines=nfieldlines,
        compute_shape_gradient=compute_shape_gradient,
    )
