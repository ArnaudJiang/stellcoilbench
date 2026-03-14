"""Coil optimization dispatch: route to Fourier continuation or standard loop.

Provides _dispatch_optimization_on_proc0, which selects the optimization path
based on case configuration: modular Fourier-continuation or standard modular
loop. Called from optimize_coils on MPI rank 0 only; other ranks wait at
barrier until optimization completes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from simsopt.geo import SurfaceRZFourier

from ..config_scheme import CaseConfig, PostProcessingConfig
from ._scipy_optimizer import _filter_optimizer_kwargs
from ._fourier_continuation import optimize_coils_with_fourier_continuation
from ._optimization_loop import optimize_coils_loop
from ._optimization_loop import _get_regularization_circ

regularization_circ = _get_regularization_circ()


def _dispatch_optimization_on_proc0(
    surface: "SurfaceRZFourier",
    case_cfg: CaseConfig,
    coil_params: dict[str, Any],
    optimizer_params: dict[str, Any],
    coil_objective_terms: dict[str, Any] | None,
    threshold_kwargs: dict[str, Any],
    output_dir: Path,
    surface_resolution: int,
    case_yaml_path_abs: Path | None,
    case_path: Path,
    vc_target: np.ndarray | None,
    vc_target_plot: np.ndarray | None,
    skip_post_processing_in_loop: bool,
    pp_flags: PostProcessingConfig,
) -> tuple[list | None, dict[str, Any]]:
    """Run coil optimization dispatch on rank 0.

    Selects between modular Fourier-continuation or standard loop based on
    *case_cfg*.

    Parameters
    ----------
    surface : SurfaceRZFourier
        Plasma boundary surface.
    case_cfg : CaseConfig
        Full case configuration from case.yaml.
    coil_params : dict
        Parsed coil parameters (ncoils, order, target_B, etc.).
    optimizer_params : dict
        Parsed optimizer parameters (algorithm, max_iterations, etc.).
    coil_objective_terms : dict | None
        Objective term config (length, curvature, flux thresholds, etc.).
    threshold_kwargs : dict
        Threshold overrides and scaling options.
    output_dir : Path
        Resolved output directory.
    surface_resolution : int
        Surface quadrature (nphi = ntheta).
    case_yaml_path_abs, case_path : Path
        Case YAML paths (absolute and as-provided).
    vc_target, vc_target_plot : ndarray | None
        Virtual-casing targets for SquaredFlux (optional).
    skip_post_processing_in_loop : bool
        Whether to skip QFM/VMEC/Poincare inside the optimization loop.
    pp_flags : PostProcessingConfig
        Bundled post-processing flags.

    Returns
    -------
    tuple[list | None, dict]
        (coils, results_dict). Coils may be None if optimization failed.
    """
    algorithm_options = optimizer_params.pop("algorithm_options", {})
    effective_case_path = (
        case_yaml_path_abs
        if case_yaml_path_abs and case_yaml_path_abs.exists()
        else case_path
    )

    opt_kwargs = dict(threshold_kwargs)
    if pp_flags.finite_build_width is not None:
        opt_kwargs["finite_build_width"] = pp_flags.finite_build_width

    fourier_continuation = case_cfg.fourier_continuation

    if fourier_continuation and fourier_continuation.get("enabled", False):
        fourier_orders = fourier_continuation.get(
            "orders", [coil_params.get("order", 16)]
        )
        if not isinstance(fourier_orders, list) or not all(
            isinstance(o, int) for o in fourier_orders
        ):
            raise ValueError("fourier_continuation.orders must be a list of integers")

        coils, results_dict = optimize_coils_with_fourier_continuation(
            surface,
            fourier_orders=fourier_orders,
            target_B=coil_params.get("target_B", 5.7),
            out_dir=str(output_dir),
            max_iterations=optimizer_params.get("max_iterations", 30),
            ncoils=coil_params.get("ncoils", 4),
            verbose=optimizer_params.get("verbose", True),
            regularization=regularization_circ
            if regularization_circ is not None
            else lambda x: None,
            coil_objective_terms=coil_objective_terms,
            surface_resolution=surface_resolution,
            algorithm_options=algorithm_options,
            case_path=effective_case_path,
            vc_target=vc_target,
            vc_target_plot=vc_target_plot,
            skip_post_processing=skip_post_processing_in_loop,
            pp_flags=pp_flags,
            **_filter_optimizer_kwargs(optimizer_params),
            **opt_kwargs,
        )
    else:
        coils, results_dict = optimize_coils_loop(
            surface,
            **coil_params,
            **optimizer_params,
            out_dir=str(output_dir),
            coil_objective_terms=coil_objective_terms,
            surface_resolution=surface_resolution,
            algorithm_options=algorithm_options,
            case_path=effective_case_path,
            vc_target=vc_target,
            vc_target_plot=vc_target_plot,
            skip_post_processing=skip_post_processing_in_loop,
            pp_flags=pp_flags,
            **opt_kwargs,
        )

    return coils, results_dict
