"""
Coil optimization for StellCoilBench.

Provides modular coil optimization via simsopt, with support for augmented
Lagrangian, L-BFGS-B, and other scipy algorithms. Handles threshold scaling
by minor radius, constraint scaling for dimensionless objectives, Fourier
continuation, and post-processing integration.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .._mpl import MATPLOTLIB_AVAILABLE  # noqa: F401 - re-exported for backward compatibility
from ..config_scheme import CaseConfig, PostProcessingConfig
from ..mpi_utils import comm_world, is_mpi_enabled, is_proc0, proc0_print
from ._config_parsing import (
    _merge_post_processing_params,
    _prepare_optimization_config,
    _resolve_case_yaml_abs_path,
)
from ._optimization_dispatch import _dispatch_optimization_on_proc0
from ._post_opt_processing import _run_post_processing_after_optimization

logger = logging.getLogger(__name__)

# Re-exports for backward compatibility
from ._external_eval import (
    _check_coils_linked_to_surface as _check_coils_linked_to_surface,
    _compute_optimization_metrics as _compute_optimization_metrics,
    evaluate_external_coils as evaluate_external_coils,
)
from ._fourier_continuation import (
    _extend_coils_to_higher_order as _extend_coils_to_higher_order,
    optimize_coils_with_fourier_continuation as optimize_coils_with_fourier_continuation,
)
from ._optimization_setup import (
    _save_optimized_coils_and_compute_metrics as _save_optimized_coils_and_compute_metrics,
)
from ._iteration_output import (
    _format_verbose_iteration_output as _format_verbose_iteration_output,
)
from ._plotting import (
    _compute_surface_vtk_data as _compute_surface_vtk_data,
    _create_plotting_surface as _create_plotting_surface,
    _plot_bn_error_3d as _plot_bn_error_3d,
)
from ._results import (
    OptimizationOutcome as OptimizationOutcome,
    compute_total_current as compute_total_current,
    _build_cached_thresholds_dict as _build_cached_thresholds_dict,
    _build_optimization_results_dict as _build_optimization_results_dict,
    _compute_final_metrics as _compute_final_metrics,
    _merge_post_processing_into_results as _merge_post_processing_into_results,
    _save_results_and_compute_metrics as _save_results_and_compute_metrics,
    _save_vtk_outputs as _save_vtk_outputs,
)
from ._virtual_casing import VIRTUAL_CASING_AVAILABLE as VIRTUAL_CASING_AVAILABLE


class LinearPenalty:
    """
    Linear penalty function that implements max(objective - threshold, 0).

    Wraps a simsopt objective so that the effective value is zero below the
    threshold and (J - threshold) above. Used for l1_threshold options in
    coil_objective_terms (length, curvature, distances, etc.).
    """

    def __init__(self, objective: Any, threshold: float) -> None:
        self.objective = objective
        self.threshold = threshold
        self._parent = None
        self._children = []

    def __getattr__(self, name):
        """Delegate attribute access to underlying objective for simsopt compatibility."""
        if name in ["objective", "threshold", "_parent", "_children", "J", "dJ", "x"]:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self.objective, name)

    def J(self):
        """Return max(J - threshold, 0)"""
        J_val = self.objective.J()
        return max(J_val - self.threshold, 0.0)

    def dJ(self, **kwargs):
        """Return gradient: dJ/dx if J > threshold, else 0"""
        import numpy as np

        J_val = self.objective.J()
        grad = self.objective.dJ(**kwargs)
        if J_val > self.threshold:
            return grad
        else:
            if isinstance(grad, np.ndarray):
                return grad * 0.0
            elif hasattr(grad, "__mul__"):
                return grad * 0.0
            else:
                try:
                    x_arr = np.asarray(self.x)
                    return np.zeros_like(x_arr)
                except (AttributeError, TypeError, ValueError) as exc:
                    logger.debug("Failed to construct zero gradient array: %s", exc)
                    return 0.0

    def __add__(self, other):
        """Allow addition with other objectives for sum() compatibility"""
        if type(other) is type(self):
            combined = self.objective + other.objective
            return type(self)(combined, self.threshold)
        elif isinstance(other, (int, float)) and other == 0:
            return self
        return NotImplemented

    def __radd__(self, other):
        """Allow right addition for sum() compatibility"""
        if isinstance(other, (int, float)) and other == 0:
            return self
        return NotImplemented

    def __mul__(self, other):
        """Allow multiplication with Weight for compatibility"""
        from simsopt.objectives import Weight

        from ..constants import WEIGHT_CALCULATION_TOL

        if isinstance(other, Weight):
            weighted_obj = other * self.objective
            try:
                unweighted_J = self.objective.J()
                weighted_J = weighted_obj.J()
                if abs(unweighted_J) > WEIGHT_CALCULATION_TOL:
                    weight_val = weighted_J / unweighted_J
                else:
                    weight_val = 1.0
                scaled_threshold = weight_val * self.threshold
            except (AttributeError, ZeroDivisionError, TypeError, ValueError) as exc:
                logger.debug(
                    "Failed to scale threshold by weight, using unscaled: %s", exc
                )
                scaled_threshold = self.threshold
            return type(self)(weighted_obj, scaled_threshold)
        return NotImplemented

    def __rmul__(self, other):
        """Allow right multiplication with Weight"""
        return self.__mul__(other)

    def _add_child(self, child):
        """Add a child objective (simsopt compatibility)."""
        if child not in self._children:
            self._children.append(child)
            if hasattr(child, "_parent"):
                child._parent = self

    @property
    def x(self):
        """Get optimization variables"""
        return self.objective.x

    @x.setter
    def x(self, value):
        """Set optimization variables"""
        self.objective.x = value


def optimize_coils(
    case_path: Path,
    coils_out_path: Path,
    case_cfg: CaseConfig | None = None,
    output_dir: Path | None = None,
    surface_resolution: int = 32,
    skip_post_processing: bool = False,
    run_vmec: bool = False,
    run_simple: bool = False,
    plot_poincare: bool = False,
    plot_finite_build: bool = False,
    finite_build_width: Optional[float] = None,
    finite_build_height: Optional[float] = None,
    run_structural: bool = False,
    structural_E: Optional[float] = None,
    structural_nu: Optional[float] = None,
    compute_shape_gradient: bool = False,
) -> Dict[str, Any]:
    """
    Run a coil optimization for a given case using parameters from case.yaml,
    and write the resulting coils file to coils_out_path.
    """
    from simsopt import save
    from ..case_loader import load_case

    is_mpi_parallel = is_mpi_enabled()

    mpi_partition = None
    if is_mpi_parallel:
        try:
            from simsopt.util.mpi import MpiPartition

            mpi_partition = MpiPartition(ngroups=1)
        except ImportError as exc:
            logger.debug("MpiPartition import failed, MPI not available: %s", exc)
            mpi_partition = None
        proc0_print(f"Running with MPI: {comm_world.size} processes")
        proc0_print(
            "Optimization runs on rank 0; structural dJ and post-processing use all processes when applicable"
        )

    if case_cfg is None:
        case_cfg = load_case(case_path)

    pp_params = case_cfg.post_processing_params or {}
    cli_pp = PostProcessingConfig(
        run_vmec=run_vmec,
        run_simple=run_simple,
        plot_poincare=plot_poincare,
        plot_finite_build=plot_finite_build,
        finite_build_width=finite_build_width,
        finite_build_height=finite_build_height,
        run_structural=run_structural,
        structural_E=structural_E,
        structural_nu=structural_nu,
        compute_shape_gradient=compute_shape_gradient,
    )
    pp_flags = _merge_post_processing_params(pp_params, cli_pp)

    case_yaml_path_abs = _resolve_case_yaml_abs_path(case_path)

    config = _prepare_optimization_config(
        case_cfg,
        case_path,
        case_yaml_path_abs,
        coils_out_path,
        output_dir,
        surface_resolution,
    )
    surface = config["surface"]
    output_dir = config["output_dir"]

    if pp_flags.finite_build_width is None:
        from ..config_scheme import ARIES_CS_MINOR_RADIUS as _ARIES_A
        from ..path_utils import get_reference_radii
        from ._thresholds import _MAX_FINITE_BUILD_WIDTH, _MIN_FINITE_BUILD_WIDTH

        try:
            _, _minor = get_reference_radii(surface)
            _a0_fb = _ARIES_A / float(_minor)
            _fb_w = max(
                _MAX_FINITE_BUILD_WIDTH / _a0_fb,
                _MIN_FINITE_BUILD_WIDTH,
            )
        except (TypeError, ValueError, ZeroDivisionError, AttributeError):
            _fb_w = _MAX_FINITE_BUILD_WIDTH
        pp_flags = PostProcessingConfig(
            **{**vars(pp_flags), "finite_build_width": _fb_w}
        )
    if pp_flags.finite_build_height is None:
        pp_flags = PostProcessingConfig(
            **{**vars(pp_flags), "finite_build_height": pp_flags.finite_build_width}
        )

    original_cwd = Path.cwd()
    try:
        os.chdir(output_dir)

        coils = None
        results_dict = {}
        skip_post_processing_in_loop = skip_post_processing or is_mpi_parallel
        use_structural_mpi = (
            is_mpi_parallel
            and config.get("coil_objective_terms")
            and "structural_stress" in config.get("coil_objective_terms", {})
        )
        if use_structural_mpi or is_proc0():
            coils, results_dict = _dispatch_optimization_on_proc0(
                surface=surface,
                case_cfg=case_cfg,
                coil_params=config["coil_params"],
                optimizer_params=config["optimizer_params"],
                coil_objective_terms=config["coil_objective_terms"],
                threshold_kwargs=config["threshold_kwargs"],
                output_dir=output_dir,
                surface_resolution=surface_resolution,
                case_yaml_path_abs=case_yaml_path_abs,
                case_path=case_path,
                vc_target=config["vc_target"],
                vc_target_plot=config["vc_target_plot"],
                skip_post_processing_in_loop=skip_post_processing_in_loop,
                pp_flags=pp_flags,
            )

        if is_mpi_parallel:
            comm_world.Barrier()
    finally:
        os.chdir(original_cwd)

    if not str(coils_out_path).endswith(".json"):
        coils_out_path = coils_out_path.with_suffix(".json")

    abs_coils_path = (
        coils_out_path
        if coils_out_path.is_absolute()
        else (output_dir / coils_out_path.name)
    )
    if is_proc0():
        if coils is None:
            raise RuntimeError("Coil optimization failed: no coils were produced")
        save(coils, abs_coils_path)

    if is_mpi_parallel:
        comm_world.Barrier()

    if not skip_post_processing and is_mpi_parallel:
        coils_json_path = abs_coils_path
        if not coils_json_path.exists():
            coils_json_path = None

        post_processing_results = _run_post_processing_after_optimization(
            output_dir,
            surface,
            case_yaml_path_abs,
            pp_flags,
            mpi=mpi_partition,
            coils_json_path=coils_json_path,
        )
        if is_proc0() and post_processing_results:
            results_dict["post_processing"] = post_processing_results

    return results_dict
