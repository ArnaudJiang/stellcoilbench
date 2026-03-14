"""
Coil optimization for StellCoilBench.

Provides modular coil optimization via simsopt, with support for augmented
Lagrangian, L-BFGS-B, and other scipy algorithms.
"""

from __future__ import annotations

from .._mpl import MATPLOTLIB_AVAILABLE
from ..path_utils import load_surface_with_range
from ._config_parsing import DEFAULT_COIL_OBJECTIVE_TERMS
from simsopt.geo import SurfaceRZFourier
from .optimization import (
    VIRTUAL_CASING_AVAILABLE,
    LinearPenalty,
    evaluate_external_coils,
    optimize_coils,
    optimize_coils_with_fourier_continuation,
    proc0_print,
)
from ._optimization_loop import initialize_coils_loop, optimize_coils_loop
from ._ci_utils import (
    _is_ci_running,
    _nullcontext,
    _redirect_verbose_to_file,
    _zip_output_files,
)
from ._scipy_optimizer import (
    _get_scipy_algorithm_options,
    _parse_optimizer_config,
    _validate_algorithm_options,
)
from ._fourier_continuation import _extend_coils_to_higher_order
from ._optimization_loop import _optimize_coils_loop_impl
from ._plotting import _plot_bn_error_3d
from ._adaptive_search import (
    _adaptive_R0_R1_search,
    _coils_via_symmetries_compat,
    _make_base_currents,
)
from ._results import (
    OptimizationOutcome,
    compute_total_current,
    _build_cached_thresholds_dict,
    _build_optimization_results_dict,
    _compute_final_metrics,
    _merge_post_processing_into_results,
    _save_results_and_compute_metrics,
    _save_vtk_outputs,
)
from ._virtual_casing import _setup_virtual_casing
from ._iteration_output import _format_verbose_iteration_output

from .._optional_imports import optional_import

StructuralStressObjective = optional_import(
    "stellcoilbench.coil_optimization._structural_objective",
    "StructuralStressObjective",
    fallback=None,
)

__all__ = [
    "StructuralStressObjective",
    "DEFAULT_COIL_OBJECTIVE_TERMS",
    "MATPLOTLIB_AVAILABLE",
    "VIRTUAL_CASING_AVAILABLE",
    "OptimizationOutcome",
    "compute_total_current",
    "LinearPenalty",
    "SurfaceRZFourier",
    "evaluate_external_coils",
    "initialize_coils_loop",
    "load_surface_with_range",
    "optimize_coils",
    "optimize_coils_loop",
    "optimize_coils_with_fourier_continuation",
    "proc0_print",
    "_adaptive_R0_R1_search",
    "_build_cached_thresholds_dict",
    "_build_optimization_results_dict",
    "_compute_final_metrics",
    "_coils_via_symmetries_compat",
    "_extend_coils_to_higher_order",
    "_format_verbose_iteration_output",
    "_get_scipy_algorithm_options",
    "_is_ci_running",
    "_make_base_currents",
    "_merge_post_processing_into_results",
    "_nullcontext",
    "_optimize_coils_loop_impl",
    "_parse_optimizer_config",
    "_plot_bn_error_3d",
    "_redirect_verbose_to_file",
    "_save_results_and_compute_metrics",
    "_save_vtk_outputs",
    "_setup_virtual_casing",
    "_validate_algorithm_options",
    "_zip_output_files",
]
