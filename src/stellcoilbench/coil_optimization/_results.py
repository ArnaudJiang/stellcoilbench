"""Optimization results dictionary construction.

Assembles flux, geometry, force/torque, and B_N metrics into a single
dict used for reporting, continuation caching, and JSON serialization.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple

from ..config_scheme import PostProcessingConfig
from ..utils import _timing_results, get_timing_results

from ._results_dict import (
    _build_cached_thresholds_dict,
    _build_optimization_results_dict,
    _merge_post_processing_into_results,
)

if TYPE_CHECKING:
    from simsopt.field import BiotSavart
    from simsopt.geo import SurfaceRZFourier

logger = logging.getLogger(__name__)


def compute_total_current(coils: list, ncoils: int | None = None) -> float:
    """Sum the currents of the first *ncoils* coils.

    Handles ``AttributeError`` / ``TypeError`` gracefully, falling back to
    a per-coil ``abs()`` approach when the straightforward sum fails.

    Parameters
    ----------
    coils : list
        Coil objects with a ``.current.get_value()`` accessor.
    ncoils : int | None
        Number of leading coils to sum.  ``None`` sums all.

    Returns
    -------
    float
        Net current [A].
    """
    subset = coils if ncoils is None else coils[:ncoils]
    try:
        return float(sum(c.current.get_value() for c in subset))
    except (AttributeError, TypeError):
        return float(
            sum(
                abs(coils[i].current.get_value())
                for i in range(len(subset))
                if hasattr(coils[i], "current")
            )
        )


@dataclass
class OptimizationOutcome:
    """Bundles optimiser outputs that flow into results-dict assembly.

    Reduces the number of keyword arguments threaded through
    ``_save_results_and_compute_metrics`` and ``_compute_final_metrics``.

    Attributes
    ----------
    Jf : Any
        Objective function (``QuadraticPenalty`` wrapper or similar).
    Jcsdist : Any
        Coil-to-surface distance penalty.
    Jccdist : Any
        Coil-to-coil distance penalty.
    Jlink : Any
        Linking-number penalty.
    opt_result : Any
        Scipy ``OptimizeResult`` or ``None`` (augmented-Lagrangian).
    cached_thresholds : Dict[str, Any]
        Snapshot of threshold values for Fourier continuation.
    th : Dict[str, Any]
        Full thresholds dictionary.
    total_current : float
        Sum of unique base-coil currents before optimisation.
    target_B : float
        Target on-axis field [T].
    start_time : float
        Wall-clock timestamp at optimisation start.
    end_time : float
        Wall-clock timestamp at optimisation end.
    iterations_used : int
        Total optimiser iterations consumed.
    lag_mul : Any
        Lagrange multipliers (``None`` for scipy methods).
    base_curves : list
        Base curves whose DOFs were optimised.
    ncoils : int
        Number of unique TF base coils.
    skip_post_processing : bool
        Whether to skip QFM/VMEC/Poincaré.
    case_path : Path | None
        Path to ``case.yaml`` for post-processing.
    pp_flags : PostProcessingConfig
        Post-processing flags (run_vmec, plot_poincare, nfieldlines, etc.).
    B_initial : Any
        Initial B-field magnitude before optimisation (for results dict).
    structural_max_von_mises_Pa : float | None
        Max von Mises stress [Pa] from the structural objective's final evaluation,
        when structural_stress was in the objective. Used to avoid redundant
        post-processing structural solve.
    """

    Jf: Any = None
    Jcsdist: Any = None
    Jccdist: Any = None
    Jlink: Any = None
    opt_result: Any = None
    cached_thresholds: Dict[str, Any] = field(default_factory=dict)
    th: Dict[str, Any] = field(default_factory=dict)
    total_current: float = 0.0
    target_B: float = 5.7
    start_time: float = 0.0
    end_time: float = 0.0
    iterations_used: int = 0
    lag_mul: Any = None
    base_curves: list = field(default_factory=list)
    ncoils: int = 0
    skip_post_processing: bool = False
    case_path: Path | None = None
    pp_flags: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    B_initial: Any = None
    structural_max_von_mises_Pa: float | None = None
    initial_geometry_metrics: Dict[str, Any] = field(default_factory=dict)


def _save_vtk_outputs(
    s: "SurfaceRZFourier",
    s_plot: "SurfaceRZFourier",
    qphi: int,
    qtheta: int,
    coils: list,
    base_curves: list,
    ncoils: int,
    bs: "BiotSavart",
    out_dir: Path,
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], list]:
    """Save optimized coils to VTK/JSON and compute B-field metrics.

    Delegates to ``_save_optimized_coils_and_compute_metrics`` for modular coils.

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface.
    s_plot : SurfaceRZFourier
        Higher-resolution surface for plotting.
    qphi, qtheta : int
        Surface quadrature resolution.
    coils : list
        Modular coil objects.
    base_curves : list
        Base curves whose DOFs were optimized.
    ncoils : int
        Number of unique base coils.
    bs : BiotSavart
        BiotSavart field object.
    out_dir : Path
        Output directory.
    kwargs : Dict[str, Any]
        Pass-through options (``vc_target_plot``, etc.).

    Returns
    -------
    tuple
        ``(metrics, coils_return)``
    """
    from .optimization import _save_optimized_coils_and_compute_metrics

    metrics = _save_optimized_coils_and_compute_metrics(
        coils, base_curves, ncoils, s, s_plot, qphi, qtheta, bs, out_dir, kwargs
    )
    return metrics, coils


def _compute_final_metrics(
    metrics: Dict[str, Any],
    coils_return: list,
    s: "SurfaceRZFourier",
    bs: "BiotSavart",
    out_dir: Path,
    kwargs: Dict[str, Any],
    outcome: OptimizationOutcome,
) -> Dict[str, Any]:
    """Assemble the final results dictionary and run optional post-processing.

    Takes pre-computed VTK/metrics outputs from ``_save_vtk_outputs``, runs
    post-processing (QFM, VMEC, Poincaré, Boozer) if requested, and builds the
    results dict via ``_build_optimization_results_dict``.

    Parameters
    ----------
    metrics : Dict[str, Any]
        B-field metrics from ``_save_vtk_outputs`` or
        ``_save_optimized_coils_and_compute_metrics``.
    coils_return : list
        Coils to return to the caller.
    s : SurfaceRZFourier
        Plasma boundary surface.
    bs : BiotSavart
        BiotSavart field object.
    out_dir : Path
        Output directory.
    kwargs : Dict[str, Any]
        Pass-through options.
    outcome : OptimizationOutcome
        Bundled optimiser outputs (objectives, thresholds, timing, flags).

    Returns
    -------
    Dict[str, Any]
        The fully assembled results dictionary.
    """
    from ._post_opt_processing import _run_post_processing_after_optimization

    B_final = metrics["B_final"]
    max_force = metrics["max_force"]
    max_torque = metrics["max_torque"]
    avg_BdotN_over_B = metrics["avg_BdotN_over_B"]
    max_BdotN_overB = metrics["max_BdotN_overB"]
    avg_BdotN_over_target_B = metrics.get("avg_BdotN_over_target_B", 0.0)
    max_BdotN_over_target_B = metrics.get("max_BdotN_over_target_B", 0.0)
    coils_linked_to_surface = metrics["coils_linked_to_surface"]
    total_current_final = metrics["total_current_final"]

    post_processing_results = (
        _run_post_processing_after_optimization(
            out_dir,
            s,
            outcome.case_path,
            outcome.pp_flags,
        )
        if not outcome.skip_post_processing
        else {}
    )

    cached_thresholds = _build_cached_thresholds_dict(outcome.th)

    bs.set_points(s.gamma().reshape((-1, 3)))

    results = _build_optimization_results_dict(
        B_initial=outcome.B_initial,
        B_final=B_final,
        target_B=outcome.target_B,
        end_time=outcome.end_time,
        start_time=outcome.start_time,
        iterations_used=outcome.iterations_used,
        Jf=outcome.Jf,
        Jcsdist=outcome.Jcsdist,
        Jccdist=outcome.Jccdist,
        Jlink=outcome.Jlink,
        opt_result=outcome.opt_result,
        cached_thresholds=cached_thresholds,
        base_curves=outcome.base_curves,
        coils=coils_return,
        ncoils=outcome.ncoils,
        total_current=outcome.total_current,
        total_current_final=total_current_final,
        max_force=max_force,
        max_torque=max_torque,
        avg_BdotN_over_B=avg_BdotN_over_B,
        max_BdotN_overB=max_BdotN_overB,
        avg_BdotN_over_target_B=avg_BdotN_over_target_B,
        max_BdotN_over_target_B=max_BdotN_over_target_B,
        coils_linked_to_surface=coils_linked_to_surface,
        lag_mul=outcome.lag_mul,
        out_dir=out_dir,
        th=outcome.th,
    )

    if post_processing_results:
        _merge_post_processing_into_results(results, post_processing_results)

    if outcome.initial_geometry_metrics:
        results.update(outcome.initial_geometry_metrics)

    # Add max von Mises for correlation studies (top-level for collect_metrics).
    # Prefer post-processing value when available; otherwise use optimization value.
    if outcome.structural_max_von_mises_Pa is not None:
        results["max_von_mises_stress_Pa"] = outcome.structural_max_von_mises_Pa
    if "structural_metrics" in results:
        vm = results["structural_metrics"].get("max_von_mises_stress_Pa")
        if vm is not None:
            results["max_von_mises_stress_Pa"] = vm

    results["timing"] = get_timing_results()
    return results


def _save_results_and_compute_metrics(
    s: "SurfaceRZFourier",
    s_plot: "SurfaceRZFourier",
    qphi: int,
    qtheta: int,
    bs: "BiotSavart",
    coils: list,
    out_dir: Path,
    kwargs: Dict[str, Any],
    outcome: OptimizationOutcome,
) -> Tuple[list, Dict[str, Any]]:
    """Save optimized coils, compute metrics, build results dict, and run post-processing.

    After optimization completes this function:

    1. Delegates VTK/JSON saving and metrics computation to
       ``_save_vtk_outputs``.
    2. Delegates results-dict assembly and post-processing to
       ``_compute_final_metrics``.

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface.
    s_plot : SurfaceRZFourier
        Higher-resolution surface for plotting.
    qphi, qtheta : int
        Surface quadrature resolution.
    bs : BiotSavart
        BiotSavart field object.
    coils : list
        Modular coil objects.
    out_dir : Path
        Output directory.
    kwargs : Dict[str, Any]
        Pass-through options (``vc_target_plot``, etc.).
    outcome : OptimizationOutcome
        Bundled optimiser outputs (objectives, thresholds, timing, coil info, flags).

    Returns
    -------
    tuple
        ``(coils_return, results)`` — the coils list to return to the
        caller and the assembled results dictionary.
    """
    save_metrics_start = time.perf_counter()
    metric_kwargs = dict(kwargs)
    metric_kwargs["target_B"] = outcome.target_B

    metrics, coils_return = _save_vtk_outputs(
        s,
        s_plot,
        qphi,
        qtheta,
        coils,
        outcome.base_curves,
        outcome.ncoils,
        bs,
        out_dir,
        metric_kwargs,
    )

    save_metrics_time = time.perf_counter() - save_metrics_start
    _timing_results["save_and_metrics"] = save_metrics_time

    results = _compute_final_metrics(
        metrics,
        coils_return,
        s,
        bs,
        out_dir,
        kwargs,
        outcome,
    )

    return coils_return, results
