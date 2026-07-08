"""Results dictionary construction for coil optimization.

Extracts threshold caching, results-dict building, and post-processing merge
logic into a dedicated module for clearer separation of concerns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ._length_balance import coil_length_distribution_metrics


def _build_cached_thresholds_dict(th: Dict[str, Any]) -> Dict[str, Any]:
    """Extract threshold values to cache for Fourier continuation.

    Parameters
    ----------
    th : Dict[str, Any]
        Full thresholds dictionary.

    Returns
    -------
    Dict[str, Any]
        Subset of thresholds needed for continuation steps.
    """
    _CACHED_KEYS = (
        "length_threshold",
        "flux_threshold",
        "cc_threshold",
        "cs_threshold",
        "msc_threshold",
        "arclength_variation_threshold",
        "length_variance_threshold",
        "curvature_threshold",
        "force_threshold",
        "torque_threshold",
        "coil_width",
        "a0",
        "major_radius",
        "minor_radius",
        "finite_build_width",
    )
    return {k: v for k, v in th.items() if k in _CACHED_KEYS}


def _build_optimization_results_dict(
    *,
    B_initial: Any,
    B_final: Any,
    target_B: float,
    end_time: float,
    start_time: float,
    iterations_used: int,
    Jf: Any,
    Jcsdist: Any,
    Jccdist: Any,
    Jlink: Any,
    opt_result: Any,
    cached_thresholds: Dict[str, Any],
    base_curves: list,
    coils: list,
    ncoils: int,
    total_current: float,
    total_current_final: float,
    max_force: list,
    max_torque: list,
    avg_BdotN_over_B: float,
    max_BdotN_overB: float,
    coils_linked_to_surface: bool,
    lag_mul: Any,
    out_dir: Path,
    th: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the optimization results dictionary.

    Aggregates flux, geometry, force/torque, B_N metrics, and thresholds
    into a single dict for reporting and continuation caching.

    Parameters
    ----------
    B_initial, B_final, target_B : float or array
        Initial/final |B| and target field.
    end_time, start_time : float
        Wall-clock times.
    iterations_used : int
        Total optimization iterations.
    Jf, Jcsdist, Jccdist : objectives
        Flux and distance objectives for final values.
    opt_result : object or None
        scipy minimize result (success, message, nfev, njev).
    cached_thresholds : Dict[str, Any]
        Thresholds to cache for Fourier continuation.
    base_curves, coils : list
        Base curves and coil objects.
    ncoils : int
        Number of base coils.
    total_current, total_current_final : float
        Current before/after optimization.
    max_force, max_torque : list
        Per-coil max force and torque.
    avg_BdotN_over_B, max_BdotN_overB : float
        B_N/|B| metrics.
    coils_linked_to_surface : bool
        Whether coils encircle plasma.
    lag_mul : Any
        Lagrange multipliers (auglag) or None.
    out_dir : Path
        Output directory.
    th : Dict[str, Any]
        Full thresholds dict for reporting.

    Returns
    -------
    Dict[str, Any]
        Results dict (without post_processing or timing; caller adds those).
    """
    from simsopt.geo import CurveLength, ArclengthVariation

    # Convert to flat floats (handles inhomogeneous shapes from dipole+TF mix)
    max_force_flat = [float(np.asarray(f).max()) for f in max_force]
    max_torque_flat = [float(np.asarray(t).max()) for t in max_torque]

    lengths = [float(CurveLength(c).J()) for c in base_curves]
    length_metrics = coil_length_distribution_metrics(lengths)

    return {
        "initial_B_field": B_initial,
        "final_B_field": B_final,
        "target_B_field": target_B,
        "optimization_time": end_time - start_time,
        "walltime_sec": end_time - start_time,
        "iterations_used": iterations_used,
        "final_squared_flux": Jf.J(),
        "optimization_success": (
            opt_result.success
            if opt_result is not None and hasattr(opt_result, "success")
            else True
        ),
        "optimization_message": (
            str(opt_result.message)
            if opt_result is not None and hasattr(opt_result, "message")
            else "Completed"
        ),
        "optimization_nfev": (
            getattr(opt_result, "nfev", None) or iterations_used
            if opt_result is not None
            else iterations_used
        ),
        "optimization_njev": (
            getattr(opt_result, "njev", None) if opt_result is not None else None
        ),
        "_cached_thresholds": cached_thresholds,
        "final_min_cs_separation": Jcsdist.shortest_distance(),
        "final_min_cc_separation": Jccdist.shortest_distance(),
        "final_length_per_coil": lengths,
        "final_current_per_coil": [
            float(abs(coils[i].current.get_value())) for i in range(ncoils)
        ],
        "total_current_before": float(total_current),
        "total_current_after": float(total_current_final),
        "final_total_length": float(sum(lengths)),
        **length_metrics,
        "final_max_curvature": max(np.max(c.kappa()) for c in base_curves),
        "final_average_curvature": float(
            np.mean(
                np.concatenate(
                    [np.atleast_1d(c.kappa()).flatten() for c in base_curves]
                )
            )
        ),
        "final_arclength_variation": np.mean(
            [ArclengthVariation(c).J() for c in base_curves]
        ),
        "final_mean_squared_curvature": np.max(
            [np.mean(c.kappa() ** 2) for c in base_curves]
        ),
        "final_max_torsion": float(
            np.max([np.max(np.abs(np.asarray(c.torsion()))) for c in base_curves])
        )
        if base_curves
        else 0.0,
        "final_linking_number": Jlink.J(),
        "coils_linked_to_surface": coils_linked_to_surface,
        "final_max_max_coil_force": float(np.max(max_force_flat))
        if max_force_flat
        else 0.0,
        "final_avg_max_coil_force": float(np.mean(max_force_flat))
        if max_force_flat
        else 0.0,
        "final_max_force_per_coil": max_force_flat,
        "final_max_torque_per_coil": max_torque_flat,
        "final_max_max_coil_torque": float(np.max(max_torque_flat))
        if max_torque_flat
        else 0.0,
        "final_avg_max_coil_torque": float(np.mean(max_torque_flat))
        if max_torque_flat
        else 0.0,
        "avg_BdotN_over_B": avg_BdotN_over_B,
        "max_BdotN_over_B": max_BdotN_overB,
        "lagrange_multipliers": lag_mul,
        "output_directory": str(out_dir),
        "flux_threshold": th.get("flux_threshold"),
        "cc_threshold": th.get("cc_threshold"),
        "cs_threshold": th.get("cs_threshold"),
        "msc_threshold": th.get("msc_threshold"),
        "arclength_variation_threshold": th.get("arclength_variation_threshold"),
        "curvature_threshold": th.get("curvature_threshold"),
        "force_threshold": th.get("force_threshold"),
        "torque_threshold": th.get("torque_threshold"),
    }


def _merge_post_processing_into_results(
    results: Dict[str, Any],
    post_processing_results: Dict[str, Any],
) -> None:
    """Merge numeric post-processing metrics into results dict (in-place).

    Parameters
    ----------
    results : Dict[str, Any]
        Optimization results dict to update.
    post_processing_results : Dict[str, Any]
        Post-processing results (QS, loss fraction, BdotN metrics, structural).
    """
    _PP_NUMERIC_KEYS = (
        "quasisymmetry_average",
        "loss_fraction",
        "BdotN",
        "BdotN_over_B",
    )
    for key in _PP_NUMERIC_KEYS:
        value = post_processing_results.get(key)
        if isinstance(value, (int, float)):
            results[key] = float(value)
    if "structural_metrics" in post_processing_results:
        results["structural_metrics"] = post_processing_results["structural_metrics"]
