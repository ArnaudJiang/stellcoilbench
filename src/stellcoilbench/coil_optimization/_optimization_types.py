"""Shared dataclasses for the coil optimization loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class OptimizationLoopContext:
    """Unified context for optimization setup summary and optimization step.

    Built in _optimize_coils_loop_impl after objectives/constraints are set up.
    Passed to _print_optimization_setup_summary and _run_optimization_step.
    """

    algorithm: str
    algorithm_options: Dict[str, Any]
    max_iterations: int
    max_iter_subopt: int
    verbose: bool
    th: Dict[str, Any]
    coil_width: float
    ncoils: int
    total_current: float
    target_B: float
    flux_threshold: float
    cc_threshold: float
    cs_threshold: float
    length_threshold: float
    curvature_threshold: float
    msc_threshold: float
    force_threshold: float
    torque_threshold: float
    arclength_variation_threshold: float
    major_radius: float
    coils: List[Any]
    c_list: List[Any]
    constraint_scaling: Dict[int, float]
    constraint_idx_to_term: Dict[int, str]
    constraint_names_and_thresholds: List[Any]
    cc_distance_index: int | None
    cs_distance_index: int | None
    base_curves: List[Any]
    Jf: Any
    Jls: List[Any]
    Jccdist: Any
    Jcsdist: Any
    Jlink: Any
    Jforce: Any
    Jtorque: Any
    Jts: List[Any] | None
    coil_objective_terms: Dict[str, Any] | None
    kwargs: Dict[str, Any]
    structural_obj: Any | None
    effective_obj_terms: Dict[str, Any] | None
    out_dir: Path | None = None
