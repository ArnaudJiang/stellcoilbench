"""
Coil optimization setup: BiotSavart creation, coil initialization,
objective/constraint construction, and post-optimization save.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

import numpy as np

from ..mpi_utils import proc0_print, proc0_try
from ..utils import suppress_output, timed_section
from ._constraint_builders import (
    _build_c_list_and_constraint_scaling_from_coil_objective_terms,
    _build_modular_coil_constraint_objects,
)
from ._plotting import (
    _compute_surface_vtk_data,
    _plot_bn_error_3d,
)

if TYPE_CHECKING:
    from simsopt.field import BiotSavart
    from simsopt.geo import SurfaceRZFourier


def _setup_biotSavart_and_initial_save(
    coils: list,
    s: "SurfaceRZFourier",
    s_plot: "SurfaceRZFourier",
    qphi: int,
    qtheta: int,
    out_dir: Path,
    *,
    save_coils_surface_vtk: bool = True,
    save_initial_state: bool = True,
) -> tuple:
    """
    Create BiotSavart and save initial state before optimization.

    Saves coils to VTK (coils_initial), surface with B_N/|B| and modB to VTK
    (surface_initial), and generates bn_error_3d_plot_initial.png.
    When save_initial_state=False (e.g. Fourier continuation order_8/order_16),
    skips initial VTK and plot to reduce submission size.

    Parameters
    ----------
    coils : list
        Coil objects.
    s, s_plot : Surface
        Optimization surface and plotting surface (full torus).
    qphi, qtheta : int
        Plotting grid dimensions.
    out_dir : Path
        Output directory.

    Returns
    -------
    tuple
        (bs, curves, B_initial) - BiotSavart, curve list, initial |B| on s_plot.
    """
    from simsopt.field import BiotSavart, coils_to_vtk
    from simsopt.util import calculate_modB_on_major_radius

    bs = BiotSavart(coils)
    with suppress_output():
        calculate_modB_on_major_radius(bs, s)
    curves = [c.curve for c in coils]

    if save_coils_surface_vtk and save_initial_state:
        with proc0_try(
            "Failed to save initial coils to VTK: {e}",
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            on_catch=lambda: proc0_print(
                "  Continuing optimization without VTK export..."
            ),
        ):
            coils_to_vtk(coils, out_dir / "coils_initial", close=True)

    with suppress_output():
        bs.set_points(s_plot.gamma().reshape((-1, 3)))
        B_initial = calculate_modB_on_major_radius(bs, s_plot)

    if save_coils_surface_vtk and save_initial_state:
        pointData = _compute_surface_vtk_data(bs, s_plot, qphi, qtheta)
        s_plot.to_vtk(out_dir / "surface_initial", extra_data=pointData)

        with proc0_try("Failed to generate initial 3D plot: {e}"):
            _plot_bn_error_3d(
                s_plot,
                bs,
                coils,
                out_dir,
                filename="bn_error_3d_plot_initial.png",
                title="B_N/|B| Error on Plasma Surface with Initial Coils",
            )

    return bs, curves, B_initial


def _save_optimized_coils_and_compute_metrics(
    coils: list,
    base_curves: list,
    ncoils: int,
    s: "SurfaceRZFourier",
    s_plot: "SurfaceRZFourier",
    qphi: int,
    qtheta: int,
    bs: "BiotSavart",
    out_dir: Path,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Save optimized coils and compute final metrics.

    Saves coils to VTK (coils_optimized) and JSON (biot_savart_optimized.json),
    surface with B_N, B_N/|B|, modB to VTK (surface_optimized), and generates
    bn_error_3d_plot.png. Computes force/torque (new or legacy API), B_N metrics
    (with optional virtual casing target), and coil-surface linking check.

    Parameters
    ----------
    coils, base_curves : list
        Coil objects and base curves.
    ncoils : int
        Number of base coils.
    s, s_plot : Surface
        Optimization and plotting surfaces.
    qphi, qtheta : int
        Plotting grid dimensions.
    bs : BiotSavart
        BiotSavart with optimized coils.
    out_dir : Path
        Output directory.
    kwargs : Dict[str, Any]
        May contain vc_target, vc_target_plot for virtual casing,
        save_coils_surface_vtk (default True) to skip VTK/plot outputs.

    Returns
    -------
    Dict[str, Any]
        B_final, max_force, max_torque, avg_BdotN_over_B, max_BdotN_overB,
        coils_linked_to_surface, total_current_final.
    """
    from ._external_eval import _compute_optimization_metrics
    from simsopt.field import coils_to_vtk

    save_coils_surface_vtk = kwargs.get("save_coils_surface_vtk", True)

    sg_extra: Dict[str, Any] | None = None
    if save_coils_surface_vtk:
        with proc0_try(
            "Shape gradient computation failed: {e}",
            ImportError,
            RuntimeError,
            ValueError,
            TypeError,
            np.linalg.LinAlgError,
        ):
            from stellcoilbench.post_processing._shape_gradient import (
                compute_shape_gradients,
                shape_gradient_to_vtk_data,
            )

            sg = compute_shape_gradients(coils, bs, s)
            sg_extra = shape_gradient_to_vtk_data(sg, coils, close=True)

    if save_coils_surface_vtk:
        with proc0_try(
            "Failed to save optimized coils to VTK: {e}",
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            on_catch=lambda: proc0_print("  Continuing without VTK export..."),
        ):
            coils_to_vtk(
                coils, out_dir / "coils_optimized", close=True, extra_data=sg_extra
            )
    bs.save(out_dir / "biot_savart_optimized.json")

    if save_coils_surface_vtk:
        pointData = _compute_surface_vtk_data(bs, s_plot, qphi, qtheta, include_bn=True)
        s_plot.to_vtk(out_dir / "surface_optimized", extra_data=pointData)

    metrics = _compute_optimization_metrics(
        bs, coils, base_curves, ncoils, s, s_plot, qphi, qtheta, kwargs
    )

    if save_coils_surface_vtk:
        with proc0_try("Failed to generate 3D plot: {e}"):
            vc_target_plot = kwargs.get("vc_target_plot", None)
            _plot_bn_error_3d(
                s_plot,
                bs,
                coils,
                out_dir,
                filename="bn_error_3d_plot.png",
                title="B_N/|B| Error on Plasma Surface with Optimized Coils",
                vc_target=vc_target_plot,
            )

    return metrics


def _initialize_coils_for_optimization(
    s: "SurfaceRZFourier",
    target_B: float,
    out_dir: Path,
    ncoils: int,
    order: int,
    coil_width: float,
    regularization: Callable | None,
    initial_coils: list | None,
    is_continuation_step: bool,
    kwargs: Dict[str, Any],
) -> Tuple[list, bool]:
    """Initialize modular TF coils for optimization.

    Handles two initialization paths:
    1. **Fresh modular coils** — calls ``initialize_coils_loop`` and
       optionally perturbs DOFs to break determinism.
    2. **Continuation step** — directly reuses ``initial_coils``.

    Returns
    -------
    tuple
        ``(coils, fix_shapes)`` — coil list and fix_shapes flag (always False).
    """
    fix_shapes: bool = False

    with timed_section("coil_initialization"):
        if initial_coils is None:
            from ._optimization_loop import initialize_coils_loop

            with suppress_output():
                coils = initialize_coils_loop(
                    s,
                    out_dir=out_dir,
                    target_B=target_B,
                    ncoils=ncoils,
                    order=order,
                    coil_width=coil_width,
                    regularization=regularization,
                )
            dof_perturbation = kwargs.get("dof_perturbation", 0.0)
            if isinstance(dof_perturbation, (int, float)) and dof_perturbation > 0:
                proc0_print(
                    f"  Applying DOF perturbation with scale {dof_perturbation}"
                )
                for coil in coils[:ncoils]:
                    x = coil.curve.x
                    noise = np.random.randn(len(x)) * dof_perturbation * np.std(x)
                    coil.curve.x = x + noise
        else:
            coils = initial_coils

    return coils, fix_shapes


def _build_objectives_and_constraints(
    s: "SurfaceRZFourier",
    bs: "BiotSavart",
    coils: list,
    coils_for_bs: list,
    base_curves: list,
    curves: list,
    ncoils: int,
    total_current: float,
    major_radius: float,
    coil_objective_terms: Dict[str, Any] | None,
    thresholds: Dict[str, float],
    kwargs: Dict[str, Any],
    *,
    out_dir: Path | None = None,
) -> Dict[str, Any]:
    """Build the objective function and constraint list for coil optimization.

    Constructs the ``SquaredFlux`` objective and assembles all constraint
    objects (length, curvature, MSC, coil-coil distance, coil-surface
    distance, arclength variation, linking number, force, torque) into the
    ``c_list`` used by both augmented-Lagrangian and scipy solvers.

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface.
    bs : BiotSavart
        BiotSavart field object (evaluation points are set internally).
    coils : list
        Modular coil objects.
    coils_for_bs : list
        All coils passed to BiotSavart.
    base_curves : list
        Base curves whose DOFs are optimized.
    curves : list
        All curves from BiotSavart (used by constraint builders).
    ncoils : int
        Number of unique TF base coils.
    total_current : float
        Sum of unique base-coil currents.
    major_radius : float
        Plasma major radius (for constraint scaling).
    coil_objective_terms : Dict[str, Any] | None
        User-supplied objective term configuration.
    thresholds : Dict[str, float]
        Dict with keys ``flux_threshold``, ``cc_threshold``,
        ``cs_threshold``, ``length_threshold``, ``curvature_threshold``,
        ``arclength_variation_threshold``, ``msc_threshold``,
        ``force_threshold``, ``torque_threshold``.
    kwargs : Dict[str, Any]
        Additional options (``vc_target``, ``fix_center``, weight keys).
    out_dir : Path | None
        Output directory for structural stress VTK (optional).

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys: ``Jf``, ``c_list``, ``constraint_scaling``,
        ``cc_distance_index``, ``cs_distance_index``,
        ``constraint_names_and_thresholds``, ``constraint_idx_to_term``,
        ``Jls``, ``Jccdist``, ``Jcsdist``, ``Jlink``, ``Jforce``,
        ``Jtorque``, ``Jmscs``, ``effective_obj_terms``,
    """
    import time
    from simsopt.objectives import SquaredFlux

    from ..utils import _timing_results
    from ._structural_stress import (
        _StructuralStressGuardWrapper,
        _StructuralStressShortCircuitWrapper,
        _build_structural_stress_objective,
    )

    objective_setup_start = time.perf_counter()
    bs.set_points(s.gamma().reshape((-1, 3)))

    flux_threshold = thresholds["flux_threshold"]
    cc_threshold = thresholds["cc_threshold"]
    cs_threshold = thresholds["cs_threshold"]
    length_threshold = thresholds["length_threshold"]
    curvature_threshold = thresholds["curvature_threshold"]
    torsion_threshold = thresholds.get("torsion_threshold", 0.0)
    arclength_variation_threshold = thresholds["arclength_variation_threshold"]
    msc_threshold = thresholds["msc_threshold"]
    force_threshold = thresholds["force_threshold"]
    torque_threshold = thresholds["torque_threshold"]

    vc_target = kwargs.get("vc_target", None)
    if vc_target is not None:
        proc0_print(
            f"Using virtual casing target for SquaredFlux (target shape: {vc_target.shape})"
        )
        Jf = SquaredFlux(s, bs, target=vc_target, threshold=flux_threshold)
    else:
        Jf = SquaredFlux(s, bs, threshold=flux_threshold)

    constraint_objs = _build_modular_coil_constraint_objects(
        curves,
        base_curves,
        coils,
        ncoils,
        s,
        cc_threshold,
        cs_threshold,
        curvature_threshold,
        torsion_threshold,
        force_threshold,
        torque_threshold,
        coil_objective_terms,
    )
    effective_obj_terms = coil_objective_terms or {}

    Jls = constraint_objs["Jls"]
    Jccdist = constraint_objs["Jccdist"]
    Jcsdist = constraint_objs["Jcsdist"]
    Jalenvar = constraint_objs["Jalenvar"]
    Jcs = constraint_objs["Jcs"]
    Jts = constraint_objs.get("Jts")
    Jlink = constraint_objs["Jlink"]
    Jforce = constraint_objs["Jforce"]
    Jtorque = constraint_objs["Jtorque"]
    Jmscs = constraint_objs["Jmscs"]

    thresholds_for_build = {
        "cc_threshold": cc_threshold,
        "cs_threshold": cs_threshold,
        "length_threshold": length_threshold,
        "curvature_threshold": curvature_threshold,
        "torsion_threshold": torsion_threshold,
        "arclength_variation_threshold": arclength_variation_threshold,
        "msc_threshold": msc_threshold,
        "force_threshold": force_threshold,
        "torque_threshold": torque_threshold,
    }
    (
        c_list,
        constraint_scaling,
        cc_distance_index,
        cs_distance_index,
        constraint_names_and_thresholds,
        constraint_idx_to_term,
    ) = _build_c_list_and_constraint_scaling_from_coil_objective_terms(
        Jf,
        Jccdist,
        Jcsdist,
        Jls,
        Jcs,
        Jalenvar,
        Jmscs,
        Jlink,
        Jforce,
        Jtorque,
        effective_obj_terms,
        thresholds_for_build,
        major_radius,
        total_current,
        Jts=Jts,
    )

    structural_obj: Any | None = None
    structural_obj_raw: Any | None = None
    if effective_obj_terms and "structural_stress" in effective_obj_terms:
        structural_obj_raw = _build_structural_stress_objective(
            coils_for_bs,
            bs,
            ncoils,
            effective_obj_terms,
            thresholds,
            out_dir=out_dir,
            animation_frames_dir=kwargs.get("_structural_animation_frames_dir"),
            animation_frame_counter=kwargs.get("_structural_animation_frame_counter"),
            animation_surface_snap=kwargs.get("_structural_animation_surface_snap"),
        )
        structural_obj = structural_obj_raw
        if structural_obj is not None:
            from ._constraint_builders import _NAME_MAP, _TERM_MAP
            from ._structural_objective import PA_TO_GPA

            term_value = effective_obj_terms["structural_stress"]
            penalty_options = _TERM_MAP.get("structural_stress", {})
            if term_value in penalty_options:
                stress_thresh_pa = float(
                    effective_obj_terms.get(
                        "structural_stress_threshold",
                        0.0,
                    )
                )
                stress_thresh_gpa = stress_thresh_pa * PA_TO_GPA
                effective_thresh_gpa = 0.9 * stress_thresh_gpa
                guarded_penalty_gpa = max(10.0 * stress_thresh_gpa, 1.0)
                structural_obj_guarded = _StructuralStressGuardWrapper(
                    structural_obj,
                    Jccdist,
                    cc_threshold,
                    safety_frac=0.5,
                    guarded_penalty_gpa=guarded_penalty_gpa,
                )
                structural_stress_weight = float(
                    effective_obj_terms.get("structural_stress_weight", 1.0)
                )
                structural_obj_wrapped = _StructuralStressShortCircuitWrapper(
                    structural_obj_guarded,
                    effective_thresh_gpa,
                    weight=structural_stress_weight,
                )
                penalty_fn = penalty_options[term_value]
                constraint = penalty_fn(structural_obj_wrapped, effective_thresh_gpa)
                constraint_idx = len(c_list)
                c_list.append(constraint)
                constraint_scaling[constraint_idx] = 1.0
                constraint_names_and_thresholds.append(
                    (_NAME_MAP.get("structural_stress", "σ_vm"), stress_thresh_gpa)
                )
                constraint_idx_to_term[constraint_idx] = "structural_stress"
                structural_obj = structural_obj_wrapped

    objective_setup_time = time.perf_counter() - objective_setup_start
    _timing_results["objective_setup"] = objective_setup_time

    return {
        "Jf": Jf,
        "c_list": c_list,
        "constraint_scaling": constraint_scaling,
        "cc_distance_index": cc_distance_index,
        "cs_distance_index": cs_distance_index,
        "constraint_names_and_thresholds": constraint_names_and_thresholds,
        "constraint_idx_to_term": constraint_idx_to_term,
        "Jls": Jls,
        "Jccdist": Jccdist,
        "Jcsdist": Jcsdist,
        "Jlink": Jlink,
        "Jforce": Jforce,
        "Jtorque": Jtorque,
        "Jts": Jts,
        "Jmscs": Jmscs,
        "effective_obj_terms": effective_obj_terms,
        "structural_obj": structural_obj,
        "structural_obj_raw": structural_obj_raw,
    }
