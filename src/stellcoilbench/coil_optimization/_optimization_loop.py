"""
Coil optimization loop: initialize coils, run optimization steps,
and orchestrate the modular optimization pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from ..config_scheme import PostProcessingConfig
from ..mpi_utils import comm_world, is_mpi_enabled, is_proc0, proc0_print
from ..constants import (
    CURRENT_SCALING_CONVERGENCE_TOL,
    DEFAULT_COIL_QUADPOINTS,
    INITIAL_TOTAL_CURRENT,
    MAX_CURRENT_ADJUSTMENT_ITERS,
)
from ..utils import _timing_results, suppress_output, timed_section
from ._adaptive_search import _coils_via_symmetries_compat, _make_base_currents
from ._ci_utils import _is_ci_running, _nullcontext, _redirect_verbose_to_file
from ._finite_section_field import (
    build_finite_section_coil_bundle,
    parse_finite_section_field_config,
)
from ._optimization_types import OptimizationLoopContext
from ._plotting import (
    _compute_surface_vtk_data,
    _create_plotting_surface,
    _print_optimization_setup_summary,
)
from ._results import (
    OptimizationOutcome,
    compute_total_current,
    _build_cached_thresholds_dict,
    _save_results_and_compute_metrics,
)
from ._scipy_optimizer import (
    _apply_distance_weights_for_auglag,
    _parse_optimizer_config,
    _run_augmented_lagrangian,
    _run_scipy_minimize_for_modular_coils,
)
from ._optimization_setup import (
    _build_objectives_and_constraints,
    compute_initial_geometry_metrics,
    _initialization_kwargs,
    _initialize_coils_for_optimization,
    _setup_biotSavart_and_initial_save,
)
from ._structural_mpi_worker import _structural_dj_worker_loop

import numpy as np


def _bcast_coil_dofs_to_workers(
    base_curves: List[Any],
    coils: list,
) -> None:
    """Broadcast coil DOFs from rank 0 to worker ranks so they can update coils in place.

    When running with structural MPI, only rank 0 runs the optimizer. Workers need
    the optimized coil geometry for Fourier continuation (to extend coils to the
    next order). Rank 0 broadcasts the concatenated DOFs of base_curves; workers
    receive and apply them to their local base_curves (which share structure with
    coils via references).

    Parameters
    ----------
    base_curves : list
        Base curves whose DOFs were optimized (rank 0) or will be updated (workers).
    coils : list
        Full coil list. Workers use this for the return value; coils reference
        base_curves, so updating base_curves updates coils in place.
    """
    if not is_mpi_enabled() or comm_world.size <= 1:
        return
    rank = comm_world.rank
    if rank == 0:
        dofs_per_curve = [bc.get_dofs() for bc in base_curves]
        full_dofs = np.concatenate(
            [np.asarray(d, dtype=np.float64) for d in dofs_per_curve]
        )
        n = np.array([full_dofs.size], dtype=np.int32)
        comm_world.Bcast(n, root=0)
        comm_world.Bcast(full_dofs, root=0)
    else:
        n = np.empty(1, dtype=np.int32)
        comm_world.Bcast(n, root=0)
        full_dofs = np.empty(n[0], dtype=np.float64)
        comm_world.Bcast(full_dofs, root=0)
        offset = 0
        for bc in base_curves:
            ndof = bc.num_dofs()
            chunk = full_dofs[offset : offset + ndof].copy()
            bc.set_dofs(chunk)
            offset += ndof


def _get_regularization_circ() -> Callable[..., Any] | None:
    """Return regularization_circ from simsopt, or regularization_length if unavailable."""
    try:
        from simsopt.field import regularization_circ

        return regularization_circ
    except ImportError:
        pass
    try:
        from simsopt.field import regularization_length

        return regularization_length
    except ImportError:
        return None


regularization_circ = _get_regularization_circ()

_SCIPY_ALGORITHMS = frozenset(
    {
        "BFGS",
        "L-BFGS-B",
        "SLSQP",
        "Nelder-Mead",
        "Powell",
        "CG",
        "Newton-CG",
        "TNC",
        "COBYLA",
        "trust-constr",
    }
)


def initialize_coils_loop(
    s,
    out_dir: Path | str = "",
    target_B: float = 5.7,
    ncoils: int = 4,
    order: int = 16,
    numquadpoints: int = DEFAULT_COIL_QUADPOINTS,
    coil_width: float = 0.4,
    regularization: Callable[..., Any] | None = regularization_circ,
    major_radius_scale: float = 1.0,
    minor_radius_scale: float = 1.0,
    radial_offset: float = 0.0,
    current_scale: float = 1.0,
    current_weights: list[float] | tuple[float, ...] | None = None,
    initialization_metadata: dict[str, Any] | None = None,
) -> List[Any]:
    """
    Initialize modular coils with adaptive R0/R1 and target B-field scaling.

    Uses an adaptive strategy to determine R0 and R1 so that coils:
    - Do not intersect the plasma surface
    - Interlink the plasma (go around it)
    - Maintain safe coil-surface and coil-coil distances
    - Do not interlink each other (linking number ≈ 0)

    Iteratively adjusts R0/R1 until constraints are satisfied, then adjusts
    total current until the field along the major radius averages to target_B.

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface.
    out_dir : Path | str, optional
        Output directory for saved files.
    target_B : float, default=5.7
        Target magnetic field strength [T] on-axis.
    ncoils : int, default=4
        Number of base coils.
    order : int, default=16
        Fourier order for coil curves.
    numquadpoints : int, default=DEFAULT_COIL_QUADPOINTS
        Quadrature points along each base coil curve.
    coil_width : float, default=0.4
        Coil width [m] for regularization.
    regularization : callable, optional
        Regularization function (default: regularization_circ).

    Returns
    -------
    list
        List of simsopt Coil objects (including symmetric copies).
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import BiotSavart
    from simsopt.util.coil_optimization_helper_functions import (
        calculate_modB_on_major_radius,
    )

    from ._adaptive_search import _adaptive_R0_R1_search

    out_dir = Path(out_dir)

    if regularization is not None:
        regularizations = [regularization(coil_width) for _ in range(ncoils)]
    else:
        regularizations = None

    total_current = INITIAL_TOTAL_CURRENT * float(current_scale)

    R0, R1 = _adaptive_R0_R1_search(
        s,
        ncoils,
        order,
        total_current,
        regularizations,
        numquadpoints=numquadpoints,
        major_radius_scale=float(major_radius_scale),
        minor_radius_scale=float(minor_radius_scale),
        radial_offset=float(radial_offset),
        current_weights=current_weights,
    )
    if initialization_metadata is not None:
        initialization_metadata.update(
            {
                "initialization_source": "fresh_adaptive",
                "initial_R0": float(R0),
                "initial_R1": float(R1),
                "requested_major_radius_scale": float(major_radius_scale),
                "requested_minor_radius_scale": float(minor_radius_scale),
                "requested_radial_offset": float(radial_offset),
                "requested_current_scale": float(current_scale),
                "requested_current_weights": list(current_weights)
                if current_weights is not None
                else None,
            }
        )

    # Final coil creation with determined R0 and R1
    base_curves = create_equally_spaced_curves(
        ncoils,
        s.nfp,
        stellsym=s.stellsym,
        R0=R0,
        R1=R1,
        order=order,
        numquadpoints=numquadpoints,
    )
    base_currents = _make_base_currents(total_current, ncoils, current_weights)
    coils = _coils_via_symmetries_compat(
        base_curves,
        base_currents,
        s.nfp,
        s.stellsym,
        regularizations=regularizations,
    )

    # Iterative current adjustment to achieve the target B-field
    max_iterations = MAX_CURRENT_ADJUSTMENT_ITERS
    tolerance = CURRENT_SCALING_CONVERGENCE_TOL
    for _ in range(max_iterations):
        # Distribute current among coils
        base_currents = _make_base_currents(total_current, ncoils, current_weights)

        # Create coils using symmetries
        coils = _coils_via_symmetries_compat(
            base_curves,
            base_currents,
            s.nfp,
            s.stellsym,
            regularizations=regularizations,
        )

        # Create BiotSavart object to evaluate field
        bs = BiotSavart(coils)

        # Calculate field strength along major radius (suppress simsopt Bmag prints)
        with suppress_output():
            B_avg = calculate_modB_on_major_radius(bs, s)

        # Check convergence
        if abs(B_avg - target_B) / target_B < tolerance:
            break

        # Adjust current based on field difference
        current_scale_factor = target_B / B_avg
        total_current *= current_scale_factor

    return coils


def optimize_coils_loop(
    s,
    target_B: float = 5.7,
    out_dir: Path | str = "",
    max_iterations: int = 30,
    ncoils: int = 4,
    order: int = 16,
    verbose: bool = True,
    regularization: Callable[..., Any] | None = regularization_circ,
    coil_objective_terms: Dict[str, Any] | None = None,
    initial_coils: List[Any] | None = None,
    surface_resolution: int = 32,
    skip_post_processing: bool = False,
    case_path: Path | None = None,
    pp_flags: PostProcessingConfig | None = None,
    **kwargs: Any,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Optimize modular coils for a plasma surface.

    Initializes coils with target B-field (or uses initial_coils for Fourier
    continuation), then optimizes flux plus constraints via augmented
    Lagrangian or scipy (L-BFGS-B, BFGS, etc.).

    Optionally runs post-processing (QFM, VMEC, Poincare, quasisymmetry).
    Delegates to _optimize_coils_loop_impl.
    """
    if pp_flags is None:
        pp_flags = PostProcessingConfig()

    out_dir = Path(out_dir).resolve()

    verbose_output_file = None
    if (
        verbose
        and _is_ci_running()
        and not os.getenv("STELLCOILBENCH_CI_VERBOSE_STDOUT")
    ):
        verbose_output_file = out_dir / "verbose_output.txt"

    redirect_context = (
        _redirect_verbose_to_file(verbose_output_file)
        if verbose_output_file
        else _nullcontext()
    )

    impl_kwargs = dict(kwargs)
    if pp_flags.finite_build_width is not None:
        impl_kwargs["finite_build_width"] = pp_flags.finite_build_width

    with redirect_context:
        return _optimize_coils_loop_impl(
            s,
            target_B,
            out_dir,
            max_iterations,
            ncoils,
            order,
            verbose,
            regularization,
            coil_objective_terms,
            initial_coils,
            surface_resolution,
            skip_post_processing,
            case_path,
            pp_flags=pp_flags,
            **impl_kwargs,
        )


def _run_optimization_step(
    ctx: OptimizationLoopContext,
) -> Tuple[Any, int, float, float]:
    """Run the optimizer (augmented Lagrangian or scipy) and record timing."""
    import time

    optimization_start = time.perf_counter()
    start_time = time.time()
    opt_result = None
    iterations_used = 0

    if ctx.algorithm == "augmented_lagrangian":
        _apply_distance_weights_for_auglag(
            ctx.c_list,
            ctx.constraint_scaling,
            ctx.cc_distance_index,
            ctx.cs_distance_index,
            ctx.kwargs,
        )
        _run_augmented_lagrangian(
            ctx.c_list, ctx.max_iterations, ctx.max_iter_subopt, ctx.verbose, ctx.kwargs
        )
        iterations_used = ctx.max_iterations
    elif ctx.algorithm in _SCIPY_ALGORITHMS:
        scipy_extra_kw: Dict[str, Any] = {}
        if ctx.structural_obj is not None:
            scipy_extra_kw["structural_obj"] = ctx.structural_obj
        if ctx.out_dir is not None:
            scipy_extra_kw["out_dir"] = ctx.out_dir
        if ctx.history_interval is not None:
            scipy_extra_kw["history_interval"] = ctx.history_interval
            scipy_extra_kw["history_output_dir"] = (
                ctx.history_output_dir or ctx.out_dir
            )
        result, iterations_used = _run_scipy_minimize_for_modular_coils(
            ctx.c_list,
            ctx.constraint_scaling,
            ctx.constraint_idx_to_term,
            ctx.cc_distance_index,
            ctx.cs_distance_index,
            ctx.constraint_names_and_thresholds,
            ctx.base_curves,
            ctx.Jls,
            ctx.Jccdist,
            ctx.Jcsdist,
            ctx.Jlink,
            ctx.coils,
            ctx.ncoils,
            ctx.Jts,
            ctx.coil_objective_terms,
            ctx.algorithm,
            ctx.max_iterations,
            ctx.algorithm_options,
            ctx.verbose,
            ctx.kwargs,
            **scipy_extra_kw,
        )
        opt_result = result

    end_time = time.time()
    optimization_time = time.perf_counter() - optimization_start
    _timing_results["coil_optimization"] = optimization_time

    return opt_result, iterations_used, start_time, end_time


def _optimize_coils_loop_impl(
    s,
    target_B: float = 5.7,
    out_dir: Path | str = "",
    max_iterations: int = 30,
    ncoils: int = 4,
    order: int = 16,
    verbose: bool = False,
    regularization: Callable | None = regularization_circ,
    coil_objective_terms: Dict[str, Any] | None = None,
    initial_coils: list | None = None,
    surface_resolution: int = 32,
    skip_post_processing: bool = False,
    case_path: Path | None = None,
    *,
    pp_flags: PostProcessingConfig | None = None,
    **kwargs,
):
    """Internal implementation of modular coil optimization (modular coils only)."""
    if pp_flags is None:
        pp_flags = PostProcessingConfig()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    is_continuation_step = initial_coils is not None
    config = _parse_optimizer_config(
        s,
        kwargs,
        max_iterations,
        is_continuation_step=is_continuation_step,
        coil_objective_terms=coil_objective_terms,
    )
    algorithm = config["algorithm"]
    algorithm_options = config["algorithm_options"]
    max_iter_subopt = config["max_iter_subopt"]
    max_iterations = config["max_iterations"]
    th = config["thresholds"]
    length_threshold = th["length_threshold"]
    flux_threshold = th["flux_threshold"]
    cc_threshold = th["cc_threshold"]
    cs_threshold = th["cs_threshold"]
    msc_threshold = th["msc_threshold"]
    arclength_variation_threshold = th.get("arclength_variation_threshold", 0.0)
    curvature_threshold = th["curvature_threshold"]
    force_threshold = th["force_threshold"]
    torque_threshold = th["torque_threshold"]
    coil_width = th.get("coil_width", 0.4 / th["a0"])
    major_radius = th["major_radius"]
    numquadpoints = int(kwargs.get("numquadpoints", DEFAULT_COIL_QUADPOINTS))

    # Step 1: Initialize coils
    coils, _fix_shapes = _initialize_coils_for_optimization(
        s,
        target_B,
        out_dir,
        ncoils,
        order,
        numquadpoints,
        coil_width,
        regularization,
        initial_coils,
        is_continuation_step,
        kwargs,
    )

    total_current = compute_total_current(coils, ncoils)

    # Scale force/torque thresholds by (I_device / I_reactor)^2
    current_scale_factor = 1.0
    if not is_continuation_step:
        with suppress_output():
            coils_backup = initialize_coils_loop(
                s,
                out_dir=out_dir,
                ncoils=ncoils,
                order=order,
                numquadpoints=numquadpoints,
                coil_width=coil_width,
                regularization=regularization,
                **_initialization_kwargs(kwargs),
            )
        total_current_reactor_scale = sum(
            [c.current.get_value() for c in coils_backup[:ncoils]]
        )
        current_scale_factor = (total_current / total_current_reactor_scale) ** 2
        force_threshold *= current_scale_factor
        torque_threshold *= current_scale_factor

    base_curves = [coil.curve for coil in coils[:ncoils]]
    initial_geometry_metrics = compute_initial_geometry_metrics(
        s=s,
        coils=coils,
        base_curves=base_curves,
        ncoils=ncoils,
        initialization_metadata=kwargs.get("_initialization_metadata", {}),
    )

    # Step 2: Create plotting surface
    s_plot, qphi, qtheta = _create_plotting_surface(s, surface_resolution, kwargs)

    # Step 3: Create BiotSavart and save initial state
    save_coils_surface_vtk = kwargs.get("save_coils_surface_vtk", True)
    save_initial_state = kwargs.get("save_initial_state", True)
    finite_section_config = parse_finite_section_field_config(
        kwargs.get("finite_section_field")
    )
    with timed_section("biotsavart_setup"):
        coils_for_bs = (
            build_finite_section_coil_bundle(coils, finite_section_config)
            if finite_section_config.enabled
            else coils
        )
        bs, curves, B_initial = _setup_biotSavart_and_initial_save(
            coils_for_bs,
            s,
            s_plot,
            qphi,
            qtheta,
            out_dir,
            save_coils_surface_vtk=save_coils_surface_vtk,
            save_initial_state=save_initial_state,
        )
        if finite_section_config.enabled:
            # Keep engineering constraints on the optimized centerline coils.
            curves = [coil.curve for coil in coils]
            proc0_print(
                "[finite-section] BiotSavart uses "
                f"{len(coils_for_bs)} bundle filaments from {len(coils)} centerline coils"
            )

    if (
        coil_objective_terms
        and coil_objective_terms.get("structural_animation_vtk")
        and is_proc0()
    ):
        root = Path(
            kwargs.get("structural_animation_vtk_root", str(out_dir))
        ).resolve()
        sub = str(coil_objective_terms.get("structural_animation_subdir") or "vtk_frames")
        frames_dir = (root / sub).resolve()
        frames_dir.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("_structural_animation_frame_counter", [0])

        def _surface_snap(idx: int) -> None:
            # compute_bdotn_point_data sets bs to s_plot; SquaredFlux needs bs on s.
            try:
                stem = frames_dir / f"snapshot_surface_{idx:06d}"
                point_data = _compute_surface_vtk_data(
                    bs, s_plot, qphi, qtheta, include_bn=True
                )
                s_plot.to_vtk(str(stem), extra_data=point_data)
            finally:
                bs.set_points(s.gamma().reshape((-1, 3)))

        kwargs["_structural_animation_frames_dir"] = frames_dir
        kwargs["_structural_animation_surface_snap"] = _surface_snap
        proc0_print(
            f"[structural_animation] Writing paired VTK to {frames_dir} "
            "(on each full structural stress evaluation)"
        )

    # Step 4: Build objectives and constraints
    obj_result = _build_objectives_and_constraints(
        s,
        bs,
        coils,
        coils_for_bs,
        base_curves,
        curves,
        ncoils,
        total_current,
        major_radius,
        coil_objective_terms,
        th,
        kwargs,
        out_dir=out_dir,
    )
    Jf = obj_result["Jf"]
    c_list = obj_result["c_list"]
    constraint_scaling = obj_result["constraint_scaling"]
    cc_distance_index = obj_result["cc_distance_index"]
    cs_distance_index = obj_result["cs_distance_index"]
    constraint_names_and_thresholds = obj_result["constraint_names_and_thresholds"]
    constraint_idx_to_term = obj_result["constraint_idx_to_term"]
    Jls = obj_result["Jls"]
    Jccdist = obj_result["Jccdist"]
    Jcsdist = obj_result["Jcsdist"]
    Jlink = obj_result["Jlink"]
    effective_obj_terms = obj_result.get("effective_obj_terms", {})
    # Only pass Jforce/Jtorque for iteration display when in objectives (saves .J() calls)
    Jforce = (
        obj_result["Jforce"] if effective_obj_terms.get("coil_coil_force") else None
    )
    Jtorque = (
        obj_result["Jtorque"] if effective_obj_terms.get("coil_coil_torque") else None
    )
    Jts = obj_result.get("Jts")
    structural_obj = obj_result.get("structural_obj")
    structural_obj_raw = obj_result.get("structural_obj_raw")

    ctx = OptimizationLoopContext(
        algorithm=algorithm,
        algorithm_options=algorithm_options,
        max_iterations=max_iterations,
        max_iter_subopt=max_iter_subopt,
        verbose=verbose,
        th=th,
        coil_width=coil_width,
        ncoils=ncoils,
        total_current=total_current,
        target_B=target_B,
        flux_threshold=flux_threshold,
        cc_threshold=cc_threshold,
        cs_threshold=cs_threshold,
        length_threshold=length_threshold,
        curvature_threshold=curvature_threshold,
        msc_threshold=msc_threshold,
        force_threshold=force_threshold,
        torque_threshold=torque_threshold,
        arclength_variation_threshold=arclength_variation_threshold,
        major_radius=major_radius,
        coils=coils,
        c_list=c_list,
        constraint_scaling=constraint_scaling,
        constraint_idx_to_term=constraint_idx_to_term,
        constraint_names_and_thresholds=constraint_names_and_thresholds,
        cc_distance_index=cc_distance_index,
        cs_distance_index=cs_distance_index,
        base_curves=base_curves,
        Jf=Jf,
        Jls=Jls,
        Jccdist=Jccdist,
        Jcsdist=Jcsdist,
        Jlink=Jlink,
        Jforce=Jforce,
        Jtorque=Jtorque,
        Jts=Jts,
        coil_objective_terms=coil_objective_terms,
        kwargs=kwargs,
        structural_obj=structural_obj,
        effective_obj_terms=effective_obj_terms,
        surface=s,
        out_dir=out_dir,
        history_interval=kwargs.get("history_interval"),
        history_output_dir=Path(kwargs["history_output_dir"]).resolve()
        if kwargs.get("history_output_dir")
        else None,
    )

    _print_optimization_setup_summary(ctx)

    # Step 6: Run optimization (or worker loop on non-zero ranks when MPI+structural)
    structural_mpi = (
        is_mpi_enabled() and comm_world.size > 1 and structural_obj_raw is not None
    )
    if structural_mpi and not is_proc0():
        _structural_dj_worker_loop(structural_obj_raw)
        _bcast_coil_dofs_to_workers(base_curves, coils)
        return coils, {}
    if structural_mpi and is_proc0():
        opt_result, iterations_used, start_time, end_time = _run_optimization_step(ctx)
        ctrl = np.array([0, 0], dtype=np.int64)
        comm_world.Bcast(ctrl, root=0)
        _bcast_coil_dofs_to_workers(base_curves, coils)
    else:
        opt_result, iterations_used, start_time, end_time = _run_optimization_step(ctx)

    # Extract max von Mises [Pa] from structural objective's final evaluation.
    # Avoids redundant post-processing structural solve for correlation studies.
    structural_max_von_mises_Pa: float | None = None
    if structural_obj_raw is not None and is_proc0():
        try:
            from ._structural_objective import PA_TO_GPA

            vm_gpa = abs(structural_obj_raw.J())
            structural_max_von_mises_Pa = float(vm_gpa / PA_TO_GPA)
        except Exception:
            structural_max_von_mises_Pa = None

    if finite_section_config.enabled:
        from simsopt.field import BiotSavart
        from simsopt.objectives import SquaredFlux

        coils_for_bs = build_finite_section_coil_bundle(coils, finite_section_config)
        bs = BiotSavart(coils_for_bs)
        Jf = SquaredFlux(s, bs, threshold=flux_threshold)
        kwargs["finite_section_field_runtime"] = {
            "enabled": True,
            "width": finite_section_config.width,
            "height": finite_section_config.height,
            "n_width": finite_section_config.n_width,
            "n_height": finite_section_config.n_height,
            "n_filaments_per_coil": finite_section_config.n_filaments,
            "n_field_coils": len(coils_for_bs),
            "current_distribution": finite_section_config.current_distribution,
        }

    outcome = OptimizationOutcome(
        Jf=Jf,
        Jcsdist=Jcsdist,
        Jccdist=Jccdist,
        Jlink=Jlink,
        opt_result=opt_result,
        cached_thresholds=_build_cached_thresholds_dict(th),
        th=th,
        total_current=total_current,
        target_B=target_B,
        start_time=start_time,
        end_time=end_time,
        iterations_used=iterations_used,
        lag_mul=None,
        base_curves=base_curves,
        ncoils=ncoils,
        skip_post_processing=skip_post_processing,
        case_path=case_path,
        pp_flags=pp_flags,
        B_initial=B_initial,
        structural_max_von_mises_Pa=structural_max_von_mises_Pa,
        initial_geometry_metrics=initial_geometry_metrics,
    )
    coils_return, results_dict = _save_results_and_compute_metrics(
        s,
        s_plot,
        qphi,
        qtheta,
        bs,
        coils,
        out_dir,
        kwargs,
        outcome,
    )
    if "finite_section_field_runtime" in kwargs:
        results_dict["finite_section_field"] = kwargs["finite_section_field_runtime"]
    return coils_return, results_dict
