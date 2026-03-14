"""
Fourier continuation for coil optimization.

Provides progressive Fourier-order refinement for modular coils.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from ..config_scheme import PostProcessingConfig
from ..constants import DEFAULT_COIL_QUADPOINTS
from ..mpi_utils import proc0_print
from ..path_utils import coils_json_path_from_dir
from ._optimization_loop import _get_regularization_circ

from simsopt.geo import SurfaceRZFourier

regularization_circ = _get_regularization_circ()

logger = logging.getLogger(__name__)


def _extend_coils_to_higher_order(
    coils: list,
    new_order: int,
    s: SurfaceRZFourier,
    ncoils: int,
    regularization: Callable | None = None,
    coil_width: float = 0.4,
) -> list:
    """
    Extend coils from a lower Fourier order to a higher order.

    This function takes coils optimized at a lower order and extends them to
    a higher order by copying existing Fourier coefficients and padding new
    modes with zeros.

    Parameters
    ----------
    coils: list
        List of Coil objects from previous optimization (lower order).
    new_order: int
        Target Fourier order for the extended coils.
    s: SurfaceRZFourier
        Plasma surface (needed for creating new curves).
    ncoils: int
        Number of base coils.
    regularization: Callable, optional
        Regularization function for new coils.
    coil_width: float
        Coil width parameter.

    Returns
    -------
    list
        New list of Coil objects with extended Fourier order.
    """
    from simsopt.geo import create_equally_spaced_curves, CurveXYZFourier
    from simsopt.field import coils_via_symmetries

    # Get the old order from the first base curve
    old_curves = [coil.curve for coil in coils[:ncoils]]
    old_order = (
        old_curves[0].order
        if hasattr(old_curves[0], "order")
        else len(old_curves[0].dofs) // 3
    )

    if new_order <= old_order:
        return coils

    # Get major radius for creating new curves
    R0 = s.major_radius()
    R1 = s.get_rc(1, 0) * 3.5

    # Create new base curves with higher order
    new_base_curves = create_equally_spaced_curves(
        ncoils,
        s.nfp,
        stellsym=s.stellsym,
        R0=R0,
        R1=R1,
        order=new_order,
        numquadpoints=DEFAULT_COIL_QUADPOINTS,
    )

    # Copy Fourier coefficients from old curves to new curves
    for curve_idx, (old_curve, new_curve) in enumerate(
        zip(old_curves, new_base_curves)
    ):
        if isinstance(old_curve, CurveXYZFourier) and isinstance(
            new_curve, CurveXYZFourier
        ):
            old_dofs = old_curve.get_dofs()
            new_dofs = new_curve.get_dofs().copy()

            # Structure: For order N, each component has (2*N + 1) dofs:
            # - (N+1) cosine modes: indices 0 to N
            # - N sine modes: indices N+1 to 2*N
            # Components are stored as: [x_dofs, y_dofs, z_dofs]
            old_dofs_per_comp = 2 * old_order + 1
            new_dofs_per_comp = 2 * new_order + 1

            for comp_idx in range(3):
                old_start = comp_idx * old_dofs_per_comp
                new_start = comp_idx * new_dofs_per_comp

                for i in range(old_dofs_per_comp):
                    if old_start + i < len(old_dofs) and new_start + i < len(new_dofs):
                        new_dofs[new_start + i] = old_dofs[old_start + i]

            new_curve.set_dofs(new_dofs)
        else:
            try:
                old_dofs = old_curve.get_dofs()
                new_dofs = new_curve.get_dofs()
                if len(old_dofs) < len(new_dofs):
                    padded_dofs = np.zeros_like(new_dofs)
                    padded_dofs[: len(old_dofs)] = old_dofs
                    new_curve.set_dofs(padded_dofs)
                else:
                    new_curve.set_dofs(old_dofs[: len(new_dofs)])
            except (AttributeError, TypeError) as exc:
                logger.debug("Could not transfer DOFs for curve %d: %s", curve_idx, exc)

    # Extract currents from old coils
    base_currents = [coil.current for coil in coils[:ncoils]]

    # Create new coils with extended curves
    if regularization is not None:
        regularizations = [regularization(coil_width) for _ in range(ncoils)]
    else:
        regularizations = None

    try:
        new_coils = coils_via_symmetries(
            new_base_curves,
            base_currents,
            s.nfp,
            s.stellsym,
            regularizations=regularizations,
        )
    except TypeError:
        new_coils = coils_via_symmetries(
            new_base_curves, base_currents, s.nfp, s.stellsym
        )

    return new_coils


def optimize_coils_with_fourier_continuation(
    s: SurfaceRZFourier,
    fourier_orders: list[int],
    target_B: float = 5.7,
    out_dir: Path | str = "",
    max_iterations: int = 30,
    ncoils: int = 4,
    verbose: bool = False,
    regularization: Callable | None = regularization_circ,
    coil_objective_terms: Dict[str, Any] | None = None,
    surface_resolution: int = 32,
    case_path: Path | None = None,
    skip_post_processing: bool = False,
    pp_flags: PostProcessingConfig | None = None,
    **kwargs,
) -> tuple[list, Dict[str, Any]]:
    """
    Perform coil optimization with Fourier continuation.

    This function solves a sequence of coil optimizations, starting with a low
    number of Fourier modes, converging that problem, and using the solution
    as an initial condition for the next optimization with more Fourier modes.

    Parameters
    ----------
    s: SurfaceRZFourier
        Plasma boundary surface.
    fourier_orders: list[int]
        Sequence of Fourier orders to use (e.g., [4, 6, 8]).
        Must be in ascending order.
    target_B: float
        Target magnetic field strength in Tesla (default: 5.7).
    out_dir: Path | str
        Output directory for saved files.
    case_path: Path, optional
        Path to case directory containing case.yaml. Used for post-processing.
    max_iterations: int
        Maximum number of optimization iterations per order (default: 30).
    ncoils: int
        Number of base coils to create (default: 4).
    verbose: bool
        Print out progress and results (default: False).
    regularization: Callable
        Regularization function (default: regularization_circ).
    coil_objective_terms: Dict[str, Any] | None
        Dictionary specifying which objective terms to include.
    surface_resolution: int
        Resolution of plasma surface (nphi=ntheta) for evaluation (default: 32).
    skip_post_processing: bool
        If True, skip post-processing after optimization (default: False).
    pp_flags : PostProcessingConfig | None
        Bundled post-processing flags.  When *None*, defaults are used.
    **kwargs
        Same as optimize_coils_loop (thresholds, algorithm, plot_upsample_factor, etc.).

    Returns
    -------
    tuple[list, Dict[str, Any]]
        Final optimized coils and combined results dictionary.
    """
    from ._optimization_loop import optimize_coils_loop
    from .optimization import _merge_post_processing_into_results
    from ._post_opt_processing import _run_post_processing_after_optimization

    if not fourier_orders:
        raise ValueError("fourier_orders must be a non-empty list")

    if not all(isinstance(o, int) and o > 0 for o in fourier_orders):
        raise ValueError("All fourier_orders must be positive integers")

    if fourier_orders != sorted(fourier_orders):
        raise ValueError("fourier_orders must be in ascending order")

    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    coils: list | None = None
    coil_width = kwargs.get("coil_width", 0.4)
    cached_thresholds: Dict[str, Any] = {}

    proc0_print(f"Starting Fourier continuation with orders: {fourier_orders}")

    for i, order in enumerate(fourier_orders):
        proc0_print(f"\n{'=' * 60}")
        proc0_print(
            f"Fourier continuation step {i + 1}/{len(fourier_orders)}: order={order}"
        )
        proc0_print(f"{'=' * 60}")

        order_dir = out_dir_path / f"order_{order}"
        order_dir.mkdir(exist_ok=True)

        fc_kwargs = {**kwargs, "save_coils_surface_vtk": True}
        if i > 0:
            fc_kwargs["save_initial_state"] = False

        if i == 0:
            proc0_print(f"Initializing coils with order={order}...")
            coils, results = optimize_coils_loop(
                s=s,
                target_B=target_B,
                out_dir=str(order_dir),
                max_iterations=max_iterations,
                ncoils=ncoils,
                order=order,
                verbose=verbose,
                regularization=regularization,
                coil_objective_terms=coil_objective_terms,
                surface_resolution=surface_resolution,
                skip_post_processing=True,
                **fc_kwargs,
            )
            cached_thresholds = results.get("_cached_thresholds", {})
        else:
            if coils is None:
                raise RuntimeError(
                    "Cannot extend coils: previous step produced None coils"
                )
            proc0_print(
                f"Extending coils from order={fourier_orders[i - 1]} to order={order}..."
            )
            coils = _extend_coils_to_higher_order(
                coils, order, s, ncoils, regularization, coil_width
            )

            continuation_kwargs = {
                **kwargs,
                "save_coils_surface_vtk": True,
                "save_initial_state": False,
            }
            if cached_thresholds:
                continuation_kwargs["_cached_thresholds"] = cached_thresholds

            proc0_print(f"Optimizing with extended coils (order={order})...")
            coils, results = optimize_coils_loop(
                s=s,
                target_B=target_B,
                out_dir=str(order_dir),
                max_iterations=max_iterations,
                ncoils=ncoils,
                order=order,
                verbose=verbose,
                regularization=regularization,
                coil_objective_terms=coil_objective_terms,
                initial_coils=coils,
                surface_resolution=surface_resolution,
                skip_post_processing=True,
                **continuation_kwargs,
            )

        results["fourier_order"] = order
        results["continuation_step"] = i + 1
        all_results.append(results)

    combined_results = {
        "fourier_continuation": True,
        "fourier_orders": fourier_orders,
        "final_order": fourier_orders[-1],
        "continuation_results": all_results,
        **all_results[-1],
    }

    proc0_print(f"\n{'=' * 60}")
    proc0_print("Fourier continuation completed!")
    proc0_print(f"Final order: {fourier_orders[-1]}")
    proc0_print(f"{'=' * 60}\n")

    if coils is None:
        raise RuntimeError("Fourier continuation failed: no coils were produced")

    post_processing_results: Dict[str, Any] = {}
    if not skip_post_processing:
        final_order_dir = out_dir_path / f"order_{fourier_orders[-1]}"
        coils_json_path = coils_json_path_from_dir(final_order_dir)

        post_processing_results = _run_post_processing_after_optimization(
            out_dir=out_dir_path,
            s=s,
            case_path=Path(case_path) if case_path is not None else None,
            pp_flags=pp_flags if pp_flags is not None else PostProcessingConfig(),
            coils_json_path=coils_json_path,
        )

    if post_processing_results:
        _merge_post_processing_into_results(combined_results, post_processing_results)

    return coils, combined_results
