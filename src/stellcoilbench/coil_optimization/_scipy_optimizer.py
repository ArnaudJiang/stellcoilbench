"""
Scipy-based optimizer helpers for StellCoilBench coil optimization.

Provides configuration parsing, scipy.optimize.minimize option building,
Taylor test, weight building for weighted-sum objectives, and runners for
both scipy algorithms (L-BFGS-B, BFGS, SLSQP, etc.) and augmented Lagrangian.
"""

from __future__ import annotations

import sys
import csv
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple
from pathlib import Path

import numpy as np

from ..constants import (
    CI_MAX_ITER_CAP,
    DEFAULT_DISTANCE_CONSTRAINT_WEIGHT,
    DEFAULT_FLUX_WEIGHT,
    FTOL_DEFAULT,
    GTOL_DEFAULT_BFGS,
    GTOL_DEFAULT_LBFGSB,
    LBFGSB_DEFAULT_MAXCOR,
    LBFGSB_DEFAULT_MAXLS,
    LBFGSB_MAXFUN_MULTIPLIER,
    NUMERICAL_FLOOR,
    TAYLOR_TEST_EPSILONS,
    TAYLOR_TEST_ERROR_RATIO_THRESHOLD,
    TAYLOR_TEST_SEED,
    TNC_DEFAULT_FTOL,
    TOL_DEFAULT,
    VERBOSE_ITERATION_INTERVAL,
)
from ..mpi_utils import proc0_print
from ._cs_guard import CoilSurfaceDistanceGuard
from ._early_stop import EarlyStopController, EarlyStopTriggered
from ._link_guard import PairwiseLinkGuard
from ._thresholds import get_full_thresholds

if TYPE_CHECKING:
    from simsopt.geo import SurfaceRZFourier


class OptimizationHistoryRecorder:
    """Write lightweight optimizer diagnostics to CSV at a fixed objective-call interval."""

    def __init__(
        self,
        output_dir: str | Path,
        interval: int,
        *,
        constraint_names_and_thresholds: list | None = None,
        weights: list | None = None,
        base_curves: list | None = None,
        Jccdist: Any | None = None,
        Jcsdist: Any | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.interval = max(1, int(interval))
        self.constraint_names_and_thresholds = constraint_names_and_thresholds or []
        self.weights = weights or []
        self.base_curves = base_curves or []
        self.Jccdist = Jccdist
        self.Jcsdist = Jcsdist
        self.objective_path = self.output_dir / "objective_history.csv"
        self.constraint_path = self.output_dir / "constraint_history.csv"
        self._initialized = False

    def reset(self) -> None:
        """Clear existing history files and rewrite headers on the next record."""
        for path in (self.objective_path, self.constraint_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self._initialized = False

    def should_record(self, iteration: int) -> bool:
        return iteration == 1 or iteration % self.interval == 0

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.objective_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iteration",
                    "objective",
                    "max_length",
                    "avg_length",
                    "min_length",
                    "length_std",
                    "length_cv",
                    "length_ratio",
                    "total_length",
                    "max_curvature",
                    "mean_squared_curvature",
                    "cc_shortest_distance",
                    "cs_shortest_distance",
                ],
            )
            writer.writeheader()
        with self.constraint_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iteration",
                    "constraint_index",
                    "constraint_name",
                    "threshold",
                    "value",
                    "weight",
                    "weighted_value",
                ],
            )
            writer.writeheader()
        self._initialized = True

    def record(self, iteration: int, objective: float, c_list: list) -> None:
        self._ensure_initialized()
        geometry = self._geometry_metrics()
        with self.objective_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iteration",
                    "objective",
                    "max_length",
                    "avg_length",
                    "min_length",
                    "length_std",
                    "length_cv",
                    "length_ratio",
                    "total_length",
                    "max_curvature",
                    "mean_squared_curvature",
                    "cc_shortest_distance",
                    "cs_shortest_distance",
                ],
            )
            writer.writerow({"iteration": iteration, "objective": objective, **geometry})

        name_threshold = {
            i: pair for i, pair in enumerate(self.constraint_names_and_thresholds)
        }
        with self.constraint_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iteration",
                    "constraint_index",
                    "constraint_name",
                    "threshold",
                    "value",
                    "weight",
                    "weighted_value",
                ],
            )
            for idx, constraint in enumerate(c_list):
                name, threshold = name_threshold.get(idx, (f"constraint_{idx}", ""))
                value = self._safe_objective_value(constraint)
                weight = self.weights[idx] if idx < len(self.weights) else 1.0
                writer.writerow(
                    {
                        "iteration": iteration,
                        "constraint_index": idx,
                        "constraint_name": name,
                        "threshold": threshold,
                        "value": value,
                        "weight": weight,
                        "weighted_value": value * float(weight)
                        if value is not None
                        else "",
                    }
                )

    def _geometry_metrics(self) -> dict[str, float | str]:
        cc_shortest = self._safe_shortest_distance(self.Jccdist)
        cs_shortest = self._safe_shortest_distance(self.Jcsdist)
        if not self.base_curves:
            return {
                "max_length": "",
                "avg_length": "",
                "min_length": "",
                "length_std": "",
                "length_cv": "",
                "length_ratio": "",
                "total_length": "",
                "max_curvature": "",
                "mean_squared_curvature": "",
                "cc_shortest_distance": cc_shortest if cc_shortest is not None else "",
                "cs_shortest_distance": cs_shortest if cs_shortest is not None else "",
            }
        try:
            from simsopt.geo import CurveLength

            lengths = [float(CurveLength(c).J()) for c in self.base_curves]
            mean_length = float(np.mean(lengths))
            length_std = float(np.std(lengths))
            min_length = min(lengths)
            kappas = [np.atleast_1d(c.kappa()).astype(float) for c in self.base_curves]
            return {
                "max_length": max(lengths),
                "avg_length": mean_length,
                "min_length": min_length,
                "length_std": length_std,
                "length_cv": length_std / mean_length if mean_length > 0 else "",
                "length_ratio": max(lengths) / min_length if min_length > 0 else "",
                "total_length": float(np.sum(lengths)),
                "max_curvature": float(max(np.max(k) for k in kappas)),
                "mean_squared_curvature": float(max(np.mean(k**2) for k in kappas)),
                "cc_shortest_distance": cc_shortest if cc_shortest is not None else "",
                "cs_shortest_distance": cs_shortest if cs_shortest is not None else "",
            }
        except Exception:
            return {
                "max_length": "",
                "avg_length": "",
                "min_length": "",
                "length_std": "",
                "length_cv": "",
                "length_ratio": "",
                "total_length": "",
                "max_curvature": "",
                "mean_squared_curvature": "",
                "cc_shortest_distance": cc_shortest if cc_shortest is not None else "",
                "cs_shortest_distance": cs_shortest if cs_shortest is not None else "",
            }

    @staticmethod
    def _safe_objective_value(obj: Any) -> float | None:
        try:
            return float(obj.J())
        except Exception:
            return None

    @staticmethod
    def _safe_shortest_distance(obj: Any) -> float | None:
        try:
            return float(obj.shortest_distance())
        except Exception:
            return None


def _parse_optimizer_config(
    s: "SurfaceRZFourier",
    kwargs: Dict[str, Any],
    max_iterations: int,
    *,
    is_continuation_step: bool = False,
    default_algorithm: str = "augmented_lagrangian",
    coil_objective_terms: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Parse optimizer configuration from kwargs and surface.

    Clamps max_iterations to CI cap, resolves algorithm name (e.g. lbfgs -> L-BFGS-B),
    merges algorithm_options from kwargs, and builds full thresholds dict.

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface.
    kwargs : Dict[str, Any]
        User config: algorithm, algorithm_options, max_iter_subopt, thresholds, etc.
    max_iterations : int
        Requested maximum iterations (may be clamped).
    is_continuation_step : bool, optional
        Whether this is a Fourier continuation step.
    default_algorithm : str, optional
        Default algorithm if not specified (default: augmented_lagrangian).

    Returns
    -------
    Dict[str, Any]
        Keys: algorithm, algorithm_options, max_iter_subopt, max_iterations, thresholds.
    """
    if max_iterations > CI_MAX_ITER_CAP:
        proc0_print(
            f"Warning: max_iterations ({max_iterations}) exceeds CI cap ({CI_MAX_ITER_CAP}); clamping."
        )
        max_iterations = CI_MAX_ITER_CAP
    cached = kwargs.get("_cached_thresholds") if is_continuation_step else None
    thresholds = get_full_thresholds(
        s,
        kwargs,
        is_continuation_step=is_continuation_step,
        cached=cached,
        coil_objective_terms=coil_objective_terms,
    )
    max_iter_subopt = kwargs.get("max_iter_subopt", 10)
    algorithm = kwargs.get("algorithm", default_algorithm)
    if isinstance(algorithm, str):
        al = algorithm.lower()
        if al in ["l-bfgs", "lbfgs", "l-bfgs-b"]:
            algorithm = "L-BFGS-B"
        elif al == "augmented_lagrangian":
            algorithm = "augmented_lagrangian"
    algorithm_options = kwargs.get("algorithm_options", {}).copy()
    valid_opts = _get_scipy_algorithm_options(algorithm)
    for opt in valid_opts:
        if opt in kwargs and opt not in algorithm_options:
            algorithm_options[opt] = kwargs[opt]
    return {
        "algorithm": algorithm,
        "algorithm_options": algorithm_options,
        "max_iter_subopt": max_iter_subopt,
        "max_iterations": max_iterations,
        "thresholds": thresholds,
    }


def _build_scipy_minimize_options(
    algorithm: str,
    max_iterations: int,
    algorithm_options: Dict[str, Any],
    max_iter_subopt: int | None = None,
) -> Dict[str, Any]:
    """
    Build options dict for scipy.optimize.minimize.

    Sets algorithm-specific defaults (ftol, gtol, maxfun) and merges user
    algorithm_options. Validates options against _get_scipy_algorithm_options.

    Parameters
    ----------
    algorithm : str
        Algorithm name (e.g. L-BFGS-B, BFGS, SLSQP).
    max_iterations : int
        Maximum iterations for outer loop.
    algorithm_options : Dict[str, Any]
        User-provided options to merge.
    max_iter_subopt : int | None, optional
        Unused; kept for API compatibility.

    Returns
    -------
    Dict[str, Any]
        Options dict suitable for scipy.optimize.minimize(..., options=...).
    """
    options = {"maxiter": max_iterations}
    if algorithm == "L-BFGS-B":
        options.setdefault("ftol", FTOL_DEFAULT)
        options.setdefault("gtol", GTOL_DEFAULT_LBFGSB)
        options.setdefault("maxcor", LBFGSB_DEFAULT_MAXCOR)
        # Increase maxls to reduce ABNORMAL_TERMINATION_IN_LNSRCH (line search failure)
        # when optimizing dipole currents or other poorly conditioned problems
        options.setdefault("maxls", LBFGSB_DEFAULT_MAXLS)
    elif algorithm == "TNC":
        options.setdefault("ftol", TNC_DEFAULT_FTOL)
        options.setdefault("gtol", GTOL_DEFAULT_BFGS)
    elif algorithm == "COBYLA":
        options.setdefault("tol", TOL_DEFAULT)
    if algorithm in ["L-BFGS-B", "TNC"]:
        if "maxfun" not in options:
            options["maxfun"] = max_iterations * LBFGSB_MAXFUN_MULTIPLIER
    if algorithm_options:
        _validate_algorithm_options(algorithm, algorithm_options)
        options.update(algorithm_options)
    return options


def _run_taylor_test(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    verbose: bool = False,
) -> bool:
    """
    Run Taylor test to verify gradient correctness.

    Checks that J(x0 + εh) ≈ J(x0) + ε * ∇J · h for decreasing ε.
    Fails if error ratio between successive ε is > 0.6 (gradient inconsistent).

    Parameters
    ----------
    objective : Callable
        Scalar objective J(x).
    gradient : Callable
        Gradient function returning ∇J(x).
    x0 : np.ndarray
        Point at which to test.
    verbose : bool, optional
        If True, print success message when test passes.

    Returns
    -------
    bool
        True if test passed, False otherwise.
    """
    np.random.seed(TAYLOR_TEST_SEED)
    h = np.random.randn(len(x0))
    h = h / np.linalg.norm(h)
    J0 = objective(x0)
    grad0 = gradient(x0)
    epsilons = list(TAYLOR_TEST_EPSILONS)
    errors = []
    for eps in epsilons:
        xp = x0 + eps * h
        Jp = objective(xp)
        Jpred = J0 + eps * np.dot(grad0, h)
        err = abs(Jp - Jpred) / (abs(J0) + NUMERICAL_FLOOR)
        errors.append(err)
    passed = True
    for i in range(len(errors) - 1):
        if (
            errors[i] > 0
            and errors[i + 1] / errors[i] > TAYLOR_TEST_ERROR_RATIO_THRESHOLD
        ):
            proc0_print(
                f"WARNING: Taylor test failed: error ratio {errors[i + 1] / errors[i]:.3f} > {TAYLOR_TEST_ERROR_RATIO_THRESHOLD} "
                f"(ε={epsilons[i]:.1e} -> {epsilons[i + 1]:.1e})",
                file=sys.stderr,
            )
            passed = False
    if passed and verbose:
        proc0_print("Taylor test passed: error decreases as expected")
    return passed


def _apply_distance_weights_for_auglag(
    c_list: list,
    constraint_scaling: Dict[int, float],
    cc_distance_idx: int | None,
    cs_distance_idx: int | None,
    kwargs: Dict[str, Any],
    extra_distance_indices: list[int] | None = None,
) -> None:
    """
    Apply weights to distance constraints for augmented Lagrangian (in-place).

    Replaces c_list[idx] with Weight(w) * c_list[idx] for each distance constraint
    index. Weight includes constraint_scaling for dimensionless objectives.
    extra_distance_indices supports dipole's second cc-distance constraint.

    Parameters
    ----------
    c_list : list
        List of constraint objectives (modified in-place).
    constraint_scaling : Dict[int, float]
        Scaling factors per constraint index.
    cc_distance_idx, cs_distance_idx : int | None
        Indices of coil-coil and coil-surface distance constraints.
    kwargs : Dict[str, Any]
        May contain constraint_weight_{idx} overrides.
    extra_distance_indices : list[int] | None, optional
        Additional distance indices (e.g. dipole cc_dist2).
    """
    from simsopt.objectives import Weight

    indices = [i for i in [cs_distance_idx, cc_distance_idx] if i is not None]
    if extra_distance_indices:
        indices.extend(extra_distance_indices)
    for idx in indices:
        w = kwargs.get(f"constraint_weight_{idx}", DEFAULT_DISTANCE_CONSTRAINT_WEIGHT)
        if idx in constraint_scaling:
            w *= constraint_scaling[idx]
        c_list[idx] = Weight(w) * c_list[idx]


def _build_weights_for_scipy_minimize(
    c_list: list,
    constraint_scaling: Dict[int, float],
    constraint_idx_to_term: Dict[int, str],
    cc_distance_idx: int | None,
    cs_distance_idx: int | None,
    kwargs: Dict[str, Any],
    coil_objective_terms: Dict[str, Any] | None,
) -> list:
    """
    Build weights list for weighted objective JF = sum(Weight(w)*c).

    Flux (index 0) gets flux_weight or 1.0. Other constraints get weights from
    coil_objective_terms (e.g. length_weight, cc_weight) or kwargs
    (constraint_weight_{i}). Distance constraints default to 1e3 if unspecified.
    Applies constraint_scaling for dimensionless objectives.

    Parameters
    ----------
    c_list : list
        Constraint objectives (flux first, then distance, length, etc.).
    constraint_scaling : Dict[int, float]
        Scaling per constraint index.
    constraint_idx_to_term : Dict[int, str]
        Maps constraint index to term name.
    cc_distance_idx, cs_distance_idx : int | None
        Indices of coil-coil and coil-surface distance constraints.
    kwargs : Dict[str, Any]
        constraint_weight_{i} overrides.
    coil_objective_terms : Dict[str, Any] | None
        Case config with flux_weight, length_weight, cc_weight, etc.

    Returns
    -------
    list
        List of float weights, one per constraint in c_list.
    """
    term_to_weight_key = {
        "total_length": "length_weight",
        "coil_length_variance": "length_variance_weight",
        "coil_coil_distance": "cc_weight",
        "coil_surface_distance": "cs_weight",
        "coil_curvature": "curvature_weight",
        "coil_torsion": "torsion_weight",
        "coil_arclength_variation": "arclength_variation_weight",
        "coil_mean_squared_curvature": "msc_weight",
        "coil_coil_force": "force_weight",
        "coil_coil_torque": "torque_weight",
        "linking_number": "linking_weight",
        "structural_stress": "structural_stress_weight",
    }
    cs_weight_specified = (
        cs_distance_idx is not None and f"constraint_weight_{cs_distance_idx}" in kwargs
    )
    cc_weight_specified = (
        cc_distance_idx is not None and f"constraint_weight_{cc_distance_idx}" in kwargs
    )

    weights = []
    for i, _ in enumerate(c_list):
        if i == 0:
            if coil_objective_terms and "flux_weight" in coil_objective_terms:
                weights.append(float(coil_objective_terms["flux_weight"]))
            else:
                weights.append(DEFAULT_FLUX_WEIGHT)
        else:
            weight_specified = f"constraint_weight_{i}" in kwargs
            weight = kwargs.get(f"constraint_weight_{i}", 1.0)
            term_name = constraint_idx_to_term.get(i)
            if term_name and coil_objective_terms:
                weight_param = term_to_weight_key.get(term_name)
                if weight_param and weight_param in coil_objective_terms:
                    weight = float(coil_objective_terms[weight_param])
                    weight_specified = True
            if cs_distance_idx is not None and i == cs_distance_idx:
                if coil_objective_terms and "cs_weight" in coil_objective_terms:
                    weight = float(coil_objective_terms["cs_weight"])
                    weight_specified = True
                elif cs_weight_specified:
                    weight = kwargs[f"constraint_weight_{i}"]
                else:
                    weight = kwargs.get(
                        f"constraint_weight_{i}", DEFAULT_DISTANCE_CONSTRAINT_WEIGHT
                    )
            elif cc_distance_idx is not None and i == cc_distance_idx:
                if coil_objective_terms and "cc_weight" in coil_objective_terms:
                    weight = float(coil_objective_terms["cc_weight"])
                    weight_specified = True
                elif cc_weight_specified:
                    weight = kwargs[f"constraint_weight_{i}"]
                else:
                    weight = kwargs.get(
                        f"constraint_weight_{i}", DEFAULT_DISTANCE_CONSTRAINT_WEIGHT
                    )
            if i in constraint_scaling:
                dist_indices = [
                    x for x in [cc_distance_idx, cs_distance_idx] if x is not None
                ]
                if i in dist_indices:
                    weight *= constraint_scaling[i]
                elif not weight_specified:
                    weight *= constraint_scaling[i]
            weights.append(weight)
    return weights


def _filter_optimizer_kwargs(
    optimizer_params: dict[str, Any],
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    """Return *optimizer_params* with the named keys removed.

    Parameters
    ----------
    optimizer_params : dict
        Raw optimizer params.
    exclude : set[str] or None
        Keys to drop.  Defaults to ``{"max_iterations", "verbose"}``.

    Returns
    -------
    dict[str, Any]
    """
    if exclude is None:
        exclude = {"max_iterations", "verbose"}
    return {k: v for k, v in optimizer_params.items() if k not in exclude}


def _get_scipy_algorithm_options(algorithm: str) -> Dict[str, List[type]]:
    """
    Get valid options for a given scipy optimization algorithm.

    Returns a dictionary mapping option names to their valid types/values.
    Based on scipy.optimize.minimize documentation.

    Parameters
    ----------
    algorithm: str
        The name of the scipy optimization algorithm.

    Returns
    -------
    Dict[str, list]
        A dictionary mapping option names to their valid types/values.
    """
    # Common options for most algorithms
    common_options = {
        "maxiter": [int],
        "disp": [bool],
    }

    # Algorithm-specific options
    algorithm_specific = {
        "BFGS": {
            "gtol": [float],
            "norm": [float],
        },
        "L-BFGS-B": {
            "maxfun": [int],
            "maxcor": [int],
            "ftol": [float],
            "gtol": [float],
            "eps": [float],
            "maxls": [int],
        },
        "SLSQP": {
            "ftol": [float],
            "eps": [float],
        },
        "Nelder-Mead": {
            "xatol": [float],
            "fatol": [float],
            "adaptive": [bool],
        },
        "Powell": {
            "xtol": [float],
            "ftol": [float],
            "maxfev": [int],
        },
        "CG": {
            "gtol": [float],
            "norm": [float],
        },
        "Newton-CG": {
            "xtol": [float],
            "eps": [float],
        },
        "TNC": {
            "maxfun": [int],
            "ftol": [float],
            "gtol": [float],
            "eps": [float],
        },
        "COBYLA": {
            "maxiter": [int],
            "rhobeg": [float],
            "tol": [float],
        },
        "trust-constr": {
            "xtol": [float],
            "gtol": [float],
            "barrier_tol": [float],
            "initial_barrier_parameter": [float],
            "initial_barrier_tolerance": [float],
            "initial_trust_radius": [float],
            "max_trust_radius": [float],
        },
    }

    # Combine common and algorithm-specific options
    options = common_options.copy()
    if algorithm in algorithm_specific:
        options.update(algorithm_specific[algorithm])

    return options


def _validate_algorithm_options(algorithm: str, options: Dict[str, Any]) -> None:
    """
    Validate that algorithm-specific options are valid for the given algorithm.

    Raises ValueError if invalid options are found.

    Parameters
    ----------
    algorithm: str
        The name of the scipy optimization algorithm.
    options: Dict[str, Any]
        A dictionary of algorithm-specific options to validate.

    Raises
    ------
    ValueError: If invalid options are found.
    """
    valid_options = _get_scipy_algorithm_options(algorithm)

    invalid_options = []
    for option_name, option_value in options.items():
        if option_name not in valid_options:
            invalid_options.append(option_name)
        else:
            # Check type
            valid_types = valid_options[option_name]
            if not any(isinstance(option_value, t) for t in valid_types):
                invalid_options.append(
                    f"{option_name} (wrong type: {type(option_value).__name__})"
                )

    if invalid_options:
        valid_option_names = ", ".join(sorted(valid_options.keys()))
        raise ValueError(
            f"Invalid algorithm options for '{algorithm}': {', '.join(invalid_options)}. "
            f"Valid options are: {valid_option_names}"
        )


def _run_augmented_lagrangian(
    c_list: list,
    max_iterations: int,
    max_iter_subopt: int,
    verbose: bool,
    kwargs: Dict[str, Any],
) -> None:
    r"""Run simsopt augmented Lagrangian optimization.

    Minimizes the augmented Lagrangian:

    .. math::
        L(\mathbf{x},\boldsymbol{\lambda},\mu) = f(\mathbf{x})
        + \sum_i \lambda_i c_i(\mathbf{x})
        + \frac{\mu}{2} \sum_i c_i(\mathbf{x})^2

    Treats all *c_list* entries as equality constraints. Supports
    ``mu_init``, ``tau``, ``minimize_method`` from *kwargs*.

    Parameters
    ----------
    c_list : list
        Constraint objectives (equality constraints).
    max_iterations : int
        Maximum outer iterations.
    max_iter_subopt : int
        Maximum inner (L-BFGS-B) iterations per outer step.
    verbose : bool
        Print progress.
    kwargs : Dict[str, Any]
        Optional mu_init, tau, minimize_method for augmented Lagrangian.
    """
    try:
        from simsopt.solve import augmented_lagrangian_method
    except ImportError as exc:
        import logging

        logger = logging.getLogger(__name__)
        logger.debug("simsopt.solve top-level import failed, trying submodule: %s", exc)
        from simsopt.solve.augmented_lagrangian import augmented_lagrangian_method
    import inspect

    _alm_sig = inspect.signature(augmented_lagrangian_method)
    _alm_params = set(_alm_sig.parameters.keys())
    opts = {
        "MAXITER": max_iterations,
        "MAXITER_lag": max_iter_subopt,
        "verbose": verbose,
    }
    if "mu_init" in kwargs:
        opts["mu_init"] = kwargs["mu_init"]
    if "tau" in kwargs:
        opts["tau"] = kwargs["tau"]
    if "minimize_method" in kwargs:
        opts["minimize_method"] = kwargs["minimize_method"]
    opts = {k: v for k, v in opts.items() if k in _alm_params}
    augmented_lagrangian_method(f=None, equality_constraints=c_list, **opts)


def _compute_max_force_torque(
    coils: list, ncoils: int
) -> tuple[float | None, float | None]:
    """Max pointwise force [N/m] and torque per unit length [N] across base coils."""
    try:
        from simsopt.field.force import coil_force, coil_torque

        max_force = max(
            float(np.max(np.linalg.norm(coil_force(c, coils), axis=1)))
            for c in coils[:ncoils]
        )
        max_torque = max(
            float(np.max(np.linalg.norm(coil_torque(c, coils), axis=1)))
            for c in coils[:ncoils]
        )
        return max_force, max_torque
    except Exception:
        return None, None


def _build_objective_and_gradient(
    c_list: list,
    weights: list,
    constraint_names_and_thresholds: list,
    base_curves: list,
    Jls: list,
    Jccdist: Any,
    Jcsdist: Any,
    Jlink: Any,
    verbose: bool,
    *,
    coils: list | None = None,
    ncoils: int | None = None,
    show_force: bool = False,
    show_torque: bool = False,
    Jts: list | None = None,
    structural_obj: Any | None = None,
    out_dir: Any = None,
    history_recorder: OptimizationHistoryRecorder | None = None,
    link_guard: PairwiseLinkGuard | None = None,
    cs_guard: CoilSurfaceDistanceGuard | None = None,
    early_stop: EarlyStopController | None = None,
) -> Tuple[
    Callable[[np.ndarray], float],
    Callable[[np.ndarray], np.ndarray],
    Any,
    np.ndarray,
]:
    """
    Build weighted objective JF and objective/gradient callables for scipy minimize.

    Creates JF = sum(Weight(w)*c) and defines objective(x) and gradient(x) with
    verbose iteration output. Returns callables plus JF and x0 for Taylor test
    and optimization.

    Parameters
    ----------
    c_list : list
        Constraint objectives (flux first, then distance, length, etc.).
    weights : list
        Per-constraint weights from _build_weights_for_scipy_minimize.
    constraint_names_and_thresholds : list
        (name, threshold) pairs for verbose output.
    base_curves, Jls, Jccdist, Jcsdist, Jlink : objectives
        Constraint objectives for verbose output.
    verbose : bool
        Print iteration progress.
    Jts : list | None, optional
        Coil torsion objectives for verbose output.
    structural_obj : Any | None, optional
        Structural objective for diagnostic output.

    Returns
    -------
    tuple
        (objective_func, grad_func, JF, x0).
    """
    from simsopt.objectives import Weight

    from ._iteration_output import _format_verbose_iteration_output

    _STRUCTURAL_VERBOSE_CAP: int = 25
    JF = sum([Weight(w) * c for c, w in zip(c_list, weights)])
    iteration_count = [0]
    structural_thresh = next(
        (t for n, t in constraint_names_and_thresholds if n == "σ_vm"),
        None,
    )

    def objective(x: np.ndarray) -> float:
        JF.x = x  # type: ignore[attr-defined]
        J = JF.J()  # type: ignore[attr-defined]
        iteration_count[0] += 1
        if link_guard is not None:
            J += link_guard.evaluate(iteration_count[0], x=x, objective=float(J))
        if cs_guard is not None:
            J += cs_guard.evaluate(iteration_count[0], x=x, objective=float(J))
        if history_recorder is not None and history_recorder.should_record(
            iteration_count[0]
        ):
            history_recorder.record(iteration_count[0], float(J), c_list)
        if early_stop is not None:
            early_stop.maybe_check(iteration_count[0], float(J))
        _should_log = (
            iteration_count[0] == 1
            or (
                structural_obj is not None
                and iteration_count[0] <= _STRUCTURAL_VERBOSE_CAP
            )
            or iteration_count[0] % VERBOSE_ITERATION_INTERVAL == 0
        )
        if verbose and _should_log:
            grad = JF.dJ()  # type: ignore[attr-defined]
            max_force, max_torque = None, None
            if (show_force or show_torque) and coils is not None and ncoils is not None:
                max_force, max_torque = _compute_max_force_torque(coils, ncoils)
                if not show_force:
                    max_force = None
                if not show_torque:
                    max_torque = None
            main_line, contrib_line = _format_verbose_iteration_output(
                iteration_count[0],
                Jls,
                Jccdist,
                Jcsdist,
                base_curves,
                Jlink,
                grad,
                weights,
                c_list,
                constraint_names_and_thresholds,
                J,
                max_force=max_force,
                max_torque=max_torque,
                structural_obj=structural_obj,
                Jts=Jts,
            )
            proc0_print(main_line)
            proc0_print(contrib_line)
            if structural_obj is not None and structural_thresh is not None:
                # Use raw objective for true max von Mises (bypass guard for mesh refinement)
                raw_obj = None
                try:
                    raw_obj = structural_obj.objective.objective
                    sigma_vm = abs(raw_obj.J())
                except Exception:
                    sigma_vm = abs(
                        structural_obj.J()
                    )  # fallback to guarded value if FEM fails
                # Adaptive mesh refinement: coarse -> fine when stress approaches threshold
                if (
                    raw_obj is not None
                    and getattr(raw_obj, "_adaptive_mesh", False)
                    and not getattr(raw_obj, "_refinement_done", True)
                ):
                    ratio = getattr(raw_obj, "_refine_stress_ratio", 0.5)
                    if sigma_vm >= ratio * float(structural_thresh):
                        raw_obj.refine_mesh(raw_obj._mesh_resolution_fine)
        return J

    def gradient(x: np.ndarray) -> np.ndarray:
        JF.x = x  # type: ignore[attr-defined]
        return JF.dJ()  # type: ignore[attr-defined]

    x0 = JF.x.copy()  # type: ignore[attr-defined]
    return objective, gradient, JF, x0


def _invoke_taylor_test_for_modular_coils(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    JF: Any,
    verbose: bool,
) -> None:
    """
    Run Taylor test for gradient verification and reset JF.x to x0.

    Invokes the module-level Taylor test, then restores JF.x to x0 in case
    the test modified optimization state.

    Parameters
    ----------
    objective : Callable
        Scalar objective J(x).
    gradient : Callable
        Gradient function returning ∇J(x).
    x0 : np.ndarray
        Point at which to test (and restore).
    JF : Any
        Weighted objective with attribute x (reset to x0).
    verbose : bool
        Print Taylor test result.
    """
    _run_taylor_test(objective, gradient, x0, verbose=verbose)
    JF.x = x0  # type: ignore[attr-defined, assignment]


def _invoke_scipy_minimize(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    JF: Any,
    algorithm: str,
    max_iterations: int,
    algorithm_options: Dict[str, Any],
    link_guard: PairwiseLinkGuard | None = None,
    cs_guard: CoilSurfaceDistanceGuard | None = None,
    early_stop: EarlyStopController | None = None,
) -> Tuple[Any, int]:
    """
    Call scipy.optimize.minimize and handle result.

    Builds options, runs minimize, and returns (result, iterations_used).

    Parameters
    ----------
    objective : Callable
        Scalar objective J(x).
    gradient : Callable
        Gradient function returning ∇J(x).
    JF : Any
        Weighted objective (provides initial x).
    algorithm : str
        Scipy algorithm name (e.g. L-BFGS-B, BFGS, SLSQP).
    max_iterations : int
        Maximum iterations.
    algorithm_options : Dict
        User-provided options for scipy.

    Returns
    -------
    tuple
        (result, iterations_used) - scipy OptimizeResult and nit.
    """
    from scipy.optimize import minimize

    options = _build_scipy_minimize_options(
        algorithm, max_iterations, algorithm_options
    )
    try:
        result = minimize(
            fun=objective,
            x0=JF.x,  # type: ignore[attr-defined]
            method=algorithm,
            jac=gradient,
            options=options,
        )
    except EarlyStopTriggered as exc:
        from scipy.optimize import OptimizeResult

        result = OptimizeResult(
            x=JF.x.copy(),  # type: ignore[attr-defined]
            success=False,
            status=1,
            message=f"early stop: {exc.status.get('reason', '')}",
            nit=int(exc.status.get("iteration") or 0),
            nfev=int(exc.status.get("iteration") or 0),
            njev=int(exc.status.get("iteration") or 0),
        )
    if link_guard is not None:
        status = link_guard.current_status()
        restored = False
        if status["has_topology_change"]:
            restored = link_guard.restore_last_safe(JF)
            if restored:
                result.x = JF.x.copy()  # type: ignore[attr-defined]
            result.success = False
            action = (
                "restored last no-link checkpoint"
                if restored
                else "no no-link checkpoint available"
            )
            result.message = (
                f"{getattr(result, 'message', '')}; {action} after topology "
                "guard violation"
            )
        link_guard.write_final_audit(restored=restored)
    if cs_guard is not None:
        status = cs_guard.current_status()
        restored = False
        if status["has_clearance_violation"]:
            restored = cs_guard.restore_last_safe(JF)
            if restored:
                result.x = JF.x.copy()  # type: ignore[attr-defined]
            result.success = False
            action = (
                "restored last safe-clearance checkpoint"
                if restored
                else "no safe-clearance checkpoint available"
            )
            result.message = (
                f"{getattr(result, 'message', '')}; {action} after coil-surface "
                "clearance guard violation"
            )
        cs_guard.write_final_audit(restored=restored)
    if early_stop is not None:
        early_stop.write_final()
    iterations_used = getattr(result, "nit", 0)
    return result, iterations_used


def _run_scipy_minimize_for_modular_coils(
    c_list: list,
    constraint_scaling: Dict[int, float],
    constraint_idx_to_term: Dict[int, str],
    cc_distance_index: int | None,
    cs_distance_index: int | None,
    constraint_names_and_thresholds: list,
    base_curves: list,
    Jls: list,
    Jccdist: Any,
    Jcsdist: Any,
    Jlink: Any,
    coils: list,
    ncoils: int,
    Jts: list | None,
    coil_objective_terms: Dict[str, Any] | None,
    algorithm: str,
    max_iterations: int,
    algorithm_options: Dict[str, Any],
    verbose: bool,
    kwargs: Dict[str, Any],
    structural_obj: Any | None = None,
    out_dir: Any = None,
    history_interval: int | None = None,
    history_output_dir: Any = None,
) -> tuple:
    """
    Run scipy minimize (BFGS, L-BFGS-B, SLSQP, etc.) for modular coil optimization.

    Builds weighted objective JF from c_list and weights, defines objective/gradient
    with verbose iteration output, runs Taylor test, then minimizes. Returns the
    scipy result and iteration count.

    Parameters
    ----------
    c_list : list
        Constraint objectives (flux first, then distance, length, etc.).
    constraint_scaling, constraint_idx_to_term : dict
        Scaling and term mapping for weight building.
    cc_distance_index, cs_distance_index : int | None
        Indices of coil-coil and coil-surface distance constraints.
    constraint_names_and_thresholds : list
        (name, threshold) pairs for verbose output.
    base_curves, Jls, Jccdist, Jcsdist, Jlink : objectives
        Constraint objectives for verbose output.
    coils, ncoils : list, int
        All coils and base coil count for max pointwise F/Tq display.
    coil_objective_terms : Dict | None
        Case config for weight overrides.
    algorithm : str
        Scipy algorithm name (e.g. L-BFGS-B, BFGS, SLSQP).
    max_iterations : int
        Maximum iterations.
    algorithm_options : Dict
        User-provided options for scipy.
    verbose : bool
        Print iteration progress.
    kwargs : Dict
        constraint_weight_{i}, flux_weight, etc.

    Returns
    -------
    tuple
        (result, iterations_used) - scipy OptimizeResult and nit.
    """
    weights = _build_weights_for_scipy_minimize(
        c_list,
        constraint_scaling,
        constraint_idx_to_term,
        cc_distance_index,
        cs_distance_index,
        kwargs,
        coil_objective_terms,
    )
    # Always print distance constraint setup (helps verify weight/threshold usage)
    if cc_distance_index is not None or cs_distance_index is not None:
        cc_thresh = next(
            (t for n, t in constraint_names_and_thresholds if n == "CC Distance"),
            None,
        )
        cs_thresh = next(
            (t for n, t in constraint_names_and_thresholds if n == "CS Distance"),
            None,
        )
        cc_eff = (
            weights[cc_distance_index]
            if cc_distance_index is not None and cc_distance_index < len(weights)
            else None
        )
        cs_eff = (
            weights[cs_distance_index]
            if cs_distance_index is not None and cs_distance_index < len(weights)
            else None
        )
        cc_str = f"{cc_eff:.2e}" if cc_eff is not None else "N/A"
        cs_str = f"{cs_eff:.2e}" if cs_eff is not None else "N/A"
        cc_t = f"{cc_thresh:.2g}" if isinstance(cc_thresh, float) else str(cc_thresh)
        cs_t = f"{cs_thresh:.2g}" if isinstance(cs_thresh, float) else str(cs_thresh)
        msg = (
            f"Distance constraints: cc_threshold={cc_t} m, cs_threshold={cs_t} m, "
            f"effective_cc_weight={cc_str}, effective_cs_weight={cs_str}"
        )
        proc0_print(msg, flush=True)

    show_force = (
        coil_objective_terms is not None and "coil_coil_force" in coil_objective_terms
    )
    show_torque = (
        coil_objective_terms is not None and "coil_coil_torque" in coil_objective_terms
    )
    history_recorder = None
    if history_interval is not None and int(history_interval) > 0:
        history_recorder = OptimizationHistoryRecorder(
            history_output_dir or out_dir or ".",
            int(history_interval),
            constraint_names_and_thresholds=constraint_names_and_thresholds,
            weights=weights,
            base_curves=base_curves,
            Jccdist=Jccdist,
            Jcsdist=Jcsdist,
        )
    objective, gradient, JF, x0 = _build_objective_and_gradient(
        c_list,
        weights,
        constraint_names_and_thresholds,
        base_curves,
        Jls,
        Jccdist,
        Jcsdist,
        Jlink,
        verbose,
        coils=coils,
        ncoils=ncoils,
        show_force=show_force,
        show_torque=show_torque,
        Jts=Jts,
        structural_obj=structural_obj,
        out_dir=out_dir,
    )
    _invoke_taylor_test_for_modular_coils(objective, gradient, x0, JF, verbose)
    if history_recorder is not None:
        history_recorder.reset()
    link_guard = None
    if coil_objective_terms and bool(coil_objective_terms.get("link_guard", False)):
        link_guard = PairwiseLinkGuard(
            base_curves,
            output_dir=out_dir,
            interval=int(coil_objective_terms.get("link_guard_interval", 1)),
            penalty=float(coil_objective_terms.get("link_guard_penalty", 1e12)),
            tolerance=float(coil_objective_terms.get("link_guard_tolerance", 0.5)),
            rollback=bool(coil_objective_terms.get("link_guard_rollback", True)),
            sample_stride=int(coil_objective_terms.get("link_guard_sample_stride", 1)),
            record_interval=(
                int(coil_objective_terms["link_guard_record_interval"])
                if coil_objective_terms.get("link_guard_record_interval") is not None
                else None
            ),
        )
    cs_guard = None
    if coil_objective_terms and bool(coil_objective_terms.get("cs_guard", False)):
        hard_min = coil_objective_terms.get("cs_guard_hard_min")
        if hard_min is None:
            hard_min = coil_objective_terms.get("cs_hard_min")
        if hard_min is None:
            hard_min = coil_objective_terms.get("cs_threshold_device")
        if hard_min is None:
            hard_min = coil_objective_terms.get("cs_threshold")
        if hard_min is None:
            hard_min = cs_thresh
        soft_min = coil_objective_terms.get("cs_guard_soft_min")
        if soft_min is None:
            soft_min = coil_objective_terms.get("cs_soft_min")
        cs_guard = CoilSurfaceDistanceGuard(
            Jcsdist,
            output_dir=out_dir,
            interval=int(coil_objective_terms.get("cs_guard_interval", 5)),
            hard_min=float(hard_min or 0.0),
            soft_min=float(soft_min) if soft_min is not None else None,
            penalty=float(coil_objective_terms.get("cs_guard_penalty", 1e8)),
            rollback=bool(coil_objective_terms.get("cs_guard_rollback", True)),
        )
    early_stop = None
    if coil_objective_terms and isinstance(coil_objective_terms.get("early_stop"), dict):
        early_stop = EarlyStopController(
            coil_objective_terms["early_stop"],
            base_curves=base_curves,
            Jccdist=Jccdist,
            Jcsdist=Jcsdist,
            output_dir=out_dir,
            link_guard=link_guard,
        )
    objective, gradient, JF, _x0 = _build_objective_and_gradient(
        c_list,
        weights,
        constraint_names_and_thresholds,
        base_curves,
        Jls,
        Jccdist,
        Jcsdist,
        Jlink,
        verbose,
        coils=coils,
        ncoils=ncoils,
        show_force=show_force,
        show_torque=show_torque,
        Jts=Jts,
        structural_obj=structural_obj,
        out_dir=out_dir,
        history_recorder=history_recorder,
        link_guard=link_guard,
        cs_guard=cs_guard,
        early_stop=early_stop,
    )
    return _invoke_scipy_minimize(
        objective,
        gradient,
        JF,
        algorithm,
        max_iterations,
        algorithm_options,
        link_guard=link_guard,
        cs_guard=cs_guard,
        early_stop=early_stop,
    )
