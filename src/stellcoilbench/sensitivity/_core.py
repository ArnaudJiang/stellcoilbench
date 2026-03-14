"""
Core logic for coil sensitivity analysis: bisection, sigma sweep, and Monte-Carlo evaluation.

This module contains the numerical logic, data classes (BisectionStep, SensitivityResult),
and the main entry point run_sensitivity_analysis. Sampling and I/O are in
_sensitivity_samplers and _sensitivity_io.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.random import Generator, PCG64DXSM, SeedSequence

from ._sensitivity_io import _save_sensitivity_results_json, export_perturbed_coils_vtk
from ._sensitivity_samplers import _build_unit_samplers

if TYPE_CHECKING:
    from simsopt.field import BiotSavart
    from simsopt.geo import SurfaceRZFourier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class BisectionStep:
    """Record of a single bisection iteration.

    Attributes
    ----------
    sigma : float
        Perturbation amplitude [m] at this step.
    percentile_ratio : float
        Ratio :math:`f_B^{\\mathrm{pert}}/f_B^{\\mathrm{nom}}` at the
        target percentile.
    n_samples : int
        Monte-Carlo sample count used.
    """

    sigma: float
    percentile_ratio: float
    n_samples: int


@dataclass
class SensitivityResult:
    """Full output of a sensitivity analysis run.

    Attributes
    ----------
    critical_sigma_m : float
        Critical perturbation amplitude :math:`\\sigma^*` [m].
    nominal_fb : float
        Squared-flux :math:`f_B` of the unperturbed coils.
    factor : float
        Maximum tolerated degradation factor (e.g. 2.0).
    percentile : float
        Percentile used for bisection (e.g. 95).
    correlation_length_m : float
        GP correlation length [m].
    n_samples : int
        Monte-Carlo samples per evaluation.
    seed : int
        Random seed.
    bisection_history : list
        Log of bisection iterations.
    sweep_sigmas, sweep_p50_ratios, sweep_p95_ratios, sweep_mean_ratios : list
        Sweep data for plotting.
    """

    critical_sigma_m: float
    nominal_fb: float
    factor: float
    percentile: float
    correlation_length_m: float
    n_samples: int
    seed: int
    bisection_history: list[BisectionStep] = field(default_factory=list)
    sweep_sigmas: list[float] = field(default_factory=list)
    sweep_p50_ratios: list[float] = field(default_factory=list)
    sweep_p95_ratios: list[float] = field(default_factory=list)
    sweep_mean_ratios: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "critical_sigma_m": self.critical_sigma_m,
            "nominal_fb": self.nominal_fb,
            "factor": self.factor,
            "percentile": self.percentile,
            "correlation_length_m": self.correlation_length_m,
            "n_samples": self.n_samples,
            "seed": self.seed,
            "bisection_history": [
                {
                    "sigma": s.sigma,
                    "percentile_ratio": s.percentile_ratio,
                    "n_samples": s.n_samples,
                }
                for s in self.bisection_history
            ],
            "final_sweep": {
                "sigmas": self.sweep_sigmas,
                "p50_ratios": self.sweep_p50_ratios,
                "p95_ratios": self.sweep_p95_ratios,
                "mean_ratios": self.sweep_mean_ratios,
            },
        }


# ---------------------------------------------------------------------------
# Core Monte-Carlo evaluation
# ---------------------------------------------------------------------------


def compute_fb_perturbed(
    bs: BiotSavart,
    surface: SurfaceRZFourier,
    sigma: float,
    correlation_length_m: float,
    n_samples: int,
    seed: int = 42,
    flux_threshold: float = 0.0,
    samplers: Optional[list] = None,
) -> np.ndarray:
    r"""Evaluate :math:`f_B` over *n_samples* stochastic coil perturbations.

    Uses a Gaussian-process perturbation model with covariance
    :math:`k(s,s')` (squared-exponential in arclength) to draw coil
    deformations, then evaluates SquaredFlux :math:`f_B` on each
    perturbed coil set.

    Parameters
    ----------
    bs : simsopt.field.BiotSavart
        Nominal (unperturbed) BiotSavart field.
    surface : simsopt.geo.SurfaceRZFourier
        Plasma surface for the SquaredFlux evaluation.
    sigma : float
        Standard deviation of the Gaussian-process perturbation (metres).
    correlation_length_m : float
        Correlation length of the perturbation along the coil (metres).
        Internally converted to the normalised [0, 1] domain used by
        ``GaussianSampler`` by dividing by each coil's arc length.
        Ignored when *samplers* is provided.
    n_samples : int
        Number of Monte-Carlo realisations to draw.
    seed : int
        Base random seed for reproducibility.
    flux_threshold : float
        Threshold passed to ``SquaredFlux`` (same semantics as optimisation).
    samplers : list[GaussianSampler] | None
        Pre-built *unit-sigma* samplers (one per coil).  When provided
        the drawn samples are scaled by *sigma*, avoiding the expensive
        sampler construction on every call.  Build with
        ``_build_unit_samplers``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples,)`` with the f_B value for each
        perturbed coil set.
    """
    from simsopt.field import BiotSavart as BS, Coil
    from simsopt.geo import CurvePerturbed, PerturbationSample
    from simsopt.objectives import SquaredFlux

    coils = bs.coils
    if samplers is None:
        samplers = _build_unit_samplers(coils, correlation_length_m)
        scale_factor = sigma
    else:
        scale_factor = sigma

    seeds = SeedSequence(seed).spawn(n_samples)
    fb_values = np.empty(n_samples)

    for i in range(n_samples):
        rg = Generator(PCG64DXSM(seeds[i]))
        perturbed_coils: list[Coil] = []
        for coil, sampler in zip(coils, samplers):
            pert = PerturbationSample(sampler, randomgen=rg)
            if scale_factor != 1.0:
                for k in range(len(pert._sample)):
                    pert._sample[k] = pert._sample[k] * scale_factor
            perturbed_coils.append(Coil(CurvePerturbed(coil.curve, pert), coil.current))

        bs_pert = BS(perturbed_coils)
        Jf = SquaredFlux(surface, bs_pert, threshold=flux_threshold)
        fb_values[i] = Jf.J()

    return fb_values


# ---------------------------------------------------------------------------
# Bisection to find critical sigma
# ---------------------------------------------------------------------------


def find_critical_sigma(
    bs: BiotSavart,
    surface: SurfaceRZFourier,
    nominal_fb: float,
    correlation_length_m: float = 1.0,
    n_samples: int = 100,
    factor: float = 2.0,
    percentile: float = 95.0,
    sigma_min: float = 1e-5,
    sigma_max: float = 0.05,
    seed: int = 42,
    flux_threshold: float = 0.0,
    bisection_tol: float = 0.05,
    max_bisection_iter: int = 20,
    samplers: Optional[list] = None,
) -> tuple[float, list[BisectionStep]]:
    r"""Find the critical perturbation amplitude :math:`\sigma^*` via bisection.

    :math:`\sigma^*` is the largest :math:`\sigma` for which the
    ``percentile``-th percentile of :math:`f_B^{\mathrm{pert}}/f_B^{\mathrm{nom}}`
    is at most ``factor``. Bisection criterion: accept :math:`\sigma` if
    :math:`P_{\mathrm{percentile}}(f_B^{\mathrm{pert}}/f_B^{\mathrm{nom}})
    \leq \mathtt{factor}`.

    Parameters
    ----------
    bs : simsopt.field.BiotSavart
        Nominal BiotSavart field.
    surface : simsopt.geo.SurfaceRZFourier
        Plasma surface.
    nominal_fb : float
        f_B of the unperturbed coil set.
    correlation_length_m : float
        Physical correlation length (metres).
    n_samples : int
        Monte-Carlo samples per sigma evaluation.
    factor : float
        Maximum tolerated f_B degradation factor (e.g. 2.0).
    percentile : float
        Percentile at which to enforce the factor (e.g. 95).
    sigma_min, sigma_max : float
        Bisection bounds in metres.
    seed : int
        Random seed.
    flux_threshold : float
        Passed to ``SquaredFlux``.
    bisection_tol : float
        Relative tolerance on sigma for convergence (fraction of interval).
    max_bisection_iter : int
        Maximum bisection iterations.
    samplers : list[GaussianSampler] | None
        Pre-built unit-sigma samplers (one per coil).  When provided,
        avoids recreating samplers on every bisection iteration.

    Returns
    -------
    sigma_star : float
        Critical perturbation amplitude (metres).
    history : list[BisectionStep]
        Bisection iteration log.
    """
    if samplers is None:
        samplers = _build_unit_samplers(bs.coils, correlation_length_m)

    lo, hi = sigma_min, sigma_max
    history: list[BisectionStep] = []

    for iteration in range(max_bisection_iter):
        mid = 0.5 * (lo + hi)
        fb_vals = compute_fb_perturbed(
            bs,
            surface,
            sigma=mid,
            correlation_length_m=correlation_length_m,
            n_samples=n_samples,
            seed=seed,
            flux_threshold=flux_threshold,
            samplers=samplers,
        )
        ratio = np.percentile(fb_vals / nominal_fb, percentile)
        step = BisectionStep(
            sigma=mid, percentile_ratio=float(ratio), n_samples=n_samples
        )
        history.append(step)
        logger.info(
            "bisection iter %d: sigma=%.2e  p%g ratio=%.3f",
            iteration,
            mid,
            percentile,
            ratio,
        )

        if ratio <= factor:
            lo = mid
        else:
            hi = mid

        if (hi - lo) / max(hi, 1e-30) < bisection_tol:
            break

    sigma_star = lo
    return sigma_star, history


# ---------------------------------------------------------------------------
# Sweep around sigma* for visualisation
# ---------------------------------------------------------------------------


def _run_sweep(
    bs: BiotSavart,
    surface: SurfaceRZFourier,
    nominal_fb: float,
    sigma_star: float,
    correlation_length_m: float,
    n_samples: int,
    seed: int,
    flux_threshold: float,
    n_sweep: int = 8,
    samplers: Optional[list] = None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    r"""Evaluate :math:`f_B` statistics at several :math:`\sigma` values around :math:`\sigma^*`.

    Returns (sigmas, p50_ratios, p95_ratios, mean_ratios) for plotting
    :math:`f_B^{\mathrm{pert}}/f_B^{\mathrm{nom}}` vs :math:`\sigma`.
    """
    if samplers is None:
        samplers = _build_unit_samplers(bs.coils, correlation_length_m)

    lo = max(sigma_star * 0.1, 1e-6)
    hi = sigma_star * 3.0
    sigmas = np.linspace(lo, hi, n_sweep).tolist()

    p50: list[float] = []
    p95: list[float] = []
    means: list[float] = []

    for s in sigmas:
        fb_vals = compute_fb_perturbed(
            bs,
            surface,
            sigma=s,
            correlation_length_m=correlation_length_m,
            n_samples=n_samples,
            seed=seed,
            flux_threshold=flux_threshold,
            samplers=samplers,
        )
        ratios = fb_vals / nominal_fb
        p50.append(float(np.percentile(ratios, 50)))
        p95.append(float(np.percentile(ratios, 95)))
        means.append(float(np.mean(ratios)))

    return sigmas, p50, p95, means


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def run_sensitivity_analysis(
    coils_json_path: Path,
    case_yaml_path: Optional[Path] = None,
    plasma_surfaces_dir: Optional[Path] = None,
    correlation_length_m: float = 1.0,
    n_samples: int = 20,
    factor: float = 2.0,
    percentile: float = 95.0,
    sigma_min: float = 1e-5,
    sigma_max: float = 0.05,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    make_plot: bool = True,
    n_sweep: int = 8,
    n_vtk_samples: int = 0,
) -> SensitivityResult:
    r"""Run a full coil sensitivity analysis.

    Loads coils and surface, computes nominal :math:`f_B`, bisects to find
    :math:`\sigma^*`, optionally runs a sweep for visualisation, exports perturbed
    coil VTK files, and writes ``sensitivity_results.json`` (and an
    optional plot) to *output_dir*.

    Parameters
    ----------
    coils_json_path : Path
        Optimised coils JSON file.
    case_yaml_path : Path | None
        Case YAML (auto-detected if *None*).
    plasma_surfaces_dir : Path | None
        Plasma-surface directory (default ``plasma_surfaces/``).
    correlation_length_m : float
        Correlation length of perturbation along the coil (metres).
        Default 1.0 m -- appropriate for reactor-scale coils (20-40 m).
    n_samples : int
        Monte-Carlo samples per sigma evaluation.
    factor : float
        Maximum tolerated f_B degradation factor.
    percentile : float
        Percentile at which the factor is enforced.
    sigma_min, sigma_max : float
        Bisection bounds (metres).
    seed : int
        Random seed.
    output_dir : Path | None
        Where to write results.  Defaults to the parent of *coils_json_path*.
    make_plot : bool
        Whether to produce a PDF plot.
    n_sweep : int
        Number of sigma values in the optional sweep.
    n_vtk_samples : int
        Number of perturbed coil sets to export as VTK files (at sigma*)
        for visual comparison.  0 disables VTK export (default).

    Returns
    -------
    SensitivityResult
        Dataclass with all outputs.
    """
    import time as _time

    from simsopt.objectives import SquaredFlux

    from ..post_processing import load_coils_and_surface

    logger.info("Loading coils from %s", coils_json_path)
    bfield, surface = load_coils_and_surface(
        coils_json_path,
        case_yaml_path,
        plasma_surfaces_dir,
    )

    flux_threshold = 0.0
    if case_yaml_path is not None and case_yaml_path.exists():
        from ..path_utils import load_yaml

        case_data = load_yaml(path=case_yaml_path)
        flux_threshold = (
            case_data.get("coil_objective_terms", {})
            .get("squared_flux", {})
            .get("threshold", 0.0)
        )

    Jf = SquaredFlux(surface, bfield, threshold=flux_threshold)
    nominal_fb = float(Jf.J())
    logger.info("Nominal f_B = %.6e", nominal_fb)

    if output_dir is None:
        output_dir = coils_json_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0_samplers = _time.time()
    samplers = _build_unit_samplers(bfield.coils, correlation_length_m)
    t_samplers = _time.time() - t0_samplers
    logger.info("Built %d unit-sigma samplers in %.1f s", len(samplers), t_samplers)

    t0_bisect = _time.time()
    sigma_star, history = find_critical_sigma(
        bfield,
        surface,
        nominal_fb=nominal_fb,
        correlation_length_m=correlation_length_m,
        n_samples=n_samples,
        factor=factor,
        percentile=percentile,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        seed=seed,
        flux_threshold=flux_threshold,
        samplers=samplers,
    )
    t_bisect = _time.time() - t0_bisect
    logger.info(
        "Critical sigma* = %.2f mm  (bisection: %.1f s)", sigma_star * 1e3, t_bisect
    )

    sweep_sigmas: list[float] = []
    sweep_p50: list[float] = []
    sweep_p95: list[float] = []
    sweep_mean: list[float] = []
    t_sweep = 0.0
    if make_plot and sigma_star > 0:
        t0_sweep = _time.time()
        sweep_sigmas, sweep_p50, sweep_p95, sweep_mean = _run_sweep(
            bfield,
            surface,
            nominal_fb,
            sigma_star,
            correlation_length_m=correlation_length_m,
            n_samples=n_samples,
            seed=seed,
            flux_threshold=flux_threshold,
            n_sweep=n_sweep,
            samplers=samplers,
        )
        t_sweep = _time.time() - t0_sweep
        logger.info("Sweep (%d points): %.1f s", n_sweep, t_sweep)

    result = SensitivityResult(
        critical_sigma_m=sigma_star,
        nominal_fb=nominal_fb,
        factor=factor,
        percentile=percentile,
        correlation_length_m=correlation_length_m,
        n_samples=n_samples,
        seed=seed,
        bisection_history=history,
        sweep_sigmas=sweep_sigmas,
        sweep_p50_ratios=sweep_p50,
        sweep_p95_ratios=sweep_p95,
        sweep_mean_ratios=sweep_mean,
    )

    results_path = output_dir / "sensitivity_results.json"
    _save_sensitivity_results_json(result, results_path)
    logger.info("Results written to %s", results_path)

    if make_plot and sigma_star > 0 and sweep_sigmas:
        from ._plotting import plot_sensitivity

        plot_path = output_dir / "sensitivity_plot.pdf"
        plot_sensitivity(result, plot_path)

    t_vtk = 0.0
    if n_vtk_samples > 0 and sigma_star > 0:
        t0_vtk = _time.time()
        vtk_paths = export_perturbed_coils_vtk(
            bfield,
            surface,
            sigma=sigma_star,
            correlation_length_m=correlation_length_m,
            output_dir=output_dir,
            n_vtk_samples=n_vtk_samples,
            seed=seed,
        )
        t_vtk = _time.time() - t0_vtk
        logger.info("Exported %d perturbed VTK files in %.1f s", len(vtk_paths), t_vtk)

    logger.info(
        "Sensitivity timing: samplers=%.1fs  bisection=%.1fs  sweep=%.1fs  vtk=%.1fs  total=%.1fs",
        t_samplers,
        t_bisect,
        t_sweep,
        t_vtk,
        t_samplers + t_bisect + t_sweep + t_vtk,
    )

    return result
