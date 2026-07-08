"""External coil evaluation logic.

Provides functions to load coils from JSON and compute leaderboard metrics
without running optimization. Used for evaluating external coil solutions
(e.g. from Zenodo) for leaderboard inclusion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from ..constants import TANGENT_NORM_FLOOR
from ..path_utils import get_target_B_from_surface, load_surface_with_range
from ..post_processing import (
    _get_coils_from_bfield,
    load_bfield_from_coils_json,
)
from ..post_processing._coil_io import _setup_surface_for_eval
from ..utils import suppress_output
from ._results import compute_total_current
from ._thresholds import _compute_thresholds_from_surface
from ._length_balance import coil_length_distribution_metrics

if TYPE_CHECKING:
    from simsopt.field import BiotSavart
    from simsopt.geo import SurfaceRZFourier

logger = logging.getLogger(__name__)

# Floor for |B| to avoid division by zero (matches optimization.py)
SAFE_ABS_B_FLOOR = 1e-10


def _check_coils_linked_to_surface(s: "SurfaceRZFourier", base_curves: list) -> bool:
    """Check that each base coil encircles the plasma.

    A coil is linked if it has points both inside and outside the local
    surface cross-section (R_min, R_max) at each toroidal angle. Uses
    per-phi cross-sections for strongly-shaped stellarators.

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface.
    base_curves : list
        Base coil curves to check.

    Returns
    -------
    bool
        True if all coils encircle the plasma, False otherwise.
    """
    surface_gamma = s.gamma()
    R_surface = np.sqrt(surface_gamma[:, :, 0] ** 2 + surface_gamma[:, :, 1] ** 2)
    R_min_per_phi = np.min(R_surface, axis=1)
    R_max_per_phi = np.max(R_surface, axis=1)
    phi_surface_slices = np.arctan2(surface_gamma[:, 0, 1], surface_gamma[:, 0, 0])
    for c in base_curves:
        gamma = c.gamma()
        R_coil = np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)
        phi_coil = np.arctan2(gamma[:, 1], gamma[:, 0])
        dphi = phi_coil[:, None] - phi_surface_slices[None, :]
        dphi = np.abs(np.arctan2(np.sin(dphi), np.cos(dphi)))
        nearest_phi_idx = np.argmin(dphi, axis=1)
        local_R_min = R_min_per_phi[nearest_phi_idx]
        local_R_max = R_max_per_phi[nearest_phi_idx]
        has_inside = np.any(R_coil < local_R_min)
        has_outside = np.any(R_coil > local_R_max)
        if not (has_inside and has_outside):
            return False
    return True


def _compute_lorentz_force_torque_fallback(
    coil_subset: list,
    all_coils: list,
) -> tuple[list[float], list[float]]:
    r"""Compute max force [N/m] and torque [N] per coil via Lorentz formula.

    Force per unit length and torque:

    .. math::
        \frac{d\mathbf{F}}{d\ell} = I\,\mathbf{t}\times\mathbf{B},
        \quad \boldsymbol{\tau} = \mathbf{r}\times\mathbf{F}

    Excludes self-field to avoid singularity.

    Parameters
    ----------
    coil_subset : list
        Coils to compute force/torque for.
    all_coils : list
        All coil objects (used for mutual field contribution).

    Returns
    -------
    tuple[list[float], list[float]]
        (max_force per coil, max_torque per coil).
    """
    from simsopt.field import BiotSavart

    max_force = []
    max_torque = []
    for c in coil_subset:
        other_coils = [ac for ac in all_coils if id(ac) != id(c)]
        if not other_coils:
            max_force.append(0.0)
            max_torque.append(0.0)
            continue
        bs = BiotSavart(other_coils)
        curve = c.curve
        gamma = curve.gamma()
        gammadash = (
            curve.gammadash() if hasattr(curve, "gammadash") else curve.dgamma_by_dphi()
        )
        I_val = float(abs(c.current.get_value()))
        pts = gamma.reshape(-1, 3)
        bs.set_points(pts)
        B = bs.B().reshape(-1, 3)
        ds = np.linalg.norm(gammadash, axis=1, keepdims=True)
        ds = np.where(ds > TANGENT_NORM_FLOOR, ds, 1.0)
        tangent = gammadash / ds
        force_density = I_val * np.cross(tangent, B)
        force_mag = np.linalg.norm(force_density, axis=1)
        max_force.append(float(np.max(force_mag)))
        torque_density = np.cross(pts, force_density)
        torque_mag = np.linalg.norm(torque_density, axis=1)
        max_torque.append(float(np.max(torque_mag)))
    return max_force, max_torque


def _compute_coil_subset_metrics(
    coil_subset: list,
    base_curves_subset: list,
    all_coils: list,
    s: "SurfaceRZFourier",
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute metrics for a subset of coils.

    Force/torque use all_coils for mutual interaction.

    Parameters
    ----------
    coil_subset : list
        Coil objects in this subset.
    base_curves_subset : list
        Base curves for this subset.
    all_coils : list
        All coil objects (for mutual force/torque).
    s : SurfaceRZFourier
        Plasma surface.
    kwargs : Dict[str, Any]
        Pass-through options.

    Returns
    -------
    Dict[str, Any]
        total_current, max_force, max_torque, coils_linked_to_surface,
        final_length_per_coil, final_current_per_coil, etc.
    """
    from simsopt.geo import ArclengthVariation, CurveLength

    n = len(coil_subset)
    if n == 0 or len(base_curves_subset) == 0:
        return {
            "total_current": 0.0,
            "max_force": [],
            "max_torque": [],
            "coils_linked_to_surface": False,
            "final_length_per_coil": [],
            "final_current_per_coil": [],
            "final_total_length": 0.0,
            "final_max_curvature": 0.0,
            "final_average_curvature": 0.0,
            "final_arclength_variation": 0.0,
            "final_mean_squared_curvature": 0.0,
            "final_max_torsion": 0.0,
            "final_mean_squared_torsion": 0.0,
        }

    currents = (
        [float(abs(c.current.get_value())) for c in coil_subset] if coil_subset else []
    )
    total_current = sum(currents)

    if n > 0 and len(all_coils) > 0:
        try:
            from simsopt.field.force import coil_force, coil_torque

            max_force = [
                np.max(np.linalg.norm(coil_force(c, all_coils), axis=1))
                for c in coil_subset
            ]
            max_torque = [
                np.max(np.linalg.norm(coil_torque(c, all_coils), axis=1))
                for c in coil_subset
            ]
        except (ImportError, Exception) as exc:
            logger.debug("coil_force/coil_torque failed, using fallback: %s", exc)
            max_force, max_torque = _compute_lorentz_force_torque_fallback(
                coil_subset, all_coils
            )
    else:
        max_force = [0.0] * n
        max_torque = [0.0] * n

    coils_linked = _check_coils_linked_to_surface(s, base_curves_subset)

    lengths = [float(CurveLength(c).J()) for c in base_curves_subset]
    length_metrics = coil_length_distribution_metrics(lengths)
    kappas = [c.kappa() for c in base_curves_subset]
    taus = [np.asarray(c.torsion()) for c in base_curves_subset]

    return {
        "total_current": float(total_current),
        "max_force": [float(f) for f in max_force],
        "max_torque": [float(t) for t in max_torque],
        "coils_linked_to_surface": coils_linked,
        "final_length_per_coil": lengths,
        **length_metrics,
        "final_current_per_coil": currents,
        "final_total_length": float(sum(lengths)),
        "final_max_curvature": float(np.max([np.max(k) for k in kappas]))
        if kappas
        else 0.0,
        "final_average_curvature": float(np.mean([np.mean(k) for k in kappas]))
        if kappas
        else 0.0,
        "final_arclength_variation": float(
            np.mean([ArclengthVariation(c).J() for c in base_curves_subset])
        ),
        "final_mean_squared_curvature": float(
            np.max([np.mean(c.kappa() ** 2) for c in base_curves_subset])
        ),
        "final_max_torsion": float(np.max([np.max(np.abs(t)) for t in taus]))
        if taus
        else 0.0,
        "final_mean_squared_torsion": float(
            np.max([np.mean(np.asarray(t) ** 2) for t in taus])
        )
        if taus
        else 0.0,
    }


def _compute_optimization_metrics(
    bs: "BiotSavart",
    coils: list,
    base_curves: list,
    ncoils: int,
    s: "SurfaceRZFourier",
    s_plot: "SurfaceRZFourier",
    qphi: int,
    qtheta: int,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute final metrics (B_final, force/torque, B_N, coil-surface linking).

    Does not save files.

    Parameters
    ----------
    bs : BiotSavart
        BiotSavart field with coils.
    coils : list
        Coil objects.
    base_curves : list
        Base coil curves.
    ncoils : int
        Number of base coils.
    s, s_plot : SurfaceRZFourier
        Optimization and plotting surfaces.
    qphi, qtheta : int
        Plotting grid dimensions.
    kwargs : Dict[str, Any]
        May contain vc_target, vc_target_plot for virtual casing.

    Returns
    -------
    Dict[str, Any]
        B_final, max_force, max_torque, avg_BdotN_over_B, max_BdotN_overB,
        coils_linked_to_surface, total_current_final.
    """
    from simsopt.util import calculate_modB_on_major_radius

    total_current_final = compute_total_current(coils, ncoils)

    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    with suppress_output():
        B_final = calculate_modB_on_major_radius(bs, s_plot)

    if ncoils > 0 and len(coils) > 0:
        if hasattr(coils[0], "force") and hasattr(coils[0], "torque"):
            max_force = [
                np.max(np.linalg.norm(c.force(coils), axis=1)) for c in coils[:ncoils]
            ]
            max_torque = [
                np.max(np.linalg.norm(c.torque(coils), axis=1)) for c in coils[:ncoils]
            ]
        else:
            try:
                from simsopt.field.force import coil_force, coil_torque

                max_force = [
                    np.max(np.linalg.norm(coil_force(c, coils), axis=1))
                    for c in coils[:ncoils]
                ]
                max_torque = [
                    np.max(np.linalg.norm(coil_torque(c, coils), axis=1))
                    for c in coils[:ncoils]
                ]
            except (ImportError, Exception) as exc:
                logger.debug("coil_force/coil_torque failed, using fallback: %s", exc)
                subset = coils[:ncoils]
                max_force, max_torque = _compute_lorentz_force_torque_fallback(
                    subset, coils
                )
    else:
        max_force = []
        max_torque = []

    vc_target = kwargs.get("vc_target", None)
    nphi = len(s.quadpoints_phi)
    ntheta = len(s.quadpoints_theta)
    bs.set_points(s.gamma().reshape((-1, 3)))
    B_field = bs.B().reshape((nphi, ntheta, 3))
    unit_normal = s.unitnormal().reshape((nphi, ntheta, 3))
    BdotN_coils = np.sum(B_field * unit_normal, axis=2)

    if vc_target is not None:
        absBn = np.abs(BdotN_coils - vc_target)
    else:
        absBn = np.abs(BdotN_coils)

    abs_B = bs.AbsB().reshape((nphi, ntheta))
    avg_BdotN_over_B = np.mean(absBn) / np.mean(abs_B) if np.mean(abs_B) > 0 else 0.0
    abs_B_safe = np.where(abs_B > SAFE_ABS_B_FLOOR, abs_B, SAFE_ABS_B_FLOOR)
    max_BdotN_overB = np.max(absBn / abs_B_safe) if np.any(abs_B > 0) else 0.0

    coils_linked_to_surface = _check_coils_linked_to_surface(s, base_curves)

    return {
        "B_final": B_final,
        "max_force": max_force,
        "max_torque": max_torque,
        "avg_BdotN_over_B": avg_BdotN_over_B,
        "max_BdotN_overB": max_BdotN_overB,
        "coils_linked_to_surface": coils_linked_to_surface,
        "total_current_final": total_current_final,
    }


def _load_and_setup_external_coils(
    coils_json_path: Path,
    surface_file: str,
    surface_range: str = "half period",
    surface_resolution: int = 32,
    plasma_surfaces_dir: Path | None = None,
) -> tuple[Any, list, Any, Any, Any, Path, float, Dict[str, Any]]:
    """Load coils and surface for external evaluation.

    Reuses post_processing coil loading via load_bfield_from_coils_json.
    When a case.yaml exists near the coils path, uses case-based surface
    resolution; otherwise uses explicit surface_file with plasma_surfaces_dir.

    Notes
    -----
    Resolution order: (1) Try find_case_and_surface_path from coils path.
    (2) On failure, use surface_file with shared _resolve_surface_path_from_hints
    (case/plasma/coils + walk-up). Supports StellCoilBench case layouts and
    standalone Zenodo/external coil packages.

    Parameters
    ----------
    coils_json_path : Path
        Path to coils.json (simsopt BiotSavart or MagneticFieldSum format).
    surface_file : str
        Plasma surface file (used when no case.yaml is found).
    surface_range : str
        Surface range: "half period" or "full torus".
    surface_resolution : int
        Quadrature resolution for surface evaluation.
    plasma_surfaces_dir : Path | None
        Directory containing plasma surface files.

    Returns
    -------
    tuple
        bfield, coils, bs, s, s_plot, surface_path, target_B, th
    """
    from simsopt.field import BiotSavart

    plasma_surfaces_dir = plasma_surfaces_dir or Path("plasma_surfaces")
    bfield = load_bfield_from_coils_json(coils_json_path)
    coils = _get_coils_from_bfield(bfield)
    if not coils:
        raise ValueError("Could not extract coils from loaded object")
    bs = bfield if isinstance(bfield, BiotSavart) else BiotSavart(coils)

    surface_path, eff_range, s = _setup_surface_for_eval(
        coils_json_path,
        case_yaml_path=None,
        plasma_surfaces_dir=plasma_surfaces_dir,
        surface_file=surface_file,
        surface_range=surface_range,
        nphi=surface_resolution,
        ntheta=surface_resolution,
    )

    s_plot = load_surface_with_range(
        surface_path,
        surface_range=eff_range,
        nphi=64,
        ntheta=64,
    )
    s_plot.filename = str(surface_path.resolve())

    target_B = get_target_B_from_surface(str(surface_path.name))
    th = _compute_thresholds_from_surface(s, {})

    return bfield, coils, bs, s, s_plot, surface_path, target_B, th


def _build_external_coil_metrics(
    base_curves: list,
    ncoils: int,
    coil_metrics: Dict[str, Any],
    opt_metrics: Dict[str, Any],
    th: Dict[str, Any],
    target_B: float,
    total_current_final: float,
    coil_order: int,
    flux_val: float,
    cc_dist: float,
    cs_dist: float,
    link_val: float,
    extra_metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build metrics dict for external coil evaluation (modular coils).

    Parameters
    ----------
    base_curves : list
        Base coil curves.
    ncoils : int
        Number of coils.
    coil_metrics : Dict[str, Any]
        From _compute_coil_subset_metrics.
    opt_metrics : Dict[str, Any]
        From _compute_optimization_metrics.
    th : Dict[str, Any]
        Cached thresholds.
    target_B : float
        Target field magnitude.
    total_current_final : float
        Total current after evaluation.
    coil_order : int
        Fourier order of coils.
    flux_val, cc_dist, cs_dist, link_val : float
        Flux, coil-coil dist, coil-surface dist, linking number.
    extra_metrics : Dict[str, Any] | None
        Optional extra keys.

    Returns
    -------
    Dict[str, Any]
        Complete metrics dict for results.json.
    """
    metrics: Dict[str, Any] = {
        "final_squared_flux": float(flux_val),
        "score_primary": float(flux_val),
        "final_min_cc_separation": float(cc_dist),
        "final_min_cs_separation": float(cs_dist),
        "final_linking_number": float(link_val),
        "coils_linked_to_surface": opt_metrics["coils_linked_to_surface"],
        "avg_BdotN_over_B": float(opt_metrics["avg_BdotN_over_B"]),
        "max_BdotN_over_B": float(opt_metrics["max_BdotN_overB"]),
        "final_total_length": float(coil_metrics["final_total_length"]),
        "final_arclength_variation": float(coil_metrics["final_arclength_variation"]),
        "final_mean_squared_curvature": float(
            coil_metrics["final_mean_squared_curvature"]
        ),
        "final_max_curvature": float(
            np.max([np.max(c.kappa()) for c in base_curves]) if base_curves else 0.0
        ),
        "final_max_torsion": float(coil_metrics.get("final_max_torsion", 0.0)),
        "final_mean_squared_torsion": float(
            coil_metrics.get("final_mean_squared_torsion", 0.0)
        ),
        "num_coils": ncoils,
        "coil_order": coil_order,
        "target_B_field": target_B,
        "total_current_after": float(total_current_final),
        "optimization_time": 0.0,
        "iterations_used": 0,
        "final_length_per_coil": [
            float(x) for x in coil_metrics["final_length_per_coil"]
        ],
        "final_current_per_coil": [
            float(x) for x in coil_metrics["final_current_per_coil"]
        ],
        "_cached_thresholds": {
            "a0": th.get("a0"),
            "major_radius": th.get("major_radius"),
            "minor_radius": th.get("minor_radius"),
        },
    }
    if opt_metrics.get("max_force"):
        metrics["final_max_max_coil_force"] = float(np.max(opt_metrics["max_force"]))
        metrics["final_max_force_per_coil"] = [
            float(f) for f in opt_metrics["max_force"]
        ]
    if opt_metrics.get("max_torque"):
        metrics["final_max_max_coil_torque"] = float(np.max(opt_metrics["max_torque"]))
        metrics["final_max_torque_per_coil"] = [
            float(t) for t in opt_metrics["max_torque"]
        ]
    if extra_metrics:
        metrics.update(extra_metrics)
    return metrics


def evaluate_external_coils(
    coils_json_path: Path,
    surface_file: str,
    surface_range: str = "half period",
    surface_resolution: int = 32,
    plasma_surfaces_dir: Path | None = None,
) -> Dict[str, Any]:
    """Load coils from JSON and compute leaderboard metrics without running optimization.

    Used to evaluate external coil solutions (e.g. from Zenodo) for leaderboard inclusion.

    Parameters
    ----------
    coils_json_path : Path
        Path to coils.json (simsopt BiotSavart or MagneticFieldSum format).
    surface_file : str
        Plasma surface file (e.g. input.LandremanPaul2021_QA).
    surface_range : str
        Surface range: "half period" or "full torus".
    surface_resolution : int
        Quadrature resolution for surface evaluation.
    plasma_surfaces_dir : Path | None
        Directory containing plasma surface files. Defaults to plasma_surfaces/.

    Returns
    -------
    Dict[str, Any]
        Metrics dict suitable for results.json (metrics, score_primary, etc.).
    """
    from simsopt.geo import (
        CurveCurveDistance,
        CurveSurfaceDistance,
        LinkingNumber,
    )
    from simsopt.objectives import SquaredFlux

    bfield, coils, bs, s, s_plot, _surface_path, target_B, th = (
        _load_and_setup_external_coils(
            coils_json_path,
            surface_file,
            surface_range,
            surface_resolution,
            plasma_surfaces_dir,
        )
    )

    nfp = s.nfp
    stellsym = s.stellsym
    symmetry_factor = nfp * (2 if stellsym else 1)
    ncoils = max(1, len(coils) // symmetry_factor)
    step = max(1, len(coils) // ncoils)
    base_coil_indices = list(range(0, len(coils), step))[:ncoils]
    base_curves = [coils[i].curve for i in base_coil_indices]
    base_coils = [coils[i] for i in base_coil_indices]
    curves = [c.curve for c in coils]

    flux_threshold = th.get("flux_threshold", 1e-8)
    cc_threshold = th.get("cc_threshold", 0.1)
    cs_threshold = th.get("cs_threshold", 0.1)
    Jf = SquaredFlux(s, bs, threshold=flux_threshold)
    Jccdist = CurveCurveDistance(curves, cc_threshold, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, cs_threshold)
    Jlink = LinkingNumber(curves, downsample=2)

    opt_metrics = _compute_optimization_metrics(
        bs, coils, base_curves, ncoils, s, s_plot, 64, 64, {}
    )
    coil_metrics = _compute_coil_subset_metrics(base_coils, base_curves, coils, s, {})

    total_current_final = compute_total_current(coils, ncoils)

    coil_order = (
        int(base_curves[0].order)
        if base_curves and hasattr(base_curves[0], "order")
        else 16
    )

    return _build_external_coil_metrics(
        base_curves=base_curves,
        ncoils=ncoils,
        coil_metrics=coil_metrics,
        opt_metrics=opt_metrics,
        th=th,
        target_B=target_B,
        total_current_final=total_current_final,
        coil_order=coil_order,
        flux_val=float(Jf.J()),
        cc_dist=float(Jccdist.shortest_distance()),
        cs_dist=float(Jcsdist.shortest_distance()),
        link_val=float(Jlink.J()),
    )
