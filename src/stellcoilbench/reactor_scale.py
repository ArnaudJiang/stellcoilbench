"""
Reactor-scale metrics and REBCO winding-pack model.

Converts device-scale coil metrics to ARIES-CS reactor-scale equivalents.
Implements the Stellaris winding-pack model (Lion et al., FED 214, 2025)
for N_turns, critical-current density, and finite-build geometry.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config_scheme import CaseConfig

import numpy as np

from .config_scheme import ARIES_CS_MINOR_RADIUS, REACTOR_REFERENCE

logger = logging.getLogger(__name__)

# Stellaris-style winding-pack constants (Lion et al., FED 214, 2025)
STELLARIS_T_OP = 20.0  # Operating temperature [K]
STELLARIS_ETA = 0.80  # Utilization cap  j_op / j_crit ≤ η
STELLARIS_I_LEAD_MAX = 50e3  # Current-lead limit [A]  (50 kA)
STELLARIS_A_TURN = 400e-6  # Turn cross-section area [m²]  (20 mm × 20 mm)
STELLARIS_A_HTS = 36e-6  # HTS tape-stack area [m²]  (6 mm × 6 mm)

# Winding-pack self-field enhancement factor
WP_B_ENHANCEMENT = 1.3


def _rebco_jc_tape_stack(B: float, T_op: float = 20.0) -> float:
    """Engineering critical-current density of an optimally-aligned REBCO
    tape stack at temperature *T_op* and peak field *B* [T].

    Returns j_crit in **A/m²** (SI).

    The model is a Kim-like parametrization calibrated to the Stellaris
    magneto-angular Jc data at 20 K with field-aligned tapes
    (Lion et al., FED 214, 2025, Table 8 / Fig. 42).

    Model:  j_crit(B) = C₀ / (1 + (B/B₀)^α)

    At T = 20 K the fitted constants are:
        C₀ = 5.0 × 10⁹  A/m²   (≈ 5000 A/mm²  at self-field)
        B₀ = 18.14 T
        α  = 0.902

    Validation against Stellaris Table 8 (tape-stack j_op/j_crit):
        B=20 T → j_crit ≈ 2450 A/mm²,  B=25 T → j_crit ≈ 2200 A/mm²

    For temperatures other than 20 K a simple linear scaling
    Jc(T) ∝ (1 − T/Tc) with Tc = 92 K is applied.
    """
    T_REF = 20.0
    T_C = 92.0  # REBCO critical temperature [K]
    C0 = 5.0e9  # A/m²  (= 5000 A/mm²)
    B0 = 18.14  # T
    ALPHA = 0.902

    if B < 0:
        B = 0.0

    jc_20K = C0 / (1.0 + (B / B0) ** ALPHA)

    if abs(T_op - T_REF) > 0.01:
        jc = jc_20K * (1.0 - T_op / T_C) / (1.0 - T_REF / T_C)
    else:
        jc = jc_20K

    return max(jc, 0.0)


def compute_N_turns_critical_current(
    per_coil_forces: list[float],
    per_coil_currents: list[float] | None,
    per_coil_lengths: list[float] | None,
    L_scale: float,
    B_scale: float,
    target_B: float,
    *,
    T_op: float = STELLARIS_T_OP,
    eta: float = STELLARIS_ETA,
    I_lead_max: float = STELLARIS_I_LEAD_MAX,
    A_HTS: float = STELLARIS_A_HTS,
    wp_enhancement: float = WP_B_ENHANCEMENT,
) -> dict[str, Any]:
    """Compute per-coil turn counts based on critical-current density.

    Following the Stellaris winding-pack model (Lion et al., FED 2025):
      - 20 K operating temperature
      - 80 % utilization margin (η = j_op / j_crit)
      - 50 kA current-lead limit
      - 6 mm × 6 mm HTS tape-stack cross-section

    Parameters
    ----------
    per_coil_forces : list[float]
        Device-scale maximum force/length per base coil [N/m].
    per_coil_currents : list[float] | None
        Device-scale current per base coil [A].  If *None*, currents
        are estimated from force data.
    per_coil_lengths : list[float] | None
        Device-scale centreline length per base coil [m].
    L_scale, B_scale : float
        Geometric and magnetic-field scaling ratios (reactor / device).
    target_B : float
        Device-scale target B-field [T].

    Returns
    -------
    dict with keys:
        N_turns_jc : list[int]
            Per-coil turn count from Jc requirements.
        NI_reactor : list[float]
            Required ampere-turns per coil at reactor scale [A].
        I_turn : list[float]
            Operating current per turn [A].
        B_peak_estimate : list[float]
            Estimated peak conductor field [T].
        jc_tape_stack : list[float]
            Tape-stack j_crit at the peak field [A/m²].
        Ic_cable : list[float]
            Critical current of the HTS cable [A].
        model_params : dict
            Constants used (T_op, eta, I_lead_max, A_HTS, wp_enhancement).
    """
    n_coils = len(per_coil_forces)

    if per_coil_currents is not None and len(per_coil_currents) == n_coils:
        I_dev = [abs(float(c)) for c in per_coil_currents]
    else:
        I_dev = [abs(float(f)) / max(target_B, 0.01) for f in per_coil_forces]

    NI_list: list[float] = []
    I_turn_list: list[float] = []
    B_peak_list: list[float] = []
    jc_list: list[float] = []
    Ic_list: list[float] = []
    N_turns_jc: list[int] = []

    for i in range(n_coils):
        NI_i = I_dev[i] * B_scale * L_scale
        NI_list.append(float(NI_i))

        if I_dev[i] > 0:
            B_ext_i = (per_coil_forces[i] / I_dev[i]) * B_scale
        else:
            B_ext_i = target_B * B_scale
        B_peak_i = B_ext_i * wp_enhancement
        B_peak_list.append(float(B_peak_i))

        jc_i = _rebco_jc_tape_stack(B_peak_i, T_op)
        jc_list.append(float(jc_i))
        Ic_cable_i = jc_i * A_HTS
        Ic_list.append(float(Ic_cable_i))

        I_turn_i = min(I_lead_max, eta * Ic_cable_i)
        I_turn_list.append(float(I_turn_i))

        if I_turn_i > 0:
            n_i = max(1, int(np.ceil(NI_i / I_turn_i)))
        else:
            n_i = 1
        N_turns_jc.append(n_i)

    return {
        "N_turns_jc": N_turns_jc,
        "NI_reactor": NI_list,
        "I_turn": I_turn_list,
        "B_peak_estimate": B_peak_list,
        "jc_tape_stack": jc_list,
        "Ic_cable": Ic_list,
        "model_params": {
            "T_op_K": T_op,
            "eta": eta,
            "I_lead_max_A": I_lead_max,
            "A_HTS_m2": A_HTS,
            "wp_enhancement": wp_enhancement,
        },
    }


def _apply_reactor_scaling(
    metrics: dict[str, Any],
    L_scale: float,
    B_scale: float,
) -> dict[str, float]:
    """Scale device-scale metrics to reactor-scale values.

    Applies the ARIES-CS scaling relationships to convert device-scale
    coil optimisation metrics into reactor-scale equivalents:

    - **Lengths** (d_cc, d_cs, total_length): ``× L``
    - **Curvature** (κ): ``× 1/L``
    - **Mean squared curvature**: ``× 1/L²``
    - **Force/length** [N/m → MN/m]: ``× B²L / 10⁶``
    - **Torque/length** [N → MN]: ``× B²L² / 10⁶``
    - **Arclength variation** [m²]: ``× L²``
    - **SquaredFlux** [T²m²]: ``× B²L²``

    Parameters
    ----------
    metrics : dict[str, Any]
        Device-scale metrics dict (keys prefixed ``final_``).
    L_scale : float
        Geometric scaling ratio (reactor / device).
    B_scale : float
        Magnetic-field scaling ratio (reactor / device).

    Returns
    -------
    dict[str, float]
        Scaled metrics with keys prefixed ``reactor_scale_``.
    """
    scaled: dict[str, float] = {}

    length_metrics = [
        "final_min_cs_separation",
        "final_min_cc_separation",
        "final_total_length",
    ]
    for key in length_metrics:
        if key in metrics:
            reactor_key = key.replace("final_", "reactor_scale_")
            scaled[reactor_key] = float(metrics[key]) * L_scale

    curvature_metrics = [
        "final_max_curvature",
        "final_average_curvature",
        "final_mean_squared_curvature",
    ]
    for key in curvature_metrics:
        if key in metrics:
            reactor_key = key.replace("final_", "reactor_scale_")
            if "mean_squared" in key:
                scaled[reactor_key] = float(metrics[key]) / (L_scale**2)
            else:
                scaled[reactor_key] = float(metrics[key]) / L_scale

    force_scale = (B_scale**2) * L_scale
    force_metrics = [
        "final_max_max_coil_force",
        "final_avg_max_coil_force",
    ]
    for key in force_metrics:
        if key in metrics:
            reactor_key = key.replace("final_", "reactor_scale_")
            scaled[reactor_key] = float(metrics[key]) * force_scale / 1e6

    torque_scale = (B_scale**2) * (L_scale**2)
    torque_metrics = [
        "final_max_max_coil_torque",
        "final_avg_max_coil_torque",
    ]
    for key in torque_metrics:
        if key in metrics:
            reactor_key = key.replace("final_", "reactor_scale_")
            scaled[reactor_key] = float(metrics[key]) * torque_scale / 1e6

    if "final_arclength_variation" in metrics:
        scaled["reactor_scale_arclength_variation"] = float(
            metrics["final_arclength_variation"]
        ) * (L_scale**2)

    if "final_squared_flux" in metrics:
        flux_scale = (B_scale**2) * (L_scale**2)
        scaled["reactor_scale_squared_flux"] = (
            float(metrics["final_squared_flux"]) * flux_scale
        )

    return scaled


def _compute_turn_counts(
    metrics: dict[str, Any],
    reactor_metrics: dict[str, Any],
    L_scale: float,
    B_scale: float,
    target_B: float,
) -> None:
    """Compute REBCO turn counts and winding-pack derived quantities.

    Determines per-coil turn counts from both force limits and
    critical-current density (Jc), then populates *reactor_metrics*
    in-place with:

    - ``reactor_scale_force_per_coil_MN_per_m``
    - ``N_turns_per_coil``, ``N_turns_force``, ``N_turns_jc``
    - ``force_limit_MN_per_m``
    - ``jc_model`` (Jc-based model details)
    - ``winding_pack_width_per_coil``, ``max_winding_pack_width``
    - ``finite_build_cc_clearance``
    - ``per_turn_max_force``, ``per_turn_max_torque``
    - ``total_superconductor_length_km``

    Parameters
    ----------
    metrics : dict[str, Any]
        Device-scale metrics dict (keys prefixed ``final_``).
    reactor_metrics : dict[str, Any]
        Reactor-scale metrics dict being built up; modified **in-place**.
    L_scale : float
        Geometric scaling ratio (reactor / device).
    B_scale : float
        Magnetic-field scaling ratio (reactor / device).
    target_B : float
        Device-scale target B-field [T].
    """
    FORCE_LIMIT_MN_PER_M = 0.5
    force_scale = (B_scale**2) * L_scale
    torque_scale = (B_scale**2) * (L_scale**2)

    per_coil_forces = metrics.get("final_max_force_per_coil")
    per_coil_currents = metrics.get("final_current_per_coil")
    per_coil_lengths = metrics.get("final_length_per_coil")

    has_forces = per_coil_forces is not None and len(per_coil_forces) > 0
    n_from_currents = len(per_coil_currents) if per_coil_currents is not None else 0
    n_from_lengths = len(per_coil_lengths) if per_coil_lengths is not None else 0
    can_run_jc = n_from_currents > 0 or n_from_lengths > 0
    n_coils_for_turns = (
        len(per_coil_forces)
        if has_forces and per_coil_forces
        else max(n_from_currents, n_from_lengths)
    )

    if not ((has_forces or can_run_jc) and n_coils_for_turns > 0):
        return

    if has_forces and per_coil_forces is not None:
        reactor_force_per_coil = [f * force_scale / 1e6 for f in per_coil_forces]
        n_turns_force = [
            max(1, int(np.ceil(f / FORCE_LIMIT_MN_PER_M)))
            for f in reactor_force_per_coil
        ]
        forces_for_jc: list = per_coil_forces
    else:
        forces_for_jc = [0.0] * n_coils_for_turns
        reactor_force_per_coil = [0.0] * n_coils_for_turns
        n_turns_force = [1] * n_coils_for_turns

    jc_result = compute_N_turns_critical_current(
        per_coil_forces=forces_for_jc,
        per_coil_currents=per_coil_currents,
        per_coil_lengths=per_coil_lengths,
        L_scale=L_scale,
        B_scale=B_scale,
        target_B=target_B,
    )
    n_turns_jc = jc_result["N_turns_jc"]

    n_turns_per_coil = [max(nf, nj) for nf, nj in zip(n_turns_force, n_turns_jc)]

    reactor_metrics["reactor_scale_force_per_coil_MN_per_m"] = reactor_force_per_coil
    reactor_metrics["N_turns_per_coil"] = n_turns_per_coil
    reactor_metrics["N_turns_force"] = n_turns_force
    reactor_metrics["N_turns_jc"] = n_turns_jc
    reactor_metrics["force_limit_MN_per_m"] = FORCE_LIMIT_MN_PER_M
    reactor_metrics["jc_model"] = {
        "NI_reactor": jc_result["NI_reactor"],
        "I_turn": jc_result["I_turn"],
        "B_peak_estimate": jc_result["B_peak_estimate"],
        "jc_tape_stack_A_per_m2": jc_result["jc_tape_stack"],
        "Ic_cable_A": jc_result["Ic_cable"],
        "params": jc_result["model_params"],
    }

    turn_side_m = np.sqrt(STELLARIS_A_TURN)
    wp_widths = [float(np.sqrt(n) * turn_side_m) for n in n_turns_per_coil]
    reactor_metrics["winding_pack_width_per_coil"] = wp_widths
    max_wp = float(max(wp_widths)) if wp_widths else 0.0
    reactor_metrics["max_winding_pack_width"] = max_wp

    d_cc_rs = reactor_metrics.get("reactor_scale_min_cc_separation")
    if d_cc_rs is not None and max_wp > 0:
        reactor_metrics["finite_build_cc_clearance"] = float(d_cc_rs - max_wp)

    per_turn_forces = [f / n for f, n in zip(reactor_force_per_coil, n_turns_per_coil)]
    reactor_metrics["per_turn_max_force"] = float(max(per_turn_forces))

    per_coil_torques = metrics.get("final_max_torque_per_coil")
    if per_coil_torques is not None and len(per_coil_torques) == len(n_turns_per_coil):
        reactor_torque_per_coil = [t * torque_scale / 1e6 for t in per_coil_torques]
        per_turn_torques = [
            t / n for t, n in zip(reactor_torque_per_coil, n_turns_per_coil)
        ]
        reactor_metrics["per_turn_max_torque"] = float(max(per_turn_torques))
    elif "reactor_scale_max_max_coil_torque" in reactor_metrics:
        max_tau = reactor_metrics["reactor_scale_max_max_coil_torque"]
        min_n = min(n_turns_per_coil)
        reactor_metrics["per_turn_max_torque"] = float(max_tau / min_n)

    if per_coil_lengths is not None and len(per_coil_lengths) == len(n_turns_per_coil):
        reactor_lengths = [ln * L_scale for ln in per_coil_lengths]
        total_sc_km = (
            sum(n * ln for n, ln in zip(n_turns_per_coil, reactor_lengths)) / 1e3
        )
        reactor_metrics["total_superconductor_length_km"] = float(total_sc_km)
    elif "final_total_length" in metrics:
        num_coils = len(n_turns_per_coil)
        avg_len = float(metrics["final_total_length"]) * L_scale / num_coils
        total_sc_km = sum(n * avg_len for n in n_turns_per_coil) / 1e3
        reactor_metrics["total_superconductor_length_km"] = float(total_sc_km)


def compute_reactor_scale_metrics(
    metrics: dict[str, Any],
    case_cfg: "CaseConfig | dict[str, Any] | None" = None,
    already_reactor_scaled: bool = False,
) -> dict[str, Any]:
    """Convert final device-scale metrics to reactor-scale equivalents.

    When already_reactor_scaled is True, the coil metrics are already in ARIES-CS
    reactor-scale (a=1.7 m, B=5.7 T); no scaling is applied (L_scale=1, B_scale=1).
    Use this for external coil sets (e.g. Zenodo) that were produced at reactor scale.

    Scaling relationships (L = L_reactor/L_device, B = B_reactor/B_device):

    - Lengths [m] (d_cc, d_cs, total_length): × L
    - Curvature [1/m] (κ): × 1/L
    - Mean squared curvature [1/m²]: × 1/L²
    - Force/length [N/m → MN/m]: × B²L / 1e6
    - Torque/length [N → MN]: × B²L² / 1e6
    - Arclength variation [m²]: × L²
    - SquaredFlux [T²m²]: × B²L²
    - Normalised quantities (B·n/|B|, linking_number): no scaling

    Also computes per-coil quantities:

    - **N_turns_per_coil** = max(N_force, N_jc) — force-based and REBCO-Jc-
      based turn counts (see ``compute_N_turns_critical_current``).
    - **winding_pack_width_per_coil** — finite-build side length
      w = sqrt(N_turns) × 20 mm (Stellaris geometry).
    - **finite_build_cc_clearance** — d_cc_min − w_max.  Negative values
      indicate the finite-build winding packs would physically overlap.
    - **total_superconductor_length_km** — Σ_i N_turns_i × length_i / 1000.

    Returns
    -------
    dict
        Reactor-scale metrics, scaling factors, and derived winding-pack data.
    """
    from .path_utils import get_surface_search_base_dirs, resolve_surface_path

    reactor_metrics: dict = {
        "reference": REACTOR_REFERENCE.copy(),
    }

    target_B = metrics.get("target_B_field", None)
    cached = metrics.get("_cached_thresholds", {})
    major_radius = cached.get("major_radius", None)
    minor_radius = cached.get("minor_radius", None)

    if major_radius is None and case_cfg is not None:
        try:
            from .path_utils import get_reference_radii, get_surface_filename

            surface_name = get_surface_filename(case_cfg)
            if surface_name:
                base_dirs = get_surface_search_base_dirs()
                surface_path = resolve_surface_path(surface_name, base_dirs)
                if surface_path is not None:
                    from .post_processing import load_surface_with_range

                    s = load_surface_with_range(
                        surface_path,
                        surface_range="half period",
                        nphi=16,
                        ntheta=16,
                    )
                    major_radius, minor_radius = (
                        float(x) for x in get_reference_radii(s)
                    )
        except Exception as exc:
            logger.debug("Failed to load surface for reactor-scale metrics: %s", exc)

    if minor_radius is None and case_cfg is not None:
        try:
            from .path_utils import get_reference_radii, get_surface_filename

            surface_name = get_surface_filename(case_cfg)
            if surface_name:
                base_dirs = get_surface_search_base_dirs()
                surface_path = resolve_surface_path(surface_name, base_dirs)
                if surface_path is not None:
                    from .post_processing import load_surface_with_range

                    s = load_surface_with_range(
                        surface_path,
                        surface_range="half period",
                        nphi=16,
                        ntheta=16,
                    )
                    _, minor_radius = get_reference_radii(s)
                    minor_radius = float(minor_radius)
        except Exception as exc:
            logger.debug(
                "Failed to load surface for minor_radius: %s", exc
            )

    if major_radius is None or target_B is None:
        reactor_metrics["error"] = "Could not determine device scale parameters"
        return reactor_metrics

    if minor_radius is None or minor_radius <= 0:
        reactor_metrics["error"] = "Could not determine minor_radius for scaling"
        return reactor_metrics

    if already_reactor_scaled:
        L_scale = 1.0
        B_scale = 1.0
        reactor_metrics["scaling_factors"] = {
            "length_scale": 1.0,
            "B_field_scale": 1.0,
            "device_major_radius": float(major_radius),
            "device_target_B": float(REACTOR_REFERENCE["B_field"]),
            "already_reactor_scaled": True,
        }
    else:
        L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
        B_scale = REACTOR_REFERENCE["B_field"] / target_B

        reactor_metrics["scaling_factors"] = {
            "length_scale": float(L_scale),
            "B_field_scale": float(B_scale),
            "device_major_radius": float(major_radius),
            "device_target_B": float(target_B),
        }

    scaled = _apply_reactor_scaling(metrics, L_scale, B_scale)
    reactor_metrics.update(scaled)

    _compute_turn_counts(metrics, reactor_metrics, L_scale, B_scale, target_B)

    return reactor_metrics
