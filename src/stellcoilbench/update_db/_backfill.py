"""Backfill reactor-scale metrics for older submissions.

Older submissions may lack N_turns_jc, winding_pack_width_per_coil,
finite_build_cc_clearance, per_turn_max_force/torque, or
total_superconductor_length_km. This module recomputes them from stored
device-scale metrics and scaling factors.
"""

from __future__ import annotations

from typing import Any, Dict


def backfill_reactor_scale_metrics(
    reactor_scale: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    """Backfill missing reactor-scale metrics for older submissions.

    Mutates *reactor_scale* in place. Handles:
    - N_turns_jc and N_turns_per_coil (Jc model when only force-based N_turns)
    - winding_pack_width_per_coil and max_winding_pack_width
    - finite_build_cc_clearance
    - per_turn_max_force and per_turn_max_torque
    - total_superconductor_length_km

    Parameters
    ----------
    reactor_scale : dict
        Reactor-scale metrics dict (mutated in place).
    metrics : dict
        Device-scale metrics (final_max_force_per_coil, etc.).
    """
    # ---- Backfill Jc-based N_turns and total SC length for older submissions ----
    if "N_turns_jc" not in reactor_scale and "N_turns_per_coil" in reactor_scale:
        n_turns_force = reactor_scale["N_turns_per_coil"]
        per_coil_forces_dev = metrics.get("final_max_force_per_coil")
        if (
            isinstance(n_turns_force, list)
            and n_turns_force
            and isinstance(per_coil_forces_dev, list)
            and len(per_coil_forces_dev) == len(n_turns_force)
        ):
            sf = reactor_scale.get("scaling_factors", {})
            L_scale_est = sf.get("length_scale")
            B_scale_est = sf.get("B_field_scale")
            target_B_est = sf.get("device_target_B", metrics.get("target_B_field"))
            if L_scale_est and B_scale_est and target_B_est:
                from stellcoilbench.reactor_scale import (
                    compute_N_turns_critical_current,
                )

                per_coil_currents = metrics.get("final_current_per_coil")
                per_coil_lengths = metrics.get("final_length_per_coil")
                jc_result = compute_N_turns_critical_current(
                    per_coil_forces=per_coil_forces_dev,
                    per_coil_currents=per_coil_currents,
                    per_coil_lengths=per_coil_lengths,
                    L_scale=L_scale_est,
                    B_scale=B_scale_est,
                    target_B=target_B_est,
                )
                n_turns_jc = jc_result["N_turns_jc"]
                new_n_turns = [max(nf, nj) for nf, nj in zip(n_turns_force, n_turns_jc)]
                reactor_scale["N_turns_per_coil"] = new_n_turns
                reactor_scale["N_turns_force"] = list(n_turns_force)
                reactor_scale["N_turns_jc"] = n_turns_jc

    # ---- Backfill winding_pack_width_per_coil ----
    n_turns_wp = reactor_scale.get("N_turns_per_coil")
    if (
        isinstance(n_turns_wp, list)
        and n_turns_wp
        and "max_winding_pack_width" not in reactor_scale
    ):
        import numpy as np

        from stellcoilbench.reactor_scale import STELLARIS_A_TURN

        turn_side = np.sqrt(STELLARIS_A_TURN)
        wp_widths = [float(np.sqrt(n) * turn_side) for n in n_turns_wp]
        reactor_scale["winding_pack_width_per_coil"] = wp_widths
        reactor_scale["max_winding_pack_width"] = float(max(wp_widths))

    # ---- Backfill finite_build_cc_clearance ----
    max_wp = reactor_scale.get("max_winding_pack_width")
    d_cc_rs = reactor_scale.get("reactor_scale_min_cc_separation")
    if (
        max_wp is not None
        and d_cc_rs is not None
        and "finite_build_cc_clearance" not in reactor_scale
    ):
        reactor_scale["finite_build_cc_clearance"] = float(d_cc_rs - max_wp)

    # ---- Backfill per_turn_max_force and per_turn_max_torque ----
    n_turns_pt = reactor_scale.get("N_turns_per_coil")
    if (
        isinstance(n_turns_pt, list)
        and n_turns_pt
        and "per_turn_max_force" not in reactor_scale
    ):
        rs_forces = reactor_scale.get("reactor_scale_force_per_coil_MN_per_m")
        if isinstance(rs_forces, list) and len(rs_forces) == len(n_turns_pt):
            per_turn_f = [f / n for f, n in zip(rs_forces, n_turns_pt)]
            reactor_scale["per_turn_max_force"] = float(max(per_turn_f))
        elif reactor_scale.get("reactor_scale_max_max_coil_force") is not None:
            reactor_scale["per_turn_max_force"] = float(
                reactor_scale["reactor_scale_max_max_coil_force"] / min(n_turns_pt)
            )
    if (
        isinstance(n_turns_pt, list)
        and n_turns_pt
        and "per_turn_max_torque" not in reactor_scale
    ):
        rs_torque_max = reactor_scale.get("reactor_scale_max_max_coil_torque")
        if rs_torque_max is not None:
            reactor_scale["per_turn_max_torque"] = float(
                rs_torque_max / min(n_turns_pt)
            )

    # ---- Backfill total_superconductor_length_km ----
    n_turns = reactor_scale.get("N_turns_per_coil")
    if (
        isinstance(n_turns, list)
        and n_turns
        and "total_superconductor_length_km" not in reactor_scale
    ):
        per_coil_len = metrics.get("final_length_per_coil")
        if isinstance(per_coil_len, list) and len(per_coil_len) == len(n_turns):
            rs_total_len = reactor_scale.get("reactor_scale_total_length")
            if rs_total_len is not None:
                device_total = metrics.get("final_total_length")
                if device_total and device_total > 0:
                    L_scale_est = rs_total_len / device_total
                else:
                    L_scale_est = (
                        rs_total_len / sum(per_coil_len)
                        if sum(per_coil_len) > 0
                        else 1.0
                    )
                reactor_lengths = [ln * L_scale_est for ln in per_coil_len]
                reactor_scale["total_superconductor_length_km"] = float(
                    sum(n * ln for n, ln in zip(n_turns, reactor_lengths)) / 1e3
                )
        elif "reactor_scale_total_length" in reactor_scale:
            rs_total = reactor_scale["reactor_scale_total_length"]
            num_coils = len(n_turns)
            avg_len = rs_total / num_coils
            reactor_scale["total_superconductor_length_km"] = float(
                sum(n * avg_len for n in n_turns) / 1e3
            )
