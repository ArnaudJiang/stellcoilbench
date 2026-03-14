"""Verbose iteration output formatting for scipy minimize callback."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def _format_verbose_iteration_output(
    iteration: int,
    Jls: list,
    Jccdist: Any,
    Jcsdist: Any,
    base_curves: list,
    Jlink: Any,
    grad: np.ndarray,
    weights: list,
    c_list: list,
    constraint_names_and_thresholds: list,
    J_total: float,
    *,
    max_force: float | None = None,
    max_torque: float | None = None,
    structural_obj: Any | None = None,
    Jts: list | None = None,
) -> tuple[str, str]:
    """
    Format verbose iteration output for scipy minimize callback.

    Builds two lines: (1) main line with iteration, L, d_cc, d_cs, κ, MSC,
    LN, F, Tq, ζ, ‖∇J‖; (2) contrib line with weighted contributions
    (weight × J) per term and total.

    F and Tq are shown when max_force/max_torque are provided (i.e. when
    coil_coil_force/coil_coil_torque are in coil_objective_terms). F and Tq
    report the max pointwise force per unit length [N/m] and torque per unit
    length [N], consistent with κ (max curvature) and ζ (max torsion).

    Parameters
    ----------
    iteration : int
        Current iteration number.
    Jls, Jccdist, Jcsdist, Jlink : objectives
        Constraint objectives for value extraction.
    max_force, max_torque : float | None, optional
        Max pointwise force [N/m] and torque per unit length [N] across coils.
        When provided, F= and Tq= are appended to the main line.
    Jts : list | None
        Per-coil LpCurveTorsion objectives.  If None, torsion is omitted from output.
    base_curves : list
        Base coil curves (for κ, MSC).
    grad : np.ndarray
        Gradient vector for ‖∇J‖.
    weights : list
        Per-constraint weights.
    c_list : list
        Constraint objectives.
    constraint_names_and_thresholds : list
        (name, threshold) pairs for contrib labels.
    J_total : float
        Total weighted objective value.
    structural_obj : StructuralStressObjective | None
        Optional. When provided, appends σ_vm (Von Mises stress) to the main line.

    Returns
    -------
    tuple[str, str]
        (main_line, contrib_line) - formatted strings for printing.
    """
    from simsopt.geo import MeanSquaredCurvature

    outstr = f"[{iteration}]"

    def _torsion_per_coil(curves: list) -> str:
        """Max |torsion| per coil, formatted like κ."""
        vals = [float(np.max(np.abs(np.asarray(c.torsion())))) for c in curves]
        return ",".join([f"{v:.2f}" for v in vals])

    outstr += f" L={sum(J.J() for J in Jls):.2f}"
    kappa_values = [c.kappa().max() for c in base_curves]
    msc_values = [MeanSquaredCurvature(c).J() for c in base_curves]
    kappa_str = ",".join([f"{k:.1f}" for k in kappa_values])
    msc_str = ",".join([f"{m:.1f}" for m in msc_values])
    outstr += f", κ=[{kappa_str}]"
    outstr += f", MSC=[{msc_str}]"
    if Jts is not None and len(Jts) > 0:
        outstr += f", ζ=[{_torsion_per_coil(base_curves)}]"

    outstr += f", d_cc={Jccdist.shortest_distance():.2f}, d_cs={Jcsdist.shortest_distance():.2f}"
    if Jlink is not None:
        try:
            outstr += f", LN={int(round(Jlink.J()))}"
        except Exception as e:
            logger.debug("Skipping Jlink in iteration string: %s", e)
    if max_force is not None:
        outstr += f", F={max_force:.2e}"
    if max_torque is not None:
        outstr += f", Tq={max_torque:.2e}"
    if structural_obj is not None:
        try:
            # Use raw objective for true max von Mises (bypass guard for display)
            raw_obj = structural_obj.objective.objective
            sigma_vm = abs(raw_obj.J())
            outstr += f", σ_vm={sigma_vm:.3e} GPa"
        except Exception:
            try:
                sigma_vm = abs(
                    structural_obj.J()
                )  # fallback to guarded value if FEM fails
                outstr += f", σ_vm={sigma_vm:.3e} GPa"
            except Exception as e:
                logger.debug("Skipping structural_obj in iteration string: %s", e)
    outstr += f", ‖∇J‖={np.linalg.norm(grad):.1e}"

    name_short: Dict[str, str] = {
        "Flux": "J_f",
        "CC Distance": "d_cc",
        "CS Distance": "d_cs",
        "Length": "L",
        "MSC": "MSC",
        "Arclength Var": "Var",
        "κ": "κ",
        "Link #": "LN",
        "Force": "F",
        "Torque": "Tq",
        "ζ": "J_ζ",
    }
    scope_map: Dict[str, str] = {}
    contrib_parts = []
    flux_short = name_short.get("Flux", "Flux")
    flux_scope = scope_map.get("Flux", "")
    flux_label = f"{flux_short} ({flux_scope})" if flux_scope else flux_short

    # Iterate over all c_list items so displayed contributions sum to J_total
    for idx in range(min(len(c_list), len(weights))):
        contrib = weights[idx] * c_list[idx].J()
        if idx == 0:
            label = flux_label
        elif idx - 1 < len(constraint_names_and_thresholds):
            name = constraint_names_and_thresholds[idx - 1][0]
            short = name_short.get(name, name)
            scope = scope_map.get(name, "")
            label = f"{short} ({scope})" if scope else short
        else:
            label = f"c[{idx}]"
        contrib_parts.append(f"{label}={contrib:.1e}")
    contrib_str = "Objs: " + ", ".join(contrib_parts)
    contrib_str += f", Total={J_total:.1e}"

    return outstr, contrib_str
