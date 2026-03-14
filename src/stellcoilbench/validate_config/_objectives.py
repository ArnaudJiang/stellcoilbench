"""Validation for coil_objective_terms and fourier_continuation sections."""

from __future__ import annotations

from typing import Any

from ._common import _is_valid_non_negative_number, _is_valid_positive_number

# Maps objective term name → allowed option values
_TERM_OPTIONS: dict[str, list[str]] = {
    "total_length": ["l2", "l2_threshold"],
    "coil_coil_distance": [""],
    "coil_surface_distance": [""],
    "coil_curvature": ["lp", "lp_threshold"],
    "coil_torsion": ["lp", "lp_threshold"],
    "coil_arclength_variation": ["l2", "l2_threshold", "l1", "l1_threshold"],
    "coil_mean_squared_curvature": ["l2", "l2_threshold", "l1", "l1_threshold"],
    "linking_number": [""],
    "coil_coil_force": ["lp", "lp_threshold"],
    "coil_coil_torque": ["lp", "lp_threshold"],
    "structural_stress": ["l1", "l1_threshold", "l2", "l2_threshold"],
}

_VALID_THRESHOLD_NAMES = {
    "length_threshold",
    "cc_threshold",
    "cs_threshold",
    "curvature_threshold",
    "torsion_threshold",
    "arclength_variation_threshold",
    "msc_threshold",
    "force_threshold",
    "torque_threshold",
    "flux_threshold",
    "structural_stress_threshold",
}

_VALID_WEIGHT_NAMES = {
    "length_weight",
    "cc_weight",
    "cs_weight",
    "curvature_weight",
    "torsion_weight",
    "arclength_variation_weight",
    "msc_weight",
    "force_weight",
    "torque_weight",
    "flux_weight",
    "linking_weight",
    "structural_stress_weight",
}

_VALID_STRUCTURAL_PARAMS = {
    "structural_mesh_resolution_coarse",
    "structural_mesh_resolution_fine",
    "structural_E",
    "structural_nu",
    "structural_eval_interval",
    "structural_stress_metric",
    "structural_fd_step",
    "structural_quadrature_degree",
    "structural_polynomial_degree",
    "structural_refine_stress_ratio",
    "structural_use_cached_K",
    "structural_backend",
}


def _validate_objective_terms(obj_terms: Any, pfx: str) -> list[str]:
    """Validate the coil_objective_terms section of a case config."""
    errors: list[str] = []
    if not isinstance(obj_terms, dict):
        return [f"{pfx}coil_objective_terms must be a dictionary"]

    all_valid_keys = (
        set(_TERM_OPTIONS)
        | _VALID_THRESHOLD_NAMES
        | _VALID_WEIGHT_NAMES
        | _VALID_STRUCTURAL_PARAMS
    )

    for term_name, term_value in obj_terms.items():
        if term_name in _VALID_THRESHOLD_NAMES or term_name in _VALID_WEIGHT_NAMES:
            if not _is_valid_non_negative_number(term_value):
                errors.append(
                    f"{pfx}coil_objective_terms.{term_name} must be a non-negative number"
                )
            continue

        if term_name in _VALID_STRUCTURAL_PARAMS:
            if term_name == "structural_stress_metric":
                allowed_metrics = ("max_von_mises", "mean_von_mises", "lp_von_mises")
                if term_value not in allowed_metrics:
                    errors.append(
                        f"{pfx}coil_objective_terms.{term_name} must be one of "
                        f"{allowed_metrics}, got '{term_value}'"
                    )
            elif term_name == "structural_eval_interval":
                if not isinstance(term_value, int) or term_value < 1:
                    errors.append(
                        f"{pfx}coil_objective_terms.{term_name} must be a positive integer"
                    )
            elif term_name == "structural_use_cached_K":
                if not isinstance(term_value, bool):
                    errors.append(
                        f"{pfx}coil_objective_terms.{term_name} must be a boolean"
                    )
            elif term_name == "structural_backend":
                if term_value not in ("dolfinx", "skfem"):
                    errors.append(
                        f"{pfx}coil_objective_terms.{term_name} must be "
                        "'dolfinx' or 'skfem'"
                    )
            elif term_name == "structural_quadrature_degree":
                try:
                    qd = int(term_value)
                except (TypeError, ValueError):
                    qd = -1
                if qd not in (1, 2):
                    errors.append(
                        f"{pfx}coil_objective_terms.{term_name} must be 1 or 2"
                    )
            elif term_name == "structural_polynomial_degree":
                try:
                    pd = int(term_value)
                except (TypeError, ValueError):
                    pd = -1
                if pd not in (1, 2, 3):
                    errors.append(
                        f"{pfx}coil_objective_terms.{term_name} must be 1, 2, or 3"
                    )
            elif term_name == "structural_refine_stress_ratio":
                if not _is_valid_positive_number(term_value):
                    errors.append(
                        f"{pfx}coil_objective_terms.{term_name} must be a positive number"
                    )
            elif not _is_valid_positive_number(term_value):
                errors.append(
                    f"{pfx}coil_objective_terms.{term_name} must be a positive number"
                )
            continue

        if term_name.endswith("_p"):
            if not _is_valid_positive_number(term_value):
                errors.append(
                    f"{pfx}coil_objective_terms.{term_name} must be a positive number"
                )
            continue

        if term_name not in _TERM_OPTIONS:
            errors.append(
                f"{pfx}Unknown coil_objective_terms key: '{term_name}'. "
                f"Valid keys: {sorted(all_valid_keys)}"
            )
            continue

        allowed = _TERM_OPTIONS[term_name]
        if term_value not in allowed:
            hint = ""
            if term_name in ("coil_coil_distance", "coil_surface_distance"):
                threshold_key = (
                    "cc_threshold" if "coil" in term_name[:10] else "cs_threshold"
                )
                hint = (
                    f" It is always included automatically - "
                    f"use {threshold_key} to set threshold."
                )
            errors.append(
                f"{pfx}coil_objective_terms.{term_name} must be one of "
                f"{allowed}, got '{term_value}'.{hint}"
            )

    return errors


def _validate_fourier_continuation(fc: Any, pfx: str) -> list[str]:
    """Validate the fourier_continuation section of a case config."""
    errors: list[str] = []
    if not isinstance(fc, dict):
        return [f"{pfx}fourier_continuation must be a dictionary"]

    if "enabled" in fc and not isinstance(fc["enabled"], bool):
        errors.append(f"{pfx}fourier_continuation.enabled must be a boolean")

    if "orders" in fc:
        orders = fc["orders"]
        if not isinstance(orders, list):
            errors.append(f"{pfx}fourier_continuation.orders must be a list")
        elif not orders:
            errors.append(f"{pfx}fourier_continuation.orders must be non-empty")
        elif not all(isinstance(o, int) and o > 0 for o in orders):
            errors.append(
                f"{pfx}fourier_continuation.orders must contain only positive integers"
            )
        elif orders != sorted(orders):
            errors.append(
                f"{pfx}fourier_continuation.orders must be in ascending order"
            )
    return errors
