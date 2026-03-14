"""Coils parameters validation for case configs."""

from __future__ import annotations

from typing import Any

from ._common import _validate_positive_int_field

__all__ = ["_validate_coils_params"]


def _validate_coils_params(coils_params: Any, pfx: str) -> list[str]:
    """Validate the ``coils_params`` section of a case config.

    Parameters
    ----------
    coils_params : Any
        Value of ``data["coils_params"]``.
    pfx : str
        Error-message prefix.

    Returns
    -------
    list[str]
        Error messages.
    """
    errors: list[str] = []
    if not isinstance(coils_params, dict):
        return [f"{pfx}coils_params must be a dictionary"]

    valid_keys = {
        "ncoils",
        "order",
        "coil_type",
        "vv_extension",
        "inboard_radius",
        "wp_fil_spacing",
        "half_per_spacing",
        "wp_n",
        "numquadpoints",
        "fix_shapes",
        "fix_currents",
        "fix_center",
        "fix_orientation",
    }
    for key in coils_params:
        if key not in valid_keys:
            errors.append(
                f"{pfx}Unknown coils_params key: '{key}'. "
                f"Valid keys: {sorted(valid_keys)}. "
                f"Note: 'target_B' is no longer used (determined from surface file)."
            )

    coil_type = coils_params.get("coil_type", "modular")
    if coil_type == "dipole":
        errors.append(
            f"{pfx}coil_type 'dipole' has been removed. Use coil_type: 'modular' instead."
        )
    if "ncoils" not in coils_params:
        errors.append(f"{pfx}coils_params must include 'ncoils'.")

    errors.extend(
        _validate_positive_int_field(coils_params, "ncoils", pfx, "coils_params")
    )
    errors.extend(
        _validate_positive_int_field(coils_params, "order", pfx, "coils_params")
    )
    return errors
