"""Optimizer parameters validation for case configs."""

from __future__ import annotations

from typing import Any

__all__ = ["_validate_optimizer_params"]


def _validate_optimizer_params(optimizer_params: Any, pfx: str) -> list[str]:
    """Validate the ``optimizer_params`` section of a case config.

    Parameters
    ----------
    optimizer_params : Any
        Value of ``data["optimizer_params"]``.
    pfx : str
        Error-message prefix.

    Returns
    -------
    list[str]
        Error messages.
    """
    errors: list[str] = []
    if not isinstance(optimizer_params, dict):
        return [f"{pfx}optimizer_params must be a dictionary"]

    if "max_iterations" in optimizer_params:
        val = optimizer_params["max_iterations"]
        if not isinstance(val, int) or val < 1:
            errors.append(
                f"{pfx}optimizer_params.max_iterations must be a positive integer"
            )
    if "max_iter_lag" in optimizer_params:
        val = optimizer_params["max_iter_lag"]
        if not isinstance(val, int) or val < 1:
            errors.append(
                f"{pfx}optimizer_params.max_iter_lag must be a positive integer"
            )
    return errors
