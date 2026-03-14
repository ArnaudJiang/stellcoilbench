"""Shared validation helpers for case and CI config validation."""

from __future__ import annotations

from typing import Any


def _is_valid_number(value: Any, min_val: float, allow_equal: bool = True) -> bool:
    """Return True if value is a number (int, float, or parseable string) >= min_val. Rejects bool."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value >= min_val if allow_equal else value > min_val
    if isinstance(value, str):
        try:
            f = float(value)
            return f >= min_val if allow_equal else f > min_val
        except ValueError:
            return False
    return False


def _is_valid_non_negative_number(value: Any) -> bool:
    """Check if *value* is a non-negative number (int, float, or parseable string)."""
    return _is_valid_number(value, 0.0, allow_equal=True)


def _is_valid_positive_number(value: Any) -> bool:
    """Check if *value* is a strictly positive number."""
    return _is_valid_number(value, 0.0, allow_equal=False)


def _validate_positive_int_field(
    section: dict[str, Any],
    field: str,
    pfx: str,
    section_name: str,
    max_val: int | None = None,
) -> list[str]:
    """Return an error if *section[field]* exists but is not a positive int.

    Also flags float values that represent integers (e.g. 5.0).
    If max_val is provided, also flags values exceeding the cap.
    """
    errors: list[str] = []
    if field not in section:
        return errors
    val = section[field]
    if isinstance(val, float) and val.is_integer():
        errors.append(
            f"{pfx}{section_name}.{field} should be an integer, not a float. "
            f"Got {val}. Use {int(val)} instead."
        )
    elif not isinstance(val, int) or val < 1:
        errors.append(
            f"{pfx}{section_name}.{field} must be a positive integer, "
            f"got {type(val).__name__}: {val}"
        )
    elif max_val is not None and val > max_val:
        errors.append(f"{pfx}{section_name}.{field} ({val}) exceeds cap ({max_val})")
    return errors


def _validate_positive_number_field(
    section: dict[str, Any],
    field: str,
    pfx: str,
    section_name: str,
    min_val: float,
    max_val: float,
) -> list[str]:
    """Return errors if section[field] exists but is not a positive number in [min_val, max_val]."""
    errors: list[str] = []
    if field not in section:
        return errors
    val = section[field]
    if not isinstance(val, (int, float)) or val <= 0:
        errors.append(
            f"{pfx}{section_name}.{field} must be a positive number, "
            f"got {type(val).__name__}: {val}"
        )
    elif val < min_val or val > max_val:
        errors.append(
            f"{pfx}{section_name}.{field} ({val}) outside allowed range "
            f"[{min_val}, {max_val}]"
        )
    return errors
