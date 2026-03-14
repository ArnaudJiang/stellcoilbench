"""CI autopilot case JSON validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._case import validate_case_config
from ._common import _validate_positive_int_field, _validate_positive_number_field

# Default resource caps (can be overridden by policy)
_DEFAULT_MAX_TOTAL_ITERATIONS = 10000
_DEFAULT_TIMEOUT_MINUTES_MIN = 5
_DEFAULT_TIMEOUT_MINUTES_MAX = 180


def validate_ci_case(
    data: dict[str, Any],
    policy: dict[str, Any] | None = None,
    file_path: Path | None = None,
) -> list[str]:
    """Validate a CI autopilot case JSON dictionary.

    The CI case wraps a standard case config inside a ``case_config`` key and
    adds ``case_id``, ``resource``, and optional ``parent_ids`` / ``tags`` /
    ``random_seed`` fields.

    Parameters
    ----------
    data : dict
        Parsed JSON for the CI case.
    policy : dict, optional
        Proposer policy (from ``policy/proposer_policy.yaml``).  If provided,
        resource caps are taken from ``policy["resource_caps"]``.
    file_path : Path, optional
        Used for error-message prefixes.

    Returns
    -------
    list[str]
        Error messages.  Empty list means validation passed.
    """
    errors: list[str] = []
    pfx = f"{file_path}: " if file_path else ""

    caps = (policy or {}).get("resource_caps", {})
    max_iter_cap = caps.get("max_total_iterations", _DEFAULT_MAX_TOTAL_ITERATIONS)
    timeout_min = caps.get("timeout_minutes_min", _DEFAULT_TIMEOUT_MINUTES_MIN)
    timeout_max = caps.get("timeout_minutes_max", _DEFAULT_TIMEOUT_MINUTES_MAX)

    if "case_id" not in data:
        errors.append(f"{pfx}Missing required field: case_id")
    elif not isinstance(data["case_id"], str) or not data["case_id"]:
        errors.append(f"{pfx}case_id must be a non-empty string")

    resource = data.get("resource", {})
    if not isinstance(resource, dict):
        errors.append(f"{pfx}resource must be a dictionary")
    else:
        errors.extend(
            _validate_positive_int_field(
                resource,
                "max_total_iterations",
                pfx,
                "resource",
                max_val=max_iter_cap,
            )
        )
        errors.extend(
            _validate_positive_number_field(
                resource,
                "timeout_minutes",
                pfx,
                "resource",
                min_val=timeout_min,
                max_val=timeout_max,
            )
        )

    if "parent_ids" in data:
        if not isinstance(data["parent_ids"], list):
            errors.append(f"{pfx}parent_ids must be a list")
    if "tags" in data:
        if not isinstance(data["tags"], list):
            errors.append(f"{pfx}tags must be a list")
    if "random_seed" in data:
        if not isinstance(data["random_seed"], int):
            errors.append(f"{pfx}random_seed must be an integer")

    cc = data.get("case_config")
    if cc is None:
        errors.append(f"{pfx}Missing required field: case_config")
    elif not isinstance(cc, dict):
        errors.append(f"{pfx}case_config must be a dictionary")
    else:
        inner = validate_case_config(cc, file_path)
        errors.extend(inner)

        opt = cc.get("optimizer_params", {})
        maxiter = opt.get("max_iterations")
        if isinstance(maxiter, int) and maxiter > max_iter_cap:
            errors.append(
                f"{pfx}case_config.optimizer_params.max_iterations ({maxiter}) "
                f"exceeds cap ({max_iter_cap})"
            )

    return errors


def validate_ci_case_file(
    file_path: Path,
    policy: dict[str, Any] | None = None,
) -> list[str]:
    """Validate a CI autopilot case JSON file on disk.

    Loads the file with json.load, then delegates to validate_ci_case.

    Parameters
    ----------
    file_path : Path
        Path to the CI case JSON file.
    policy : dict[str, Any], optional
        Proposer policy for resource caps.

    Returns
    -------
    list[str]
        Error messages. Empty list means validation passed.
    """
    try:
        with open(file_path, "r") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        return [f"{file_path}: JSON parse error: {exc}"]
    except OSError as exc:
        return [f"{file_path}: Error reading file: {exc}"]

    if not isinstance(data, dict):
        return [f"{file_path}: Root element must be a JSON object"]

    return validate_ci_case(data, policy=policy, file_path=file_path)
