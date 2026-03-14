"""
Validation functions for case.yaml configuration files and CI autopilot case JSON.

Validates case YAML (surface_params, coils_params, optimizer_params,
coil_objective_terms, fourier_continuation, and related fields) and CI case JSON
(case_id, resource caps, case_config embedding, policy limits).
"""

from __future__ import annotations

from ._case import validate_case_config, validate_case_yaml_file
from ._ci import validate_ci_case, validate_ci_case_file
from ._common import (
    _is_valid_non_negative_number,
    _is_valid_positive_number,
)
from ._surface import _validate_surface_exists

__all__ = [
    "_is_valid_non_negative_number",
    "_is_valid_positive_number",
    "_validate_surface_exists",
    "validate_case_config",
    "validate_case_yaml_file",
    "validate_ci_case",
    "validate_ci_case_file",
]
