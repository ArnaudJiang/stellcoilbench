"""Case YAML configuration validation.

Orchestrates validation of surface_params, coils_params, optimizer_params,
coil_objective_terms, and fourier_continuation sections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yaml import YAMLError

from ._algorithm import _validate_optimizer_params
from ._coils import _validate_coils_params
from ._objectives import (
    _validate_finite_section_field,
    _validate_fourier_continuation,
    _validate_objective_terms,
)
from ._surface import _validate_surface_exists, _validate_surface_params


def validate_case_config(
    data: dict[str, Any],
    file_path: Path | None = None,
    surfaces_dir: Path | None = None,
) -> list[str]:
    """Validate a case.yaml configuration dictionary.

    Checks required fields (description, surface_params, coils_params,
    optimizer_params), valid keys for each section, type/value constraints,
    and that the referenced surface file exists in plasma_surfaces/.

    Parameters
    ----------
    data : dict[str, Any]
        Parsed YAML/JSON configuration to validate.
    file_path : Path, optional
        Path to source file; used for error message prefixes.
    surfaces_dir : Path, optional
        Directory containing surface files. If None, uses repo-relative
        ``plasma_surfaces/``.

    Returns
    -------
    list[str]
        Error messages. Empty list means validation passed.
    """
    errors: list[str] = []
    pfx = f"{file_path}: " if file_path else ""

    for field in ("description", "surface_params", "coils_params", "optimizer_params"):
        if field not in data:
            errors.append(f"{pfx}Missing required field: {field}")

    if "surface_params" in data:
        errors.extend(_validate_surface_params(data["surface_params"], pfx))
        errors.extend(
            _validate_surface_exists(
                data["surface_params"], pfx, surfaces_dir=surfaces_dir
            )
        )
    if "coils_params" in data:
        errors.extend(_validate_coils_params(data["coils_params"], pfx))
    if "optimizer_params" in data:
        errors.extend(
            _validate_optimizer_params(
                data["optimizer_params"],
                pfx,
                focus_params=data.get("focus_params"),
            )
        )
    if "coil_objective_terms" in data:
        errors.extend(_validate_objective_terms(data["coil_objective_terms"], pfx))
    if "fourier_continuation" in data:
        errors.extend(_validate_fourier_continuation(data["fourier_continuation"], pfx))
    if "finite_section_field" in data:
        errors.extend(_validate_finite_section_field(data["finite_section_field"], pfx))

    return errors


def validate_case_yaml_file(
    file_path: Path,
    surfaces_dir: Path | None = None,
) -> list[str]:
    """Validate a case.yaml file on disk.

    Loads the file with load_yaml, then delegates to validate_case_config.

    Parameters
    ----------
    file_path : Path
        Path to the case YAML file.
    surfaces_dir : Path, optional
        Directory containing plasma surface files. If None, uses repo-relative
        plasma_surfaces/.

    Returns
    -------
    list[str]
        Error messages. Empty list means validation passed.
    """
    try:
        from ..path_utils import load_yaml

        data = load_yaml(path=file_path)

        if data is None or (isinstance(data, dict) and not data):
            return [f"{file_path}: File is empty or contains no valid YAML"]

        if not isinstance(data, dict):
            return [f"{file_path}: Root element must be a dictionary"]

        return validate_case_config(
            data, file_path=file_path, surfaces_dir=surfaces_dir
        )
    except YAMLError as e:
        return [f"{file_path}: YAML parsing error: {e}"]
    except OSError as e:
        return [f"{file_path}: Error reading file: {e}"]
