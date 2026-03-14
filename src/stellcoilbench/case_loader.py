"""
Case loading and path resolution for StellCoilBench.

Provides a single entry point for loading, validating, and resolving case.yaml
configurations across the codebase.
"""

from __future__ import annotations

from pathlib import Path

from .config_scheme import CaseConfig
from .path_utils import load_yaml
from .validate_config import validate_case_config


def load_case(
    path: Path | str,
    *,
    validate: bool = True,
) -> CaseConfig:
    """Load a case configuration from a file path or directory.

    Accepts either a path to case.yaml or a directory containing case.yaml.
    Uses path_utils.load_yaml, validate_config.validate_case_config, and
    CaseConfig.from_dict as a single entry point for load+validate+CaseConfig.

    Parameters
    ----------
    path : Path | str
        Path to case.yaml file, or directory containing case.yaml.
    validate : bool, default=True
        If True, run validate_case_config before constructing CaseConfig.
        If False, skip validation (not recommended for untrusted input).

    Returns
    -------
    CaseConfig
        Parsed and validated case configuration.

    Raises
    ------
    FileNotFoundError
        If case.yaml is not found at the expected location.
    ValueError
        If validation fails (when validate=True).
    """
    p = Path(path)
    if p.is_file():
        cfg_path = p
    elif p.is_dir():
        cfg_path = p / "case.yaml"
    else:
        cfg_path = p

    if not cfg_path.is_file():
        searched = sorted(set([str(cfg_path), str(p), str(p / "case.yaml")]))
        raise FileNotFoundError(
            f"Expected case.yaml at {cfg_path}. "
            f"Searched: {searched}. "
            "Ensure the case directory contains case.yaml or pass a path to the YAML file."
        )

    data = load_yaml(path=cfg_path)

    if validate:
        errors = validate_case_config(data, cfg_path)
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            raise ValueError(error_msg)

    return CaseConfig.from_dict(data)
