"""Optimizer parameters validation for case configs."""

from __future__ import annotations

from typing import Any

__all__ = ["_validate_optimizer_params"]

_VALID_BACKENDS = {"simsopt", "focus"}
_VALID_FOCUS_PARSERS = {
    "auto",
    "focus_fourier",
    "focus_filament",
}


def _validate_focus_params(
    focus_params: Any,
    pfx: str,
    *,
    require_executable: bool,
) -> list[str]:
    """Validate the optional ``focus_params`` section."""
    errors: list[str] = []
    if not isinstance(focus_params, dict):
        return [f"{pfx}focus_params must be a dictionary when backend is 'focus'"]

    valid_keys = {
        "executable",
        "arguments",
        "case_postproc",
        "ccsep_alpha",
        "ccsep_beta",
        "curv_k0",
        "cssep_factor",
        "df_maxiter",
        "exit_tol",
        "init_current",
        "init_radius",
        "input_files",
        "is_symmetric",
        "is_vary_current",
        "nseg",
        "nteta",
        "nzeta",
        "output_harmonics_file",
        "output_filaments_file",
        "output_h5_file",
        "parser",
        "run_stem",
        "save_freq",
        "timeout_seconds",
        "skip_run",
        "nfp",
        "stellsym",
        "numquadpoints",
        "target_length",
        "weight_bnorm",
        "weight_ccsep",
        "weight_cssep",
        "weight_curv",
        "weight_ttlen",
    }
    for key in focus_params:
        if key not in valid_keys:
            errors.append(
                f"{pfx}Unknown focus_params key: '{key}'. "
                f"Valid keys: {sorted(valid_keys)}."
            )

    skip_run = bool(focus_params.get("skip_run", False))
    if require_executable and not skip_run and not focus_params.get("executable"):
        errors.append(
            f"{pfx}focus_params.executable is required when optimizer_params.backend is 'focus'"
        )

    parser = focus_params.get("parser", "auto")
    if not isinstance(parser, str) or parser not in _VALID_FOCUS_PARSERS:
        errors.append(
            f"{pfx}focus_params.parser must be one of {sorted(_VALID_FOCUS_PARSERS)}"
        )

    if "arguments" in focus_params and not (
        isinstance(focus_params["arguments"], list)
        and all(isinstance(v, str) for v in focus_params["arguments"])
    ):
        errors.append(f"{pfx}focus_params.arguments must be a list of strings")

    if "input_files" in focus_params and not (
        isinstance(focus_params["input_files"], list)
        and all(isinstance(v, str) for v in focus_params["input_files"])
    ):
        errors.append(f"{pfx}focus_params.input_files must be a list of strings")

    if "timeout_seconds" in focus_params:
        val = focus_params["timeout_seconds"]
        if not isinstance(val, int) or val < 1:
            errors.append(f"{pfx}focus_params.timeout_seconds must be a positive integer")

    for key in ("nfp", "numquadpoints"):
        if key in focus_params:
            val = focus_params[key]
            if not isinstance(val, int) or val < 1:
                errors.append(f"{pfx}focus_params.{key} must be a positive integer")

    if "stellsym" in focus_params and not isinstance(focus_params["stellsym"], bool):
        errors.append(f"{pfx}focus_params.stellsym must be a boolean")

    return errors


def _validate_optimizer_params(
    optimizer_params: Any,
    pfx: str,
    focus_params: Any | None = None,
) -> list[str]:
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

    backend = optimizer_params.get("backend", "simsopt")
    if not isinstance(backend, str) or backend not in _VALID_BACKENDS:
        errors.append(
            f"{pfx}optimizer_params.backend must be one of {sorted(_VALID_BACKENDS)}"
        )
    if backend == "focus":
        errors.extend(
            _validate_focus_params(
                focus_params,
                pfx,
                require_executable=True,
            )
        )

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
