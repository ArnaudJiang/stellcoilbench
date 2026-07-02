"""Surface parameters and surface-existence validation for case configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["_validate_surface_params", "_validate_surface_exists"]


def _validate_surface_params(surface_params: Any, pfx: str) -> list[str]:
    """Validate the ``surface_params`` section of a case config.

    Parameters
    ----------
    surface_params : Any
        Value of ``data["surface_params"]``.
    pfx : str
        Error-message prefix (file path).

    Returns
    -------
    list[str]
        Error messages.
    """
    errors: list[str] = []
    if not isinstance(surface_params, dict):
        return [f"{pfx}surface_params must be a dictionary"]

    valid_keys = {"surface", "range", "virtual_casing"}
    for key in surface_params:
        if key not in valid_keys:
            errors.append(
                f"{pfx}Unknown surface_params key: '{key}'. "
                f"Valid keys: {sorted(valid_keys)}"
            )

    if "surface" not in surface_params:
        errors.append(f"{pfx}surface_params must contain 'surface' field")
    if "range" in surface_params:
        valid_ranges = ["half period", "full torus"]
        if surface_params["range"] not in valid_ranges:
            errors.append(f"{pfx}surface_params.range must be one of {valid_ranges}")
    if "virtual_casing" in surface_params:
        if not isinstance(surface_params["virtual_casing"], bool):
            errors.append(
                f"{pfx}surface_params.virtual_casing must be a boolean (true/false)"
            )
    return errors


def _validate_surface_exists(
    surface_params: dict[str, Any],
    pfx: str,
    surfaces_dir: Path | None = None,
) -> list[str]:
    """Check that the referenced surface file exists in plasma_surfaces/.

    Parameters
    ----------
    surface_params : dict
        Value of ``data["surface_params"]``.
    pfx : str
        Error-message prefix (file path).
    surfaces_dir : Path, optional
        Directory containing surface files. If None, uses repo-relative
        ``plasma_surfaces/``.

    Returns
    -------
    list[str]
        Error messages. Empty list if surface exists or check is skipped.
    """
    errors: list[str] = []
    if not isinstance(surface_params, dict):
        return errors
    surface = surface_params.get("surface")
    if not surface:
        return errors
    if surfaces_dir is None:
        # Package is at src/stellcoilbench/validate_config/_surface.py -> parents[3] = repo root
        surfaces_dir = Path(__file__).resolve().parents[3] / "plasma_surfaces"
    surface_path = Path(str(surface))
    candidates = [
        surface_path if surface_path.is_absolute() else surfaces_dir / surface_path,
    ]
    if not surface_path.is_absolute() and surface_path.parts[:1] == ("plasma_surfaces",):
        candidates.append(surfaces_dir.parent / surface_path)
    if surfaces_dir.is_dir() and not any(candidate.exists() for candidate in candidates):
        available = sorted(f.name for f in surfaces_dir.iterdir() if f.is_file())
        errors.append(
            f'{pfx}surface_params.surface "{surface}" not found in '
            f"plasma_surfaces/. Available: {available}. "
            "Check plasma_surfaces/ for available surfaces."
        )
    return errors
