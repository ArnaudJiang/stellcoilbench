"""Surface filename extraction from case config (no path_utils dependencies)."""

from __future__ import annotations

from typing import Any


def get_surface_filename(case_data: dict | Any) -> str:
    """Extract surface filename from case config (dict or CaseConfig).

    Parameters
    ----------
    case_data : dict | CaseConfig
        Case configuration (parsed YAML dict or CaseConfig instance).

    Returns
    -------
    str
        Surface filename (e.g. ``input.LandremanPaul2021_QA``), or empty string.
    """
    if isinstance(case_data, dict):
        sp = case_data.get("surface_params", {}) or {}
    else:
        sp = getattr(case_data, "surface_params", {}) or {}
    if isinstance(sp, dict):
        return sp.get("surface", "") or ""
    return getattr(sp, "surface", "") or ""
