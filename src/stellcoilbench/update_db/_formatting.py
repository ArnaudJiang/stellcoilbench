"""Formatting utilities for leaderboard display."""

from __future__ import annotations

import json
import logging
import re
from importlib import resources
from pathlib import Path
from typing import Any, Dict

from ._metric_definitions import METRIC_DEFINITIONS, METRIC_DETAILED_DEFINITIONS

logger = logging.getLogger(__name__)

_METRIC_DEFS_CACHE: dict[str, Any] | None = None


def _get_metric_definitions_path() -> Path | None:
    """Return path to metric_definitions.json if it exists in repo docs."""
    try:
        from ..path_utils import find_repo_root

        repo_root = find_repo_root(Path(__file__).resolve().parent)
        if repo_root is None:
            return None
        path = repo_root / "docs" / "leaderboard" / "metric_definitions.json"
        return path if path.exists() else None
    except (IndexError, ValueError):
        return None


def _load_metric_definitions() -> dict[str, Any]:
    """Load metric definitions from JSON (repo docs/leaderboard or package fallback)."""
    global _METRIC_DEFS_CACHE
    if _METRIC_DEFS_CACHE is not None:
        return _METRIC_DEFS_CACHE
    path = _get_metric_definitions_path()
    if path is not None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if "reactor_scale_exclude" in data:
                data["reactor_scale_exclude"] = set(data["reactor_scale_exclude"])
            _METRIC_DEFS_CACHE = data
            return data
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.debug("Could not load metric_definitions.json: %s", e)
    # Fallback: load from package-bundled default
    try:
        data = json.loads(
            resources.files("stellcoilbench.update_db")
            .joinpath("metric_definitions_default.json")
            .read_text(encoding="utf-8")
        )
        if "reactor_scale_exclude" in data:
            data["reactor_scale_exclude"] = set(data["reactor_scale_exclude"])
        _METRIC_DEFS_CACHE = data
        return data
    except (OSError, json.JSONDecodeError, TypeError) as e:
        logger.debug("Could not load package metric_definitions_default.json: %s", e)
    _METRIC_DEFS_CACHE = {
        "shorthand": {},
        "short_def": {},
        "detailed_def": {},
        "units": {},
        "surface_display_names": {},
        "reactor_scale_display_order": [],
        "reactor_scale_exclude": set(),
    }
    return _METRIC_DEFS_CACHE


def _get_surface_display_names() -> dict[str, str]:
    """Return surface name -> display name mapping (from config or builtin)."""
    data = _load_metric_definitions()
    return data.get("surface_display_names") or {}


def _get_reactor_scale_display_order() -> list[str]:
    """Return ordered list of reactor-scale metric keys (from config or builtin)."""
    data = _load_metric_definitions()
    return data.get("reactor_scale_display_order") or []


def _get_reactor_scale_exclude() -> set[str]:
    """Return set of reactor-scale keys to exclude from columns (from config or builtin)."""
    data = _load_metric_definitions()
    val = data.get("reactor_scale_exclude")
    return val if isinstance(val, set) else set()


def _metric_shorthand(metric_name: str) -> str:
    """
    Convert metric names to compact shorthand/acronyms for leaderboard display.

    Maps internal metric keys (e.g. ``final_squared_flux``) to compact symbols
    (e.g. ``f_B``) for use in narrow leaderboard columns. Uses LaTeX-style
    notation where appropriate.

    Parameters
    ----------
    metric_name : str
        Internal metric key (e.g. ``final_min_cc_separation``).

    Returns
    -------
    str
        Shorthand string (e.g. ``d_cc``), or metric_name with underscores
        replaced by spaces if no mapping exists.
    """
    data = _load_metric_definitions()
    shorthand = data.get("shorthand") or {}
    return shorthand.get(metric_name, metric_name.replace("_", " "))


_SLASH_DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$")
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})")
_ISO_TWO_DIGIT_YEAR_RE = re.compile(r"^(\d{2})-(\d{1,2})-(\d{1,2})$")


def _format_date(date_str: str | None) -> str:
    """Format a date string to DD/MM/YY for leaderboard display.

    Accepts ISO-8601 (``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:MM:SS``) and
    legacy slash-separated formats (``MM/DD/YY``).  Returns
    ``"_unknown_"`` for *None* or missing values.

    Parameters
    ----------
    date_str : str | None
        Input date string.

    Returns
    -------
    str
        DD/MM/YY formatted string, or ``"_unknown_"``.
    """
    if date_str is None:
        return "_unknown_"
    if not isinstance(date_str, str):
        return date_str  # type: ignore[return-value]
    if not date_str or date_str == "_unknown_":
        return date_str

    date_part = date_str.split("T")[0] if "T" in date_str else date_str

    # ISO: YYYY-MM-DD (canonical path)
    iso_match = _ISO_DATE_RE.match(date_part)
    if iso_match:
        year, month, day = iso_match.groups()
        return f"{day.zfill(2)}/{month.zfill(2)}/{year[2:]}"

    # ISO-like with two-digit year: YY-MM-DD
    two_digit_match = _ISO_TWO_DIGIT_YEAR_RE.match(date_part)
    if two_digit_match:
        year, month, day = two_digit_match.groups()
        return f"{day.zfill(2)}/{month.zfill(2)}/{year}"

    # Legacy slash format: assume MM/DD/YY(YY) and convert to DD/MM/YY
    slash_match = _SLASH_DATE_RE.match(date_part)
    if slash_match:
        first, second, year = slash_match.groups()
        first_int, second_int = int(first), int(second)
        year_short = year[2:] if len(year) == 4 else year
        if first_int > 12:
            day, month = first, second
        elif second_int > 12:
            day, month = second, first
        else:
            day, month = second, first
        return f"{day.zfill(2)}/{month.zfill(2)}/{year_short}"

    logger.debug("Unrecognised date format, returning as-is: %r", date_str)
    return date_str


# Metric keys that should be displayed as integers
_INTEGER_METRICS: set[str] = {"final_linking_number", "coil_order", "num_coils"}


def _format_metric_value(
    value: Any,
    metric_key: str = "",
    compact: bool = False,
) -> str:
    """
    Format a metric value for leaderboard display.

    Parameters
    ----------
    value : Any
        Raw metric value.
    metric_key : str, optional
        Metric key (affects formatting for integer/special metrics).
    compact : bool, default=False
        If True, use ultra-compact scientific notation (for HTML tables).
        If False, use standard .1e format.

    Returns
    -------
    str
        Formatted string for display.
    """
    if metric_key in _INTEGER_METRICS:
        if isinstance(value, (float, int)):
            return str(int(round(value)))
        return str(value)
    if metric_key == "fourier_continuation_orders":
        return str(value) if value else "—"
    if isinstance(value, (float, int)):
        val = float(value)
        if abs(val) < 1e-100:
            return "0"
        if compact:
            s = f"{val:.1e}".replace("e+", "e")
            if s.startswith("0."):
                s = "." + s[2:]
            elif s.startswith("-0."):
                s = "-." + s[3:]
            if "e" in s:
                parts = s.split("e")
                if len(parts) == 2:
                    base, exp = parts[0], parts[1]
                    if exp.startswith("0") and len(exp) > 1:
                        exp = exp[1:]
                    s = base + "e" + exp
            return s
        return f"{val:.1e}"
    return str(value)


def _format_numeric_for_leaderboard(
    value: Any,
    *,
    metric_key: str = "",
    scientific_for_large: float | None = None,
    scientific_for_small: float | None = None,
) -> str:
    """
    Format a numeric value for leaderboard display (reactor-scale, dipole, etc.).

    Parameters
    ----------
    value : Any
        Raw value to format.
    metric_key : str, optional
        Metric key; forces and torques use .1e format.
    scientific_for_large : float, optional
        If abs(v) >= this, use scientific notation (e.g. 1000 for dipole F_max).
    scientific_for_small : float, optional
        If 0 < abs(v) < this, use scientific notation (e.g. 0.01 for dipole).

    Returns
    -------
    str
        Formatted string or "—" for non-numeric.
    """
    if value is None:
        return "—"
    if isinstance(value, (dict, list, str)):
        return "—"
    try:
        v = float(value)
    except (ValueError, TypeError):
        return "—"
    if abs(v) < 1e-100:
        return "0"
    # Forces and torques: always use .1e format
    if "force" in metric_key or "torque" in metric_key:
        return f"{v:.1e}"
    if scientific_for_large is not None and abs(v) >= scientific_for_large:
        return f"{v:.2e}"
    if scientific_for_small is not None and 0 < abs(v) < scientific_for_small:
        return f"{v:.2e}"
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.2e}"


def _metric_display_name(metric_key: str) -> str:
    """
    Convert metric key to human-readable display name.

    Replaces underscores with spaces and title-cases each word.
    Used when no shorthand mapping exists or for documentation headers.

    Parameters
    ----------
    metric_key : str
        Internal metric key (e.g. ``final_squared_flux``).

    Returns
    -------
    str
        Title-cased label (e.g. ``Final Squared Flux``).
    """
    return metric_key.replace("_", " ").title()


def _shorthand_to_math(shorthand: str) -> str:
    r"""
    Convert metric shorthand to RST math mode format.

    Wraps shorthand in ``:math:`...` `` for Sphinx/RST rendering, converting
    underscores to subscripts and Unicode symbols to LaTeX equivalents.

    Parameters
    ----------
    shorthand : str
        Metric shorthand (e.g. ``d_cc``, ``F_max``, ``B̄_n``).

    Returns
    -------
    str
        RST math directive string (e.g. ``:math:`d_{cc}` ``).
    """
    # If it's already a simple variable or Greek letter, wrap it
    if shorthand in ["n", "N", "L", "t"]:
        return f":math:`{shorthand}`"

    # Handle special Unicode characters and new formats
    unicode_map = {
        "κ̄": r":math:`\bar{\kappa}`",
        "F̄": r":math:`\bar{F}`",
        "τ̄": r":math:`\bar{\tau}`",
        "B̄_n": r":math:`\bar{B}_n`",
        "avg(B_n)": r":math:`\text{avg}(B_n)`",
        "max(B_n)": r":math:`\max(B_n)`",
        "Var(l_i)": r":math:`\mathrm{Var}(l_i)`",
        "FC": r":math:`\text{FC}`",  # Fourier continuation
        "F_max": r":math:`F_\text{max}`",
        "τ_max": r":math:`\tau_\text{max}`",
        "κ_max": r":math:`\kappa_\text{max}`",
        "L_SC": r":math:`L_\text{SC}`",
        "w_WP": r":math:`w_\text{WP}`",
        "F_turn": r":math:`F_\text{turn}`",
        "τ_turn": r":math:`\tau_\text{turn}`",
        "ζ_max": r":math:`\zeta_\text{max}`",
    }
    if shorthand in unicode_map:
        return unicode_map[shorthand]

    # Handle function calls like "max(κ)", "max(B_n)" (d_cc, d_cs, F_max, τ_max, κ_max are now direct variables)
    func_match = re.match(r"(\w+)\(([^)]+)\)", shorthand)
    if func_match:
        func_name = func_match.group(1)
        arg = func_match.group(2)
        # Handle special cases
        if arg == "κ":
            arg_math = r"\kappa"
        elif arg == "F":
            arg_math = r"F"
        elif arg == "τ":
            arg_math = r"\tau"
        elif arg == "d_cc":
            arg_math = r"d_{cc}"
        elif arg == "d_cs":
            arg_math = r"d_{cs}"
        elif arg == "B_n":
            arg_math = r"B_n"
        else:
            # Default: convert underscores to subscripts
            parts = arg.split("_")
            if len(parts) == 2:
                arg_math = f"{parts[0]}_{{{parts[1]}}}"
            else:
                # Multiple underscores - convert all to subscripts properly
                result = parts[0]
                for part in parts[1:]:
                    result += f"_{{{part}}}"
                arg_math = result

        # Use LaTeX operators for min/max, \text{} for other functions
        if func_name == "min":
            func_math = "\\min"
        elif func_name == "max":
            func_math = "\\max"
        elif func_name == "avg":
            func_math = "\\text{avg}"
        else:
            func_math = func_name
        # Format the math expression - func_math already contains proper escaping
        return f":math:`{func_math}({arg_math})`"

    # Handle simple variable names with underscores (e.g., "d_cc", "d_cs")
    if "_" in shorthand:
        parts = shorthand.split("_")
        if len(parts) == 2:
            return f":math:`{parts[0]}_{{{parts[1]}}}`"
        else:
            # Multiple underscores - convert all to subscripts
            result = parts[0]
            for part in parts[1:]:
                result += f"_{{{part}}}"
            return f":math:`{result}`"

    # Handle strings with spaces - wrap in \text{} for RST math mode
    if " " in shorthand:
        # Escape spaces by wrapping in \text{}
        escaped = shorthand.replace(" ", r"\ ")
        return f":math:`\\text{{{escaped}}}`"

    # Default: wrap in math mode
    return f":math:`{shorthand}`"


def _shorthand_to_html_math(shorthand: str) -> str:
    """
    Convert metric shorthand to HTML that renders correctly without MathJax.

    Uses Unicode symbols and ``<sub>`` tags for subscripts so tables display
    correctly in raw HTML (e.g. ``d_cc`` → ``d<sub>cc</sub>``).

    Parameters
    ----------
    shorthand : str
        Metric shorthand from :func:`_metric_shorthand` (e.g. ``d_cc``, ``f_B``).

    Returns
    -------
    str
        HTML fragment suitable for embedding in table cells.
    """
    # Plain-text labels (no formatting)
    plain_labels = {"Score", "Date", "User", "i", "f"}
    if shorthand in plain_labels:
        return shorthand

    # Shorthand -> HTML using Unicode and subscripts (no MathJax required)
    html_map = {
        "f_B": "f<sub>B</sub>",
        "B̄_n": "B̄<sub>n</sub>",  # B with macron + subscript
        "κ̄": "κ̄",
        "max(B_n)": "max(B<sub>n</sub>)",
        "Var(l_i)": "Var(l<sub>i</sub>)",
        "d_cc": "d<sub>cc</sub>",
        "d_cs": "d<sub>cs</sub>",
        "MSC": "MSC",
        "F_max": "F<sub>max</sub>",
        "τ_max": "τ<sub>max</sub>",
        "κ_max": "κ<sub>max</sub>",
        "LN": "LN",
        "FC": "FC",
        "avg(QS)": "avg(QS)",
        "LF": "LF",
        "n": "n",
        "N": "N",
        "L": "L",
        "t": "t",
        "L_SC": "L<sub>SC</sub>",
        "w_WP": "w<sub>WP</sub>",
        "F_turn": "F<sub>turn</sub>",
        "τ_turn": "τ<sub>turn</sub>",
        "ζ_max": "ζ<sub>max</sub>",
        # Dipole leaderboard columns
        "L_dip": "L<sub>dip</sub>",
        "I_dip": "I<sub>dip</sub>",
        "κ_dip": "κ<sub>dip</sub>",
        "L_tf": "L<sub>TF</sub>",
        "I_tf": "I<sub>TF</sub>",
        "κ_tf": "κ<sub>TF</sub>",
    }
    return html_map.get(shorthand, shorthand)


# Units for reactor-scale metric columns (LaTeX math fragments)
_RS_UNITS: Dict[str, str] = {
    "reactor_scale_squared_flux": r"\text{T}^2\text{m}^2",
    "reactor_scale_min_cs_separation": r"\text{m}",
    "reactor_scale_min_cc_separation": r"\text{m}",
    "reactor_scale_total_length": r"\text{m}",
    "reactor_scale_max_curvature": r"\text{m}^{-1}",
    "reactor_scale_average_curvature": r"\text{m}^{-1}",
    "reactor_scale_mean_squared_curvature": r"\text{m}^{-2}",
    "reactor_scale_max_max_coil_force": r"\text{MN/m}",
    "reactor_scale_avg_max_coil_force": r"\text{MN/m}",
    "reactor_scale_max_max_coil_torque": r"\text{MN}",
    "reactor_scale_avg_max_coil_torque": r"\text{MN}",
    "per_turn_max_force": r"\text{MN/m}",
    "per_turn_max_torque": r"\text{MN}",
    "total_superconductor_length_km": r"\text{km}",
    "max_winding_pack_width": r"\text{m}",
    "reactor_scale_arclength_variation": r"\text{m}^2",
}


def _metric_definition(metric_name: str) -> str:
    """
    Get detailed mathematical definition for a metric.

    Returns a LaTeX-style string describing the metric (symbol, expression,
    units). Falls back to :func:`_metric_display_name` when no definition exists.

    Parameters
    ----------
    metric_name : str
        Internal metric key (e.g. ``final_squared_flux``).

    Returns
    -------
    str
        LaTeX-formatted definition string, or title-cased metric name if unknown.
    """
    return METRIC_DEFINITIONS.get(metric_name, _metric_display_name(metric_name))


def _metric_detailed_definition(metric_name: str) -> dict | None:
    """
    Get detailed mathematical definition for a metric in structured format.

    Matches the "Available objectives" page format with title, symbol,
    description, math forms, units, and optional notes.

    Parameters
    ----------
    metric_name : str
        Internal metric key (e.g. ``final_normalized_squared_flux``).

    Returns
    -------
    dict | None
        Structured definition with keys: ``title``, ``symbol``, ``description``,
        ``math_forms``, ``units``, optionally ``where`` and ``notes``.
        None if the metric has no detailed definition.
    """
    return METRIC_DETAILED_DEFINITIONS.get(metric_name)
