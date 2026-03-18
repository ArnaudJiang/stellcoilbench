"""Metric definition RST generation for the leaderboard documentation.

Generates the ``leaderboard/metric_definitions.rst`` file containing
notation, mathematical definitions for every tracked metric, the composite
score formula, reactor-scale constraint tables, and the winding-pack model.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Iterable

from ._constraints import REACTOR_SCALE_CONSTRAINTS
from ._formatting import (
    _get_reactor_scale_display_order,
    _metric_definition,
    _metric_detailed_definition,
    _metric_display_name,
    _metric_shorthand,
)

__all__ = [
    "_write_metric_definitions_rst",
]


# ---------------------------------------------------------------------------
# Metric categorisation rules
# ---------------------------------------------------------------------------

_CATEGORY_RULES: list[tuple[str, Any]] = [
    ("field_quality", lambda k: "flux" in k.lower() or "BdotN" in k or "B" in k),
    (
        "coil_geometry",
        lambda k: (
            "curvature" in k.lower()
            or "length" in k.lower()
            or "arclength" in k.lower()
            or "torsion" in k.lower()
            or k in ["coil_order", "num_coils", "fourier_continuation_orders"]
        ),
    ),
    ("separations", lambda k: "separation" in k.lower() or "distance" in k.lower()),
    ("forces_torques", lambda k: "force" in k.lower() or "torque" in k.lower()),
    ("topology", lambda k: "linking" in k.lower()),
    ("performance", lambda k: "time" in k.lower()),
    (
        "particle_confinement",
        lambda k: k in ["loss_fraction", "quasisymmetry_average"],
    ),
]


def _categorise_metrics(
    combined_metric_keys: list[str],
) -> dict[str, list[tuple[str, dict]]]:
    """Sort metric keys into named categories for RST rendering.

    Parameters
    ----------
    combined_metric_keys : list[str]
        All metric keys (surface + reactor-scale).

    Returns
    -------
    dict[str, list[tuple[str, dict]]]
        Mapping from category name to ``(key, detailed_def)`` pairs.
    """
    categories: dict[str, list[tuple[str, dict]]] = {
        "field_quality": [],
        "coil_geometry": [],
        "separations": [],
        "forces_torques": [],
        "topology": [],
        "performance": [],
        "particle_confinement": [],
        "config": [],
    }
    for key in combined_metric_keys:
        detailed_def = _metric_detailed_definition(key)
        if not detailed_def:
            # Fallback for metrics in METRIC_DEFINITIONS only
            if _metric_definition(key) != _metric_display_name(key):
                detailed_def = {
                    "symbol": _metric_shorthand(key),
                    "description": _metric_definition(key),
                    "units": "",
                }
            else:
                continue
        matched = False
        for cat_name, predicate in _CATEGORY_RULES:
            if predicate(key):
                categories[cat_name].append((key, detailed_def))
                matched = True
                break
        if not matched:
            categories["config"].append((key, detailed_def))
    return categories


# ---------------------------------------------------------------------------
# RST building blocks
# ---------------------------------------------------------------------------

_CATEGORY_TITLES: list[tuple[str, str]] = [
    ("field_quality", "Field Quality Metrics"),
    ("coil_geometry", "Coil Geometry Metrics"),
    ("separations", "Separation Metrics"),
    ("forces_torques", "Force and Torque Metrics"),
    ("topology", "Topology Metrics"),
    ("performance", "Performance Metrics"),
    ("particle_confinement", "Particle Confinement Metrics"),
    ("config", "Configuration Metrics"),
]


def _format_metric_def(key: str, def_dict: dict) -> list[str]:
    """Format a single metric definition as a one-line RST bullet.

    Uses plain symbol (no bold) when it contains :math: role, since RST does
    not support nesting interpreted text inside **bold**. Plain symbols use **.

    Parameters
    ----------
    key : str
        Internal metric key name.
    def_dict : dict
        Dictionary with ``symbol``, ``title``, ``description``, ``units``.

    Returns
    -------
    list[str]
        Single formatted RST line (one-line definition).
    """
    symbol = def_dict.get("symbol", _metric_shorthand(key))
    desc = def_dict.get("description", _metric_definition(key))
    units = def_dict.get("units", "")
    # Truncate description to ~120 chars for one-liner
    if len(desc) > 120:
        desc = desc[:117].rsplit(" ", 1)[0] + "..."
    # Do not wrap :math: in ** — RST cannot nest interpreted text in emphasis
    if ":math:" in str(symbol):
        symbol_part = f"{symbol}"
    else:
        symbol_part = f"**{symbol}**"
    if units:
        line = f"- {symbol_part}: {desc} ({units})"
    else:
        line = f"- {symbol_part}: {desc}"
    return [line]


def _build_notation_lines() -> list[str]:
    """Return the opening notation section of the RST file.

    Returns
    -------
    list[str]
        RST lines for the title, introduction, and compact notation.
    """
    return [
        "Metric Definitions",
        "===================",
        "",
        "The following metrics are used to evaluate coil optimization submissions. "
        "Notation: :math:`C_i` coil curve, :math:`S` plasma surface, "
        ":math:`\\mathbf{r}_i` point on coil, :math:`\\kappa_i` curvature, "
        ":math:`N` number of coils, :math:`\\mathbf{B}` magnetic field, "
        ":math:`\\mathbf{n}` surface normal.",
        "",
    ]


def _build_metric_definition_lines(
    all_metric_keys: list[str],
) -> list[str]:
    """Build RST lines for all metric definitions (flat one-line list).

    Parameters
    ----------
    all_metric_keys : list[str]
        Union of all metric keys.

    Returns
    -------
    list[str]
        RST lines for the metric definitions section.
    """
    combined_metric_keys = list(all_metric_keys)
    for rk in _get_reactor_scale_display_order():
        if rk not in combined_metric_keys and _metric_detailed_definition(rk):
            combined_metric_keys.append(rk)

    categories = _categorise_metrics(combined_metric_keys)
    # Flatten all categories into one list, preserving display order
    ordered_keys: list[str] = []
    seen: set[str] = set()
    for cat_key, _ in _CATEGORY_TITLES:
        for key, _ in categories.get(cat_key, []):
            if key not in seen:
                seen.add(key)
                ordered_keys.append(key)

    lines: list[str] = ["Definitions", "-" * len("Definitions"), ""]
    for key in ordered_keys:
        detailed_def = _metric_detailed_definition(key)
        if detailed_def:
            lines.extend(_format_metric_def(key, detailed_def))
    lines.append("")
    return lines


def _load_rst_template(name: str) -> str:
    """Load an RST template from the package.

    Parameters
    ----------
    name : str
        Template filename (e.g. ``composite_score.rst``).

    Returns
    -------
    str
        Template content.
    """
    return (
        resources.files("stellcoilbench.update_db")
        .joinpath("_rst_templates", name)
        .read_text(encoding="utf-8")
    )


def _format_composite_bound(bound: float, units: str) -> str:
    """Format bound with RST units for composite score table.

    Parameters
    ----------
    bound : float
        Numeric bound.
    units : str
        Units string (e.g. ``m``, ``km``, ``m⁻¹``, ``(dimensionless)``).

    Returns
    -------
    str
        Formatted bound string.
    """
    if units == "(dimensionless)" or not units:
        return f"{bound:.2g}" if bound >= 1 or bound == int(bound) else str(bound)
    if "⁻¹" in units or units == "m⁻¹":
        return f"{bound:.1f} m\\ :sup:`-1`"
    # Use int for whole numbers (220, 100) but keep 1.0 as "1.0"
    if bound == 1.0:
        num = "1.0"
    elif bound == int(bound):
        num = str(int(bound))
    else:
        num = f"{bound:.1f}"
    return f"{num} {units}".rstrip()


def _format_margin_bound(bound: float) -> str:
    """Format bound for margin formula (matches composite table display)."""
    if bound == 1.0:
        return "1.0"
    if bound == int(bound):
        return str(int(bound))
    return str(bound)


def _build_soft_constraint_composite_table_rows() -> list[str]:
    """Build RST table row lines for soft constraints in composite score section.

    Returns
    -------
    list[str]
        RST lines for each table row (Constraint, Direction, Bound, Margin).
    """
    lines: list[str] = []
    for c in REACTOR_SCALE_CONSTRAINTS:
        if c.get("hard", False):
            continue
        label = c.get("composite_score_label", c["label"])
        metric = c["metric"]
        bound = c["bound"]
        units = c.get("units", "")
        direction = c["direction"]
        value_rst = c.get("margin_value_rst", r"\text{value}")

        bound_str = _format_composite_bound(bound, units)
        margin_bound = _format_margin_bound(bound)
        if direction == "max":
            direction_rst = r":math:`\leq`"
            margin_rst = rf":math:`1 - {value_rst}\;/\;{margin_bound}`"
        else:
            direction_rst = r":math:`\geq`"
            margin_rst = rf":math:`{value_rst}\;/\;{margin_bound} - 1`"

        constraint_col = f"{label} (``{metric}``)"
        lines.extend(
            [
                f"   * - {constraint_col}",
                f"     - {direction_rst}",
                f"     - {bound_str}",
                f"     - {margin_rst}",
            ]
        )
    return lines


def _build_composite_score_lines() -> list[str]:
    """Build RST lines for the Composite Score section.

    Returns
    -------
    list[str]
        RST lines describing the two-stage scoring algorithm with a worked
        example.
    """
    template = _load_rst_template("composite_score.rst")
    table_rows = _build_soft_constraint_composite_table_rows()
    table_text = "\n".join(table_rows)
    content = template.replace("{{SOFT_CONSTRAINT_TABLE}}", table_text)
    return content.rstrip().split("\n") + [""]


def _build_constraint_table_lines(
    constraints: Iterable[dict[str, Any]],
    *,
    hard_only: bool,
) -> list[dict[str, Any]]:
    """Build row data for hard or soft constraint tables.

    Iterates over reactor-scale constraints and returns formatted row dicts.
    For hard constraints: each row has ``label``, ``bound_str``, ``desc``.
    For soft constraints: each row has ``label``, ``bound_str``, ``direction``,
    ``units``.

    Parameters
    ----------
    constraints : iterable of dict
        Constraint dicts (e.g. REACTOR_SCALE_CONSTRAINTS) with ``label``,
        ``bound``, ``units``, ``direction``, and optional ``hard``.
    hard_only : bool
        If True, yield hard constraints only (with description); else soft only.

    Returns
    -------
    list[dict[str, Any]]
        List of row dicts for RST table formatting.
    """
    rows: list[dict[str, Any]] = []
    for c in constraints:
        if bool(c.get("hard", False)) != hard_only:
            continue
        label = c["label"]
        bound = c["bound"]
        units = c.get("units", "")
        direction = c["direction"]

        if hard_only:
            if direction == "eq":
                bound_str = f"= {bound}"
            elif direction == "max":
                bound_str = f"≤ {bound}"
            elif direction == "min":
                bound_str = f"≥ {bound}"
            else:
                bound_str = str(bound)
            if units and units not in ("(boolean)", "(turns)"):
                bound_str += f" {units}"
            desc = ""
            if "linked to" in label.lower():
                desc = "Every base coil must topologically encircle the plasma."
            elif "linking" in label.lower():
                desc = "Coils must not interlink with one another."
            elif "finite" in label.lower() or "clearance" in label.lower():
                desc = (
                    "Centreline distance :math:`d_{\\text{cc,min}}` must exceed "
                    "the largest winding-pack width :math:`w_{\\text{WP,max}}` "
                    "to prevent physical overlap of finite-build coils."
                )
            rows.append({"label": label, "bound_str": bound_str, "desc": desc})
        else:
            if direction == "max":
                bound_str = f":math:`\\leq {bound}`"
            elif direction == "min":
                bound_str = f":math:`\\geq {bound}`"
            else:
                bound_str = str(bound)
            rows.append(
                {
                    "label": label,
                    "bound_str": bound_str,
                    "direction": direction,
                    "units": units,
                }
            )
    return rows


def _build_hard_constraints_table() -> list[str]:
    """Build RST lines for the hard-constraints table.

    Returns
    -------
    list[str]
        RST lines describing each hard reactor-scale constraint.
    """
    lines = [
        "Reactor-Scale Constraints",
        "-" * len("Reactor-Scale Constraints"),
        "",
        r"Scaled to ARIES-CS (:math:`a = 1.7\,\text{m}`, :math:`B_0 = 5.7\,\text{T}`). "
        "Hard constraints (any violation → score = 0):",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Constraint",
        "     - Bound",
        "     - Description",
    ]
    for row in _build_constraint_table_lines(REACTOR_SCALE_CONSTRAINTS, hard_only=True):
        lines.extend(
            [
                f"   * - {row['label']}",
                f"     - {row['bound_str']}",
                f"     - {row['desc']}",
            ]
        )
    lines.append("")
    return lines


def _build_soft_constraints_table() -> list[str]:
    """Build RST lines for the soft-constraints table.

    Returns
    -------
    list[str]
        RST lines describing each soft reactor-scale constraint.
    """
    lines = [
        "**Soft constraints** — margin factors; violations lower score but do not set to 0:",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Metric",
        "     - Bound",
        "     - Direction",
        "     - Units",
    ]
    for row in _build_constraint_table_lines(
        REACTOR_SCALE_CONSTRAINTS, hard_only=False
    ):
        lines.extend(
            [
                f"   * - {row['label']}",
                f"     - {row['bound_str']}",
                f"     - {row['direction']}",
                f"     - {row['units']}",
            ]
        )
    return lines


def _build_winding_pack_model_lines() -> list[str]:
    """Build RST lines for the winding-pack turn-count model section.

    Returns
    -------
    list[str]
        RST lines for the REBCO Jc model, Stellaris parameters, the
        five-step algorithm, finite-build extent, and per-turn force/torque.
    """
    template = _load_rst_template("winding_pack_model.rst")
    return template.rstrip().split("\n") + [""]


def _build_visualization_legend_lines() -> list[str]:
    """Build RST lines for the visualization link legend.

    Returns
    -------
    list[str]
        Empty; plot columns were removed from the leaderboard.
    """
    return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _write_metric_definitions_rst(
    all_metric_keys: list[str],
    leaderboard_dir: Path,
) -> None:
    """Write the ``metric_definitions.rst`` file inside *leaderboard_dir*.

    Parameters
    ----------
    all_metric_keys : list[str]
        Union of all metric keys seen across surfaces and overall entries.
    leaderboard_dir : Path
        Directory where ``metric_definitions.rst`` will be written.
    """
    metric_def_lines = _build_notation_lines()

    if all_metric_keys:
        metric_def_lines.extend(_build_metric_definition_lines(all_metric_keys))
        metric_def_lines.extend(_build_composite_score_lines())
        metric_def_lines.extend(_build_hard_constraints_table())
        metric_def_lines.extend(_build_soft_constraints_table())
        metric_def_lines.extend(_build_winding_pack_model_lines())
        metric_def_lines.extend(_build_visualization_legend_lines())

    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    (leaderboard_dir / "metric_definitions.rst").write_text("\n".join(metric_def_lines))
