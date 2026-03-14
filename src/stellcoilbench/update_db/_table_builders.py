"""Table builders for leaderboard HTML/RST tables.

Provides CSS styling, sortable table rendering, and row-cell builders
used by surface-specific, reactor-scale, and dipole leaderboard writers.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Sequence

from ..constants import CONSTRAINT_VIOLATIONS_KEY

from ._formatting import (
    _format_date,
    _format_metric_value,
    _format_numeric_for_leaderboard,
)
from ._viz_links import resolve_visualization_links

__all__ = [
    "_build_leaderboard_row_cells",
    "_build_reactor_scale_row_cells",
    "_format_metric_html",
    "_get_constraint_violation_sets",
    "_leaderboard_table_styles",
    "_leaderboard_table_wrapper_open",
    "_sortable_table_script",
    "_user_cell",
    "_viz_link_cells",
    "render_html_table",
]


def _leaderboard_table_styles(indent: str = "") -> list[str]:
    """Return CSS lines for leaderboard table wrapper (sortable hover).

    Parameters
    ----------
    indent : str, default ""
        Prefix for each line (e.g. ``"   "`` for RST raw HTML blocks).

    Returns
    -------
    list[str]
        Style block for table wrapper.
    """
    return [
        f"{indent}<style>.leaderboard-table-wrapper .sortable:hover {{ background: #f0f0f0; }}</style>",
    ]


def _leaderboard_table_wrapper_open(wrapper_id: str = "", indent: str = "") -> str:
    """Return opening div line for scrollable leaderboard table wrapper.

    Parameters
    ----------
    wrapper_id : str, default ""
        Optional id attribute (e.g. ``leaderboard-wrapper-{safe_id}``).
    indent : str, default ""
        Prefix for the line.

    Returns
    -------
    str
        Single line: ``<div class="leaderboard-table-wrapper" ...>``.
    """
    id_attr = f' id="{wrapper_id}"' if wrapper_id else ""
    return f'{indent}<div{id_attr} class="leaderboard-table-wrapper" style="max-height: 420px; overflow-y: auto; margin-bottom: 1em;">'


def _sortable_table_script(table_id: str) -> list[str]:
    """Return inline JavaScript for sortable table. Used by Markdown leaderboards.

    Parameters
    ----------
    table_id : str
        HTML ``id`` of the table element to make sortable.

    Returns
    -------
    list[str]
        Lines of JavaScript to wrap in ``<script>...</script>``.
    """
    return [
        "(function() {",
        "  var table = document.getElementById('" + table_id + "');",
        "  if (!table) return;",
        "  var headers = table.querySelectorAll('th.sortable');",
        "  var sortDir = {};",
        "  headers.forEach(function(th, i) { sortDir[i] = 1; });",
        "  function sortTable(col) {",
        "    var tbody = table.querySelector('tbody');",
        "    var rows = Array.from(tbody.querySelectorAll('tr'));",
        "    sortDir[col] = sortDir[col] || 1;",
        "    var mult = sortDir[col];",
        "    sortDir[col] = -sortDir[col];",
        "    rows.sort(function(a, b) {",
        "      var ac = a.children[col]; var bc = b.children[col];",
        "      var av = ac && ac.getAttribute('data-sort-value');",
        "      var bv = bc && bc.getAttribute('data-sort-value');",
        "      var an = parseFloat(av); var bn = parseFloat(bv);",
        "      if (!isNaN(an) && !isNaN(bn)) return mult * (an - bn);",
        "      return mult * String(av || '').localeCompare(String(bv || ''));",
        "    });",
        "    rows.forEach(function(r) { tbody.appendChild(r); });",
        "  }",
        "  headers.forEach(function(th, i) {",
        "    th.addEventListener('click', function() { sortTable(i); });",
        "  });",
        "})();",
    ]


def _get_constraint_violation_sets(entry: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Extract hard and soft constraint violation metric sets from an entry.

    Parameters
    ----------
    entry : dict
        Leaderboard entry with optional ``constraint_violations`` list.
        Each violation has ``metric`` and ``hard`` keys.

    Returns
    -------
    tuple[set[str], set[str]]
        (hard_metric_set, soft_metric_set) for use with :func:`_format_metric_html`.
    """
    violations = entry.get(CONSTRAINT_VIOLATIONS_KEY, [])
    hard_violated = [v for v in violations if v.get("hard")]
    soft_violated = [v for v in violations if not v.get("hard")]
    hard_metric_set = {v["metric"] for v in hard_violated}
    soft_metric_set = {v["metric"] for v in soft_violated}
    return (hard_metric_set, soft_metric_set)


def _format_metric_html(
    val_str: str,
    metric: str,
    hard_set: set[str],
    soft_set: set[str],
) -> str:
    """Format metric value for HTML with constraint violation highlighting.

    Used in reactor-scale leaderboard tables to visually distinguish
    hard vs soft constraint violations (red vs orange spans).

    Parameters
    ----------
    val_str : str
        Display string for the metric value (e.g. "1.2e-5").
    metric : str
        Internal metric key (e.g. ``"reactor_scale_min_cc_separation"``).
    hard_set : set[str]
        Set of metric keys with hard constraint violations.
    soft_set : set[str]
        Set of metric keys with soft constraint violations.

    Returns
    -------
    str
        HTML string: ``<span class="red">...</span>`` for hard violations,
        ``<span class="orange">...</span>`` for soft, or escaped ``val_str``
        when the metric is not violated.
    """
    if metric in hard_set:
        return f'<span class="red">{html.escape(val_str)}</span>'
    if metric in soft_set:
        return f'<span class="orange">{html.escape(val_str)}</span>'
    return html.escape(val_str)


def render_html_table(
    table_id: str,
    header_labels: Sequence[str],
    rows: Sequence[Sequence[tuple[str, str]]],
    *,
    row_classes: Sequence[str] | None = None,
    indent: str = "   ",
    font_size: str = "0.85em",
    rst_directive: bool = True,
) -> list[str]:
    """Render a sortable HTML table used by all leaderboard writers.

    Encapsulates the common pattern: style block, wrapper div, ``<table>``,
    ``<thead>`` with sortable headers, ``<tbody>`` with data-sort rows,
    and closing tags.

    Parameters
    ----------
    table_id : str
        HTML ``id`` attribute for the ``<table>`` element.
    header_labels : sequence[str]
        Column header labels (may contain HTML/math).
    rows : sequence[sequence[tuple[str, str]]]
        Each row is a sequence of ``(display_html, sort_value)`` pairs.
    row_classes : sequence[str] or None, default None
        Per-row CSS classes (e.g. ``"highlighted"``).  Must match
        ``len(rows)`` if provided; empty string for no extra class.
    indent : str, default ``"   "``
        Whitespace prefix for every line (RST raw blocks need 3 spaces).
    font_size : str, default ``"0.85em"``
        CSS ``font-size`` on the ``<table>`` element.
    rst_directive : bool, default True
        If True, prepend ``.. raw:: html`` directive (for RST files).
        Set to False for raw Markdown files.

    Returns
    -------
    list[str]
        Complete HTML lines ready to append to a line list.
    """
    lines: list[str] = []
    if rst_directive:
        lines.append(".. raw:: html")
        lines.append("")
    lines.extend(_leaderboard_table_styles(indent))
    lines.append(_leaderboard_table_wrapper_open(indent=indent))
    lines.append(
        f'{indent}<table id="{table_id}" class="leaderboard-sortable" style="font-size: {font_size};">'
    )
    lines.append(f"{indent}<thead>")
    lines.append(f"{indent}<tr>")
    for ci, label in enumerate(header_labels):
        lines.append(
            f'{indent}<th class="sortable" data-col="{ci}" style="font-size: 0.9em; padding: 4px 8px; cursor: pointer; user-select: none;" title="Click to sort">'
            f'{label} <span class="sort-icon">↕</span></th>'
        )
    lines.append(f"{indent}</tr>")
    lines.append(f"{indent}</thead>")
    lines.append(f"{indent}<tbody>")

    for idx, row in enumerate(rows):
        cls = ""
        if row_classes and row_classes[idx]:
            cls = f' class="{row_classes[idx]}"'
        lines.append(f"{indent}<tr{cls}>")
        for disp, sort_val in row:
            sv = str(sort_val).replace('"', "&quot;").replace("<", "&lt;")
            if "<" not in disp:
                disp = html.escape(disp)
            lines.append(
                f'{indent}<td style="font-size: 0.9em; padding: 4px 8px;" data-sort-value="{sv}">{disp}</td>'
            )
        lines.append(f"{indent}</tr>")

    lines.append(f"{indent}</tbody>")
    lines.append(f"{indent}</table>")
    lines.append(f"{indent}</div>")
    return lines


def _score_cell(entry: dict[str, Any]) -> tuple[str, str]:
    """Format the composite score as a ``(display, sort_value)`` pair.

    Parameters
    ----------
    entry : dict
        Leaderboard entry with optional ``composite_score`` key.

    Returns
    -------
    tuple[str, str]
    """
    cs = entry.get("composite_score")
    score_str = f"{cs:.3f}" if cs is not None else "—"
    score_sort = float(cs) if cs is not None else -1e9
    return (score_str, str(score_sort))


def _int_metric_cell(
    metrics: dict[str, Any],
    key: str,
) -> tuple[str, str]:
    """Format an integer metric as ``(display, sort_value)``.

    Parameters
    ----------
    metrics : dict
        Metrics dict.
    key : str
        Metric key to look up.

    Returns
    -------
    tuple[str, str]
    """
    val = metrics.get(key)
    s = str(int(round(float(val)))) if val is not None else "—"
    return (s, s if s != "—" else "")


def _user_cell(entry: dict[str, Any]) -> tuple[str, str]:
    """Format the user/contact column as a ``(display, sort_value)`` pair.

    Parameters
    ----------
    entry : dict
        Leaderboard entry with optional ``contact`` key.

    Returns
    -------
    tuple[str, str]
    """
    user_val = entry.get("contact", "?")[:15]
    return (user_val, user_val)


def _viz_link_cells(
    entry: dict[str, Any],
    repo_root: Path,
    *,
    use_local_links: bool = False,
) -> list[tuple[str, str]]:
    """Build visualization link cells (initial and final) for a row.

    Parameters
    ----------
    entry : dict
        Leaderboard entry with ``rank`` for link resolution.
    repo_root : Path
        Repository root for resolving visualization links.
    use_local_links : bool, default False
        If True, use file: URLs for local development.

    Returns
    -------
    list[tuple[str, str]]
        Two cells: ``[(i_html, i_sort), (f_html, f_sort)]``.
    """
    i_link_html, i_link_sort, f_link_html, f_link_sort = resolve_visualization_links(
        entry, repo_root, use_local_links=use_local_links
    )
    return [(i_link_html, i_link_sort), (f_link_html, f_link_sort)]


def _build_reactor_scale_row_cells(
    entry: dict[str, Any],
    rs_keys: list[str],
    hard_metric_set: set[str],
    soft_metric_set: set[str],
    repo_root: Path,
    *,
    use_local_links: bool = False,
) -> list[tuple[str, str]]:
    """Build (display, sort_value) cells for a reactor-scale leaderboard row.

    Column order: Score, N, n, rs_keys, LN, N_turns, User, viz_i, viz_f.
    Applies constraint violation highlighting via :func:`_format_metric_html`.

    Parameters
    ----------
    entry : dict
        Leaderboard entry with metrics, reactor_scale_metrics, composite_score.
    rs_keys : list[str]
        Reactor-scale metric keys to include (e.g. avg_BdotN_over_B).
    hard_metric_set, soft_metric_set : set[str]
        From :func:`_get_constraint_violation_sets`.
    repo_root : Path
        For resolving visualization links.

    Returns
    -------
    list[tuple[str, str]]
        List of (display, sort_value) for each column.
    """
    rs = entry.get("reactor_scale_metrics") or {}
    metrics = entry.get("metrics") or {}

    row_parts: list[tuple[str, str]] = []
    row_parts.append(_score_cell(entry))
    row_parts.append(_int_metric_cell(metrics, "num_coils"))
    row_parts.append(_int_metric_cell(metrics, "coil_order"))

    for k in rs_keys:
        raw_val = rs.get(k)
        if raw_val is None:
            raw_val = metrics.get(k)
        val_str = _format_numeric_for_leaderboard(raw_val)
        disp = _format_metric_html(val_str, k, hard_metric_set, soft_metric_set)
        sort_val = (
            raw_val
            if isinstance(raw_val, (int, float))
            else (val_str if val_str != "—" else "")
        )
        row_parts.append((disp, str(sort_val) if sort_val != "" else ""))

    ln_val = metrics.get("final_linking_number")
    ln_str = str(int(round(float(ln_val)))) if ln_val is not None else "—"
    ln_disp = _format_metric_html(
        ln_str, "final_linking_number", hard_metric_set, soft_metric_set
    )
    ln_sort = str(int(round(float(ln_val)))) if ln_val is not None else ""
    row_parts.append((ln_disp, ln_sort))

    n_turns = rs.get("N_turns_per_coil")
    if isinstance(n_turns, list) and n_turns:
        n_turns_str = str(max(n_turns))
        n_turns_sort = str(max(n_turns))
    else:
        n_turns_str = "—"
        n_turns_sort = ""
    n_turns_disp = _format_metric_html(
        n_turns_str, "N_turns_per_coil", hard_metric_set, soft_metric_set
    )
    row_parts.append((n_turns_disp, n_turns_sort))

    row_parts.append(_user_cell(entry))
    row_parts.extend(_viz_link_cells(entry, repo_root, use_local_links=use_local_links))

    return row_parts


def _build_leaderboard_row_cells(
    entry: dict[str, Any],
    metric_keys: list[str],
    columns: list[str],
    *,
    with_sort_values: bool = True,
    compact_metrics: bool = False,
    repo_root: Path | None = None,
    use_local_links: bool = False,
) -> list[tuple[str, str | float]]:
    """Build (display, sort_value) for each column in a leaderboard row.

    Shared row builder for Markdown, RST, and per-surface leaderboard tables.
    Supports configurable column order via the ``columns`` list. When
    ``"metrics"`` is in columns, iterates over ``metric_keys`` and formats
    each using :func:`_format_metric_value`.

    Parameters
    ----------
    entry : dict
        Leaderboard entry with keys: ``metrics``, ``composite_score``,
        ``contact``, ``run_date``, ``rank``.
    metric_keys : list[str]
        Metric keys to include when ``"metrics"`` appears in columns.
    columns : list[str]
        Column order. Valid values: ``"rank"``, ``"score"``, ``"user"``,
        ``"date"``, ``"metrics"``, ``"viz_i"``, ``"viz_f"``.
    with_sort_values : bool, default True
        If True, each tuple is (display, sort_value) for data-sort-value
        attributes. If False, sort_value equals display (for simple tables).
    compact_metrics : bool, default False
        Pass ``compact=True`` to :func:`_format_metric_value` for metric cells.
    repo_root : Path | None
        Repository root for resolving visualization links. Required when
        ``"viz_i"`` or ``"viz_f"`` is in columns.

    Returns
    -------
    list[tuple[str, str | float]]
        List of (display_str, sort_value) for each column. sort_value is
        numeric for score/metrics, string for dates/users, etc.
    """
    metrics = entry.get("metrics", {})
    run_date = _format_date(entry.get("run_date", "_unknown_"))
    run_date_raw = entry.get("run_date", "") or "0000-00-00"
    cs = entry.get("composite_score")
    score_str = f"{cs:.3f}" if cs is not None else "—"
    score_sort = float(cs) if cs is not None else -1e9
    user_val = entry.get("contact", "?")[:15]
    rank_val = entry.get("rank", 0)

    def _metric_cell(key: str) -> tuple[str, str | float]:
        value = metrics.get(key)
        disp = (
            _format_metric_value(value, metric_key=key, compact=compact_metrics)
            if value is not None
            else "—"
        )
        sort_val: str | float = (
            float(value)
            if isinstance(value, (int, float))
            else ("" if value is None else str(value))
        )
        return (disp, sort_val if with_sort_values else disp)
        return (disp, sort_val if with_sort_values else disp)

    def _cell(disp: str, sort_val: str | float) -> tuple[str, str | float]:
        return (disp, sort_val if with_sort_values else disp)

    viz_links = None
    if "viz_i" in columns or "viz_f" in columns:
        i_html, i_sort, f_html, f_sort = resolve_visualization_links(
            entry, repo_root or Path("."), use_local_links=use_local_links
        )
        if f_html == "—" and f_sort == "":
            f_html, f_sort = str(entry.get("rank", "-")), str(entry.get("rank", "-"))
        viz_links = (i_html, i_sort, f_html, f_sort)

    cells: list[tuple[str, str | float]] = []
    for col in columns:
        if col == "rank":
            cells.append(
                _cell(
                    str(rank_val), rank_val if isinstance(rank_val, (int, float)) else 0
                )
            )
        elif col == "score":
            cells.append(_cell(score_str, score_sort))
        elif col == "user":
            cells.append(_cell(user_val, user_val))
        elif col == "date":
            cells.append(_cell(run_date, run_date_raw))
        elif col == "metrics":
            for key in metric_keys:
                cells.append(_metric_cell(key))
        elif col == "viz_i" and viz_links:
            cells.append(_cell(viz_links[0], viz_links[1]))
        elif col == "viz_f" and viz_links:
            cells.append(_cell(viz_links[2], viz_links[3]))
    return cells
