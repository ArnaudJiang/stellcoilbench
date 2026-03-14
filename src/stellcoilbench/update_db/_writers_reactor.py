"""Reactor-scale leaderboard RST writer.

Generates ``leaderboard/reactor_scale.rst`` with per-surface tables showing
engineering metrics scaled to the ARIES-CS reference reactor. Violated
constraints are highlighted (red = hard, orange = soft).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from ._formatting import (
    _metric_shorthand,
    _shorthand_to_html_math,
)
from ._writers_common import (
    _build_reactor_scale_row_cells,
    _get_constraint_violation_sets,
    _write_surface_leaderboard_rst,
    _resolve_repo_root,
)
from ._formatting import (
    _get_reactor_scale_display_order,
    _get_reactor_scale_exclude,
)

logger = logging.getLogger(__name__)

__all__ = [
    "write_reactor_scale_leaderboard",
]


def write_reactor_scale_leaderboard(
    leaderboard: Dict[str, Any],
    surface_leaderboards: Dict[str, Dict[str, Any]],
    out_rst: Path,
    repo_root: Path | None = None,
    *,
    use_local_links: bool = False,
) -> None:
    """Write a reactor-scale leaderboard RST file with per-surface tables.

    Shows reactor-scale engineering metrics (MN/m forces, curvatures in 1/m)
    alongside composite score and constraint status. Violated constraints
    are highlighted (red=hard, orange=soft).

    Parameters
    ----------
    leaderboard : dict
        Leaderboard (unused; surface_leaderboards carries entries).
    surface_leaderboards : dict[str, dict]
        Per-surface data; includes excluded entries for diagnostics.
    out_rst : Path
        Output path for reactor_scale.rst.
    repo_root : Path, optional
        Repository root for visualization links; inferred from out_rst if None.
    """

    def _get_rs_keys(entries: list[Dict[str, Any]]) -> list[str]:
        """Collect reactor-scale metric keys present in entries, in display order.

        Keys may live in either ``reactor_scale_metrics`` or device-scale
        ``metrics`` (e.g. ``avg_BdotN_over_B`` is dimensionless and stored
        at device scale only).
        """
        exclude = _get_reactor_scale_exclude()
        display_order = _get_reactor_scale_display_order()
        available: set[str] = set()
        for e in entries:
            rs = e.get("reactor_scale_metrics") or {}
            ms = e.get("metrics") or {}
            for k in rs:
                if k not in exclude:
                    available.add(k)
            for k in display_order:
                if k in ms:
                    available.add(k)
        ordered = [k for k in display_order if k in available]
        for k in sorted(available - set(ordered)):
            ordered.append(k)
        return ordered

    resolved_repo_root = _resolve_repo_root(out_rst, repo_root)

    def _reactor_row_builder(
        entries: list[Dict[str, Any]], surface_name: str, surf_data: Dict[str, Any]
    ):
        rs_keys = _get_rs_keys(entries)
        if not rs_keys:
            return None
        header_labels = [
            _shorthand_to_html_math("Score"),
            _shorthand_to_html_math("N"),
            _shorthand_to_html_math("n"),
        ]
        for k in rs_keys:
            shorthand = _metric_shorthand(k)
            header_labels.append(_shorthand_to_html_math(shorthand))
        header_labels.extend(
            [
                _shorthand_to_html_math("LN"),
                r'<span class="math notranslate nohighlight">\(\max_i N_{\text{turns}}\)</span>',
                _shorthand_to_html_math("User"),
                _shorthand_to_html_math("i"),
                _shorthand_to_html_math("f"),
            ]
        )
        all_rows: list[list[tuple[str, str]]] = []
        all_row_classes: list[str] = []
        for entry in entries:
            hard_metric_set, soft_metric_set = _get_constraint_violation_sets(entry)
            row_parts = _build_reactor_scale_row_cells(
                entry,
                rs_keys,
                hard_metric_set,
                soft_metric_set,
                resolved_repo_root,
                use_local_links=use_local_links,
            )
            all_rows.append(list(row_parts))
            all_row_classes.append("")
        return (header_labels, all_rows, all_row_classes)

    header_lines: list[str] = [
        "Reactor-Scale Leaderboard",
        "=========================",
        "",
        ".. role:: red",
        ".. role:: orange",
        "",
        ".. raw:: html",
        "",
        "   <style>",
        "   .red { color: #dc3545; font-weight: bold; }",
        "   .orange { color: #e67e22; font-weight: bold; }",
        "   </style>",
        "",
        "All values are scaled to the **ARIES-CS reference** "
        "(minor radius :math:`a = 1.7` m, on-axis field :math:`B_0 = 5.7` T).",
        "",
        "Entries are ranked by **composite score** (higher = better engineering margin). "
        "See :doc:`metric_definitions` for constraint bounds and the scoring formula.",
        "",
    ]

    _write_surface_leaderboard_rst(
        surface_leaderboards,
        _reactor_row_builder,
        header_lines,
        out_rst,
        "rs",
        empty_entries_message="No submissions for this surface.",
        empty_no_data_message="No reactor-scale data available for this surface.",
    )
