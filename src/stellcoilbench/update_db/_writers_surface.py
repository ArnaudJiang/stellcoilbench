"""Surface-specific leaderboard writers (RST and Markdown).

Contains :func:`write_rst_leaderboard` (which writes the main toctree RST,
delegates metric definitions to :mod:`_writers_metric_defs`, and generates
the ``surface_specific.rst`` tables), :func:`write_surface_leaderboards`
(per-surface Markdown files), and :func:`write_markdown_leaderboard`
(overall Markdown leaderboard).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from ._formatting import (
    _metric_definition,
    _metric_display_name,
    _metric_shorthand,
    _shorthand_to_html_math,
)
from .submission_io import (
    _DEVICE_LEADERBOARD_ALWAYS_INCLUDE,
    _ensure_flux_first,
    _get_all_metrics_from_entries,
    _get_ordered_metrics_for_entries,
)
from ..path_utils import normalize_surface_id
from ._writers_common import (
    _build_leaderboard_row_cells,
    _leaderboard_footer_md,
    _leaderboard_footer_rst,
    _sortable_table_script,
    _surface_display_name,
    _write_surface_leaderboard_rst,
    render_html_table,
)
from ._writers_metric_defs import _write_metric_definitions_rst

logger = logging.getLogger(__name__)

__all__ = [
    "write_markdown_leaderboard",
    "write_rst_leaderboard",
    "write_surface_leaderboards",
]


def write_markdown_leaderboard(leaderboard: Dict[str, Any], out_md: Path) -> None:
    """Write a markdown leaderboard table to out_md.

    Parameters
    ----------
    leaderboard : dict
        Leaderboard with ``entries`` list (each entry has rank, metrics, etc.).
    out_md : Path
        Output path for the Markdown file.
    """
    entries = leaderboard.get("entries") or []

    lines = [
        "# CoilBench Leaderboard",
        "",
        "Welcome to the CoilBench leaderboard! Compare coil optimization methods across different plasma surfaces.",
        "",
        "---",
        "",
    ]

    nav_lines = ["- [Plasma surface leaderboards](leaderboards/)"]
    lines.append("## Quick Navigation")
    lines.extend(nav_lines)
    lines.append("")

    lines.append("## Overall Leaderboard")
    lines.append("")

    if not entries:
        lines.append("_No valid submissions found._")
        lines.append("")
        lines.append(
            "To add submissions, place `results.json` files in the `submissions/` directory following the format:"
        )
        lines.append("```json")
        lines.append("{")
        lines.append('  "metadata": {')
        lines.append('    "contact": "your_username",')
        lines.append('    "method_version": "v1.0.0",')
        lines.append('    "hardware": "your_hardware"')
        lines.append("  },")
        lines.append('  "metrics": {...}')
        lines.append("}")
        lines.append("```")
    else:
        all_metric_keys = _get_all_metrics_from_entries(entries)

        header_labels = ["#", "Score", "User", "Date"]
        header_labels.extend([_metric_shorthand(key) for key in all_metric_keys])

        rows = []
        for e in entries:
            row_cells = _build_leaderboard_row_cells(
                e,
                all_metric_keys,
                ["rank", "score", "user", "date", "metrics"],
                with_sort_values=False,
                compact_metrics=True,
            )
            rows.append(row_cells)

        table_lines = render_html_table(
            "overall",
            header_labels,
            rows,
            rst_directive=False,
            indent="",
        )
        lines.extend(table_lines)

        lines.append("")
        lines.append("### Legend")
        lines.append("")

        legend_items = []
        for key in all_metric_keys:
            shorthand = _metric_shorthand(key)
            full_name = _metric_display_name(key)
            legend_items.append(f"- **{shorthand}**: {full_name}")

        lines.extend(legend_items)
        lines.append("")

    lines.extend(_leaderboard_footer_md())

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def write_rst_leaderboard(
    leaderboard: Dict[str, Any],
    out_rst: Path,
    surface_leaderboards: Dict[str, Dict[str, Any]],
) -> None:
    """Write a ReadTheDocs-friendly RST leaderboard with metric definitions
    and per-surface tables.

    Writes leaderboard.rst, leaderboard/metric_definitions.rst, and
    leaderboard/surface_specific.rst.

    Parameters
    ----------
    leaderboard : dict
        Leaderboard with ``entries`` list.
    out_rst : Path
        Path for main leaderboard.rst file.
    surface_leaderboards : dict[str, dict]
        Per-surface data from :func:`build_surface_leaderboards`.
    """
    entries = leaderboard.get("entries") or []
    surface_names = sorted(surface_leaderboards.keys())

    # Collect all unique metrics across all surfaces for definitions
    all_metric_keys_set = set()
    for surface_name in surface_names:
        entries_for_surface = surface_leaderboards[surface_name].get("entries", [])
        if entries_for_surface:
            surface_metrics = _get_ordered_metrics_for_entries(
                entries_for_surface,
                always_include=_DEVICE_LEADERBOARD_ALWAYS_INCLUDE,
            )
            all_metric_keys_set.update(surface_metrics)

    if entries:
        all_metric_keys_set.update(_get_all_metrics_from_entries(entries))

    all_metric_keys = sorted(all_metric_keys_set)
    _ensure_flux_first(all_metric_keys)

    leaderboard_dir = out_rst.parent / "leaderboard"
    leaderboard_dir.mkdir(parents=True, exist_ok=True)

    # Main leaderboard.rst file with toctree
    main_lines = [
        "StellCoilBench Leaderboard",
        "===========================",
        "",
        "The StellCoilBench leaderboard provides a comprehensive comparison of coil optimization",
        "methods across different plasma surfaces. Each submission is evaluated using standardized",
        "metrics that measure both the quality of the magnetic field produced and the engineering",
        "feasibility of the coil designs.",
        "",
        ".. note::",
        "   This page is automatically regenerated by CI after each successful submission.",
        "   For local development, run ``stellcoilbench update-db`` to refresh the leaderboard.",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "   :caption: Leaderboard Contents",
        "",
        "   leaderboard/metric_definitions",
        "   leaderboard/surface_specific",
        "   leaderboard/reactor_scale",
        "",
    ]

    # Delegate metric definitions generation
    _write_metric_definitions_rst(all_metric_keys, leaderboard_dir)

    # Surface-specific leaderboards file
    repo_root = Path(out_rst.parent.parent).resolve()

    def _surface_header_fn(
        surface_name: str, surf_data: Dict[str, Any], entries: list
    ) -> list:
        display_name = _surface_display_name(surface_name)
        anchor = surface_name.replace(".", "-").replace("_", "-").lower()
        header = [
            f".. _{anchor}:",
            "",
            display_name,
            "^" * len(display_name),
            "",
            f"**Surface file:** ``{surface_name}``",
            "",
        ]
        if entries:
            header.extend(
                [
                    f"This surface has {len(entries)} submission(s).",
                    "",
                ]
            )
        return header

    def _surface_row_builder(
        entries: list, surface_name: str, surf_data: Dict[str, Any]
    ):
        surface_metric_keys = _get_ordered_metrics_for_entries(
            entries,
            always_include=_DEVICE_LEADERBOARD_ALWAYS_INCLUDE,
        )
        header_labels = [
            _shorthand_to_html_math("Score"),
            *[
                _shorthand_to_html_math(_metric_shorthand(k))
                for k in surface_metric_keys
            ],
            _shorthand_to_html_math("Date"),
            _shorthand_to_html_math("User"),
        ]
        all_rows: list[list[tuple[str, str]]] = []
        all_row_classes: list[str] = []
        for entry in entries:
            row_cells = _build_leaderboard_row_cells(
                entry,
                surface_metric_keys,
                ["score", "metrics", "date", "user"],
                with_sort_values=True,
                repo_root=repo_root,
            )
            all_rows.append(list(row_cells))
            all_row_classes.append("")
        return (header_labels, all_rows, all_row_classes)

    surface_specific_header_lines = [
        "Surface-Specific Leaderboards",
        "===============================",
        "",
    ]

    surface_specific_file = leaderboard_dir / "surface_specific.rst"
    _write_surface_leaderboard_rst(
        surface_leaderboards,
        _surface_row_builder,
        surface_specific_header_lines,
        surface_specific_file,
        "rst",
        empty_entries_message="No submissions found for this surface.",
        empty_entries_lines=[
            "No submissions found for this surface.",
            "",
            "Submit results using cases that reference this surface to appear on this leaderboard.",
            "",
        ],
        surface_header_fn=_surface_header_fn,
        footer_fn=_leaderboard_footer_rst,
        no_surfaces_lines=["No surface leaderboards generated yet.", ""],
    )

    out_rst.parent.mkdir(parents=True, exist_ok=True)
    out_rst.write_text("\n".join(main_lines))


def write_surface_leaderboards(
    surface_leaderboards: Dict[str, Dict[str, Any]],
    docs_dir: Path,
    repo_root: Path,
) -> list[str]:
    """Write per-surface leaderboard Markdown files.

    Each surface gets a file in docs_dir/leaderboards/ with sortable HTML
    tables and metric legend.

    Parameters
    ----------
    surface_leaderboards : dict[str, dict]
        Per-surface data from :func:`build_surface_leaderboards`.
    docs_dir : Path
        Docs root (e.g. repo_root / "docs").
    repo_root : Path
        Repository root (unused; kept for API compatibility).

    Returns
    -------
    list[str]
        Sorted list of surface names written.
    """
    surface_dir = docs_dir / "leaderboards"
    surface_dir.mkdir(parents=True, exist_ok=True)
    if not surface_dir.exists() or not surface_dir.is_dir():
        raise RuntimeError(f"Failed to create or access surface_dir: {surface_dir}")

    surface_names = sorted(surface_leaderboards.keys())

    for surface_name in surface_names:
        surf_data = surface_leaderboards[surface_name]
        entries = surf_data.get("entries", [])

        all_metric_keys = _get_ordered_metrics_for_entries(
            surf_data.get("entries", []),
            desired_order=[
                "final_squared_flux",
                "final_normalized_squared_flux",
                "num_coils",
                "fourier_continuation_orders",
            ],
        )

        display_name = _surface_display_name(surface_name)

        lines = [
            f"# {display_name} Leaderboard",
            "",
            f"**Plasma Surface:** `{surface_name}`",
            "",
            "[View all surfaces](../leaderboards/)",
            "",
            "---",
            "",
        ]

        if not entries:
            lines.append("_No submissions found for this plasma surface yet._")
            lines.append("")
            lines.append(
                "Submit results using cases that reference this surface to appear on this leaderboard."
            )
        else:
            header_cols = ["#", "Score", "User", "Date"]
            header_cols.extend(
                [
                    _shorthand_to_html_math(_metric_shorthand(key))
                    for key in all_metric_keys
                ]
            )
            safe_id = normalize_surface_id(surface_name, for_filename=True)
            table_id = f"leaderboard-{safe_id}"

            all_rows: list[list[tuple[str, str]]] = []
            all_row_classes: list[str] = []
            for entry in entries:
                row_cells = _build_leaderboard_row_cells(
                    entry,
                    all_metric_keys,
                    ["rank", "score", "user", "date", "metrics"],
                    with_sort_values=True,
                )
                all_rows.append(list(row_cells))
                all_row_classes.append("")

            lines.extend(
                render_html_table(
                    table_id,
                    header_cols,
                    all_rows,
                    row_classes=all_row_classes,
                    indent="",
                    rst_directive=False,
                )
            )
            n_entries = len(entries)
            if n_entries > 10:
                lines.append(
                    f'<p style="font-size: 0.9em; margin-top: 0.5em;">Showing top 10 of {n_entries} entries. Scroll to see more, or click column headers to sort.</p>'
                )
            lines.append("")
            lines.append("<script>")
            lines.extend(_sortable_table_script(table_id))
            lines.append("</script>")
            lines.append("")

            lines.append("")
            lines.append("### Legend")
            lines.append("")

            legend_items = []
            for key in all_metric_keys:
                definition = _metric_definition(key)
                legend_items.append(f"- {definition}")

            lines.extend(legend_items)
            lines.append("")

        safe_filename = normalize_surface_id(surface_name, for_filename=True)
        output_file = surface_dir / f"{safe_filename}.md"
        try:
            output_file.write_text("\n".join(lines))
        except OSError as e:
            logger.error("Failed to write %s: %s", output_file, e)
            raise

    return surface_names
