"""Shared table formatting utilities, RST/HTML table builders, and helper
functions used across the leaderboard writer sub-modules.

The :class:`LeaderboardRowBuilder` protocol formalizes the row-builder callback
signature used by :func:`_write_surface_leaderboard_rst` across surface-specific,
reactor-scale, and dipole leaderboard writers. Shared helpers ``_user_cell`` and
``_viz_link_cells`` reduce duplication when building table rows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple

from ..path_utils import normalize_surface_id

from ._formatting import (
    _get_surface_display_names,
    _metric_display_name,
)
from ._table_builders import (
    _build_leaderboard_row_cells,
    _build_reactor_scale_row_cells,
    _format_metric_html,
    _get_constraint_violation_sets,
    _leaderboard_table_styles,
    _leaderboard_table_wrapper_open,
    _sortable_table_script,
    _user_cell,
    _viz_link_cells,
    render_html_table,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LeaderboardRowBuilder",
    "_build_leaderboard_row_cells",
    "_sortable_table_script",
    "_build_reactor_scale_row_cells",
    "_format_metric_html",
    "_get_constraint_violation_sets",
    "_leaderboard_footer_md",
    "_leaderboard_footer_rst",
    "_leaderboard_table_styles",
    "_leaderboard_table_wrapper_open",
    "_surface_display_name",
    "_user_cell",
    "_viz_link_cells",
    "_write_surface_leaderboard_rst",
    "_write_surface_leaderboard_table",
    "render_html_table",
]


def _resolve_repo_root(out_rst: Path, repo_root: Path | None) -> Path:
    """Resolve repo root from output RST path when not explicitly provided.

    Infers docs/leaderboard/<name>.rst -> repo is parent.parent.parent;
    verifies by checking for submissions/ directory.
    """
    if repo_root is not None:
        return repo_root
    resolved = Path(out_rst.parent.parent.parent).resolve()
    if not (resolved / "submissions").is_dir():
        return Path.cwd()
    return resolved


def _surface_display_name(surface_name: str) -> str:
    """Return human-friendly display name for a plasma surface (e.g. Landreman-Paul QA)."""
    names = _get_surface_display_names()
    if surface_name in names:
        return names[surface_name]
    base = normalize_surface_id(surface_name)
    if base in names:
        return names[base]
    return _metric_display_name(surface_name)


# ---------------------------------------------------------------------------
# Footer helpers
# ---------------------------------------------------------------------------


def _leaderboard_footer_md() -> list[str]:
    """Return Markdown footer lines for leaderboard update instruction.

    Returns
    -------
    list[str]
        Lines to append to a Markdown leaderboard: separator, blank line,
        and italic text instructing users to run ``stellcoilbench update-db``.
    """
    return [
        "",
        "---",
        "",
        "*Last updated: Run `stellcoilbench update-db` to refresh.*",
    ]


def _leaderboard_footer_rst() -> list[str]:
    """Return RST note block for leaderboard update instruction.

    Returns
    -------
    list[str]
        Lines for an RST ``.. note::`` block instructing users to run
        ``stellcoilbench update-db`` to refresh locally.
    """
    return [
        "",
        ".. note::",
        "   Last updated: run ``stellcoilbench update-db`` to refresh locally.",
        "",
    ]


# ---------------------------------------------------------------------------
# LeaderboardRowBuilder protocol
# ---------------------------------------------------------------------------


class LeaderboardRowBuilder(Protocol):
    """Protocol for row builder callbacks passed to :func:`_write_surface_leaderboard_rst`.

    Implementations (e.g. surface-specific, reactor-scale, dipole) receive
    entries and per-surface metadata, and return header labels plus rows
    of (display, sort_value) cell tuples.
    """

    def __call__(
        self,
        entries: list,
        surface_name: str,
        surf_data: Dict[str, Any],
    ) -> Optional[
        Tuple[Sequence[str], Sequence[Sequence[Tuple[str, str]]], Sequence[str]]
    ]:
        """Build table header labels and row data for a surface.

        Parameters
        ----------
        entries : list
            Leaderboard entries for this surface.
        surface_name : str
            Surface identifier.
        surf_data : dict
            Per-surface metadata (may include excluded entries, etc.).

        Returns
        -------
        tuple or None
            ``(header_labels, rows, row_classes)`` when a table can be
            rendered, or ``None`` when no table data is available.
        """
        ...


# ---------------------------------------------------------------------------
# Per-surface leaderboard table loop helper
# ---------------------------------------------------------------------------


def _write_surface_leaderboard_rst(
    surface_leaderboards: Dict[str, Dict[str, Any]],
    row_builder_fn: LeaderboardRowBuilder,
    header_lines: Sequence[str],
    out_path: Path,
    table_id_prefix: str,
    *,
    empty_entries_message: str = "No submissions for this surface.",
    empty_no_data_message: Optional[str] = None,
    empty_entries_lines: Optional[Sequence[str]] = None,
    trailer_lines: Optional[Sequence[str]] = None,
    surface_header_fn: Optional[
        Callable[[str, Dict[str, Any], list], Sequence[str]]
    ] = None,
    footer_fn: Optional[Callable[[], Sequence[str]]] = None,
    no_surfaces_lines: Optional[Sequence[str]] = None,
) -> None:
    """Write a complete RST leaderboard file with per-surface tables.

    Composes header_lines, per-surface table sections (via row_builder_fn),
    optional trailer (e.g. plot section), and footer. Each writer provides
    its own row_builder_fn and header_lines; the common loop is shared.

    Parameters
    ----------
    surface_leaderboards : dict[str, dict]
        Per-surface data; each value has at least an ``"entries"`` list.
    row_builder_fn : callable
        Called as ``row_builder_fn(entries, surface_name, surf_data)``.
        Returns ``(header_labels, rows, row_classes)`` when a table can be
        rendered, or ``None`` when the surface has entries but no table data.
    header_lines : sequence[str]
        Document header lines (title, description, etc.) written before tables.
    out_path : Path
        Output path for the RST file.
    table_id_prefix : str
        Prefix for HTML table ids (e.g. ``"rs"``, ``"dipole"``).
    empty_entries_message : str, default "No submissions for this surface."
        Message when a surface has no entries. Ignored when
        empty_entries_lines is provided.
    empty_no_data_message : str or None, default None
        Message when entries exist but row_builder_fn returns ``None``.
    empty_entries_lines : sequence[str] or None, default None
        When provided, used instead of empty_entries_message for the
        no-entries case (allows multi-line empty messages).
    trailer_lines : sequence[str] or None, default None
        Optional lines inserted before the footer (e.g. composite score plot).
    surface_header_fn : callable or None, default None
        When provided, called as ``surface_header_fn(surface_name, surf_data,
        entries)`` to generate per-surface header lines. When None, uses
        display_name and underline.
    footer_fn : callable or None, default None
        When provided, called to get footer lines. When None, uses
        :func:`_leaderboard_footer_rst`.
    no_surfaces_lines : sequence[str] or None, default None
        When surface_leaderboards is empty and this is provided, these lines
        are used instead of the (empty) table sections.
    """
    lines: list[str] = list(header_lines)
    if not surface_leaderboards and no_surfaces_lines is not None:
        lines.extend(no_surfaces_lines)
    else:
        lines.extend(
            _write_surface_leaderboard_table(
                surface_leaderboards,
                row_builder_fn,
                table_id_prefix,
                empty_entries_message=empty_entries_message,
                empty_no_data_message=empty_no_data_message,
                empty_entries_lines=empty_entries_lines,
                surface_header_fn=surface_header_fn,
            )
        )
    if trailer_lines:
        lines.extend(trailer_lines)
    if footer_fn is None:
        footer_fn = _leaderboard_footer_rst
    lines.extend(footer_fn())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def _write_surface_leaderboard_table(
    surface_leaderboards: Dict[str, Dict[str, Any]],
    row_builder_fn: LeaderboardRowBuilder,
    table_id_prefix: str,
    *,
    empty_entries_message: str = "No submissions for this surface.",
    empty_no_data_message: Optional[str] = None,
    empty_entries_lines: Optional[Sequence[str]] = None,
    surface_header_fn: Optional[
        Callable[[str, Dict[str, Any], list], Sequence[str]]
    ] = None,
) -> list[str]:
    """Generate per-surface leaderboard table sections for RST/Markdown.

    Iterates over sorted surface_leaderboards, writing a header (display name,
    underline) and either an empty message or a rendered HTML table for each
    surface. Used by reactor-scale and dipole leaderboard writers.

    Parameters
    ----------
    surface_leaderboards : dict[str, dict]
        Per-surface data; each value has at least an ``"entries"`` list.
    row_builder_fn : callable
        Called as ``row_builder_fn(entries, surface_name, surf_data)``.
        Returns ``(header_labels, rows, row_classes)`` when a table can be
        rendered, or ``None`` when the surface has entries but no table data
        (e.g. no reactor-scale keys). Each row is a sequence of
        ``(display_html, sort_value)`` tuples.
    table_id_prefix : str
        Prefix for HTML table ids (e.g. ``"rs"``, ``"dipole"``).
    empty_entries_message : str, default "No submissions for this surface."
        Message to show when a surface has no entries.
    empty_no_data_message : str or None, default None
        Message when entries exist but ``row_builder_fn`` returns ``None``.
        Falls back to ``empty_entries_message`` when not set.
    empty_entries_lines : sequence[str] or None, default None
        When provided, used instead of empty_entries_message for the
        no-entries case.
    surface_header_fn : callable or None, default None
        When provided, called as ``surface_header_fn(surface_name, surf_data,
        entries)`` to generate per-surface header lines.

    Returns
    -------
    list[str]
        Lines for the per-surface sections (header + table or message + blanks).
    """
    no_data_msg = empty_no_data_message or empty_entries_message
    no_entries_lines: Sequence[str] = (
        empty_entries_lines
        if empty_entries_lines is not None
        else [empty_entries_message, ""]
    )
    lines: list[str] = []

    for surface_name, surf_data in sorted(surface_leaderboards.items()):
        entries = surf_data.get("entries", [])
        if surface_header_fn is not None:
            lines.extend(surface_header_fn(surface_name, surf_data, entries))
        else:
            display_name = _surface_display_name(surface_name)
            lines.extend([display_name, "-" * len(display_name), ""])

        if not entries:
            lines.extend(no_entries_lines)
            continue

        result = row_builder_fn(entries, surface_name, surf_data)
        if result is None:
            lines.extend([no_data_msg, ""])
            continue

        header_labels, rows, row_classes = result
        safe_id = normalize_surface_id(surface_name, for_filename=True)
        table_id = f"leaderboard-{table_id_prefix}-{safe_id}"
        lines.extend(
            render_html_table(table_id, header_labels, rows, row_classes=row_classes)
        )
        lines.extend(["", ""])

    return lines
