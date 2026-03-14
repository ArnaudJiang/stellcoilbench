"""
Leaderboard generation and database update for StellCoilBench.

Scans ``submissions/`` for zip files containing results.json, aggregates metrics,
and generates per-surface leaderboards in RST, Markdown, and JSON under
``docs/leaderboards/``. Handles reactor-scale constraints, composite scoring,
and submission formatting.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from ._constraints import (
    N_TURNS_MODEL,
    REACTOR_SCALE_CONSTRAINTS,
    check_reactor_constraints,
    compute_composite_score,
    normalize_submission_metrics,
)
from ._formatting import (
    _format_date,
    _format_metric_value,
    _format_numeric_for_leaderboard,
    _metric_definition,
    _metric_detailed_definition,
    _metric_display_name,
    _metric_shorthand,
    _shorthand_to_html_math,
    _shorthand_to_math,
)
from .submission_io import (
    _get_all_metrics_from_entries,
    _get_ordered_metrics_for_entries,
    _load_submissions,
    build_leaderboard_json,
    build_methods_json,
    build_surface_leaderboards,
)
from ._path_parsing import _extract_surface_from_submission_path, parse_submission_path
from ._recompute import _recompute_coils_linked_to_surface
from ._writers import (
    _surface_display_name,
    write_markdown_leaderboard,
    write_reactor_scale_leaderboard,
    write_rst_leaderboard,
    write_surface_leaderboards,
)

logger = logging.getLogger(__name__)

__all__ = [
    "N_TURNS_MODEL",
    "REACTOR_SCALE_CONSTRAINTS",
    "_extract_surface_from_submission_path",
    "_format_date",
    "_format_metric_value",
    "_format_numeric_for_leaderboard",
    "_get_all_metrics_from_entries",
    "_get_ordered_metrics_for_entries",
    "_load_submissions",
    "_metric_definition",
    "_metric_detailed_definition",
    "_metric_display_name",
    "_metric_shorthand",
    "_recompute_coils_linked_to_surface",
    "_shorthand_to_html_math",
    "_shorthand_to_math",
    "_surface_display_name",
    "build_leaderboard_json",
    "build_methods_json",
    "build_surface_leaderboards",
    "check_reactor_constraints",
    "compute_composite_score",
    "normalize_submission_metrics",
    "parse_submission_path",
    "update_database",
    "write_markdown_leaderboard",
    "write_reactor_scale_leaderboard",
    "write_rst_leaderboard",
    "write_surface_leaderboards",
]


def _build_and_write_leaderboard_json(
    submissions_root: Path,
    repo_root: Path,
    docs_dir: Path,
) -> dict:
    """Build methods JSON, leaderboard JSON, and write leaderboard.json to disk.

    Returns
    -------
    dict
        The leaderboard dict with entries and excluded_entries.
    """
    methods = build_methods_json(submissions_root=submissions_root, repo_root=repo_root)
    leaderboard = build_leaderboard_json(methods)
    if not isinstance(leaderboard, dict):
        leaderboard = {"entries": []}
    if "entries" not in leaderboard:
        leaderboard["entries"] = []

    leaderboard_file = docs_dir / "leaderboard.json"
    leaderboard_file.write_text(json.dumps(leaderboard, indent=2))
    if not leaderboard_file.exists() or leaderboard_file.stat().st_size == 0:
        logger.error("leaderboard.json was not written correctly!")
        raise RuntimeError("leaderboard.json was not written correctly")
    return leaderboard


def _write_all_leaderboard_artifacts(
    leaderboard: dict,
    submissions_root: Path,
    plasma_surfaces_dir: Path,
    docs_dir: Path,
    repo_root: Path,
    use_local_viz_links: bool = False,
) -> None:
    """Write surface leaderboards, RST, and reactor-scale leaderboards."""
    surface_leaderboards = build_surface_leaderboards(
        leaderboard, submissions_root, plasma_surfaces_dir
    )
    logger.info("Surface leaderboards built: %s", sorted(surface_leaderboards.keys()))
    for surface, data in surface_leaderboards.items():
        logger.info("  %s: %d entries", surface, len(data.get("entries", [])))

    surface_names = write_surface_leaderboards(
        surface_leaderboards, docs_dir=docs_dir, repo_root=repo_root
    )

    write_rst_leaderboard(
        leaderboard, docs_dir / "leaderboard.rst", surface_leaderboards
    )

    all_entries = (leaderboard.get("entries") or []) + (
        leaderboard.get("excluded_entries") or []
    )
    rs_entries = all_entries
    rs_leaderboard = {"entries": rs_entries}
    rs_surface_leaderboards = build_surface_leaderboards(
        rs_leaderboard, submissions_root, plasma_surfaces_dir
    )
    write_reactor_scale_leaderboard(
        leaderboard,
        rs_surface_leaderboards,
        docs_dir / "leaderboard" / "reactor_scale.rst",
        repo_root=repo_root,
        use_local_links=use_local_viz_links,
    )

    logger.info(
        "Generated %d surface leaderboard files: %s",
        len(surface_names),
        sorted(surface_names),
    )


def update_database(
    repo_root: Path,
    submissions_root: Path | None = None,
    docs_dir: Path | None = None,
    cases_root: Path | None = None,
    plasma_surfaces_dir: Path | None = None,
    *,
    use_local_viz_links: bool = False,
) -> dict:
    """
    High-level entry point to rebuild the leaderboard.

    It does several things:
      1. Scans submissions_root for results.json files
      2. Aggregates data from submissions (in-memory)
      3. Writes docs/leaderboards/ (per-surface leaderboards)
      4. Writes docs/leaderboard.json for reference

    Parameters
    ----------
    repo_root:
        Root of the git repo (e.g. Path.cwd() when called from repo root).
    submissions_root:
        Directory containing per-method submissions. Defaults to repo_root / "submissions".
    docs_dir:
        Directory where docs/leaderboards/ leaderboards and leaderboard.json are written. Defaults to repo_root / "docs".
    cases_root:
        Directory containing case.yaml files. Defaults to repo_root / "cases".
    plasma_surfaces_dir:
        Directory containing plasma surface files. Defaults to repo_root / "plasma_surfaces".
    use_local_viz_links : bool, optional
        If True, leaderboard PDF links use relative paths (e.g.
        ../../../../submissions/.../file.pdf) instead of jsDelivr CDN URLs.
        Use for local docs so PDFs open from disk and avoid the CDN 50 MB limit.

    Returns
    -------
    dict
        Summary with ``submissions_count``, ``surfaces_updated``, and ``errors``.

    Raises
    ------
    RuntimeError
        If leaderboard.json could not be written.
    """
    submissions_root = submissions_root or (repo_root / "submissions")
    docs_dir = docs_dir or (repo_root / "docs")
    plasma_surfaces_dir = plasma_surfaces_dir or (repo_root / "plasma_surfaces")

    docs_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = _build_and_write_leaderboard_json(
        submissions_root, repo_root, docs_dir
    )
    surface_leaderboards = build_surface_leaderboards(
        leaderboard, submissions_root, plasma_surfaces_dir
    )
    _write_all_leaderboard_artifacts(
        leaderboard,
        submissions_root,
        plasma_surfaces_dir,
        docs_dir,
        repo_root,
        use_local_viz_links=use_local_viz_links,
    )

    all_entries = (leaderboard.get("entries") or []) + (
        leaderboard.get("excluded_entries") or []
    )
    summary: dict = {
        "submissions_count": len(all_entries),
        "surfaces_updated": len(surface_leaderboards),
        "errors": [],
    }

    # Rebuild Sphinx HTML so docs/_build/html reflects the updated leaderboard
    sphinx_srcdir = docs_dir
    sphinx_outdir = docs_dir / "_build" / "html"
    try:
        subprocess.run(
            ["sphinx-build", "-b", "html", str(sphinx_srcdir), str(sphinx_outdir)],
            check=True,
        )
        logger.info("Rebuilt docs HTML.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("Could not rebuild docs HTML (%s). Open docs manually.", e)
        summary["errors"].append(str(e))

    return summary
