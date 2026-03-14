"""update-db CLI command implementation.

Scans the submissions directory for results.json (and packaged zip files),
aggregates metrics into docs/leaderboard.json, and writes per-surface
leaderboards (RST, Markdown) to docs/leaderboards/. Used by CI and for
local leaderboard regeneration.
"""

from __future__ import annotations

from pathlib import Path

import typer


def update_db_cmd(
    submissions_dir: Path = typer.Argument(
        Path("submissions"),
        help="Directory containing per-method submissions (results.json files).",
    ),
    docs_dir: Path = typer.Option(
        Path("docs"),
        "--docs-dir",
        help="Directory where docs/leaderboards/ leaderboards and leaderboard.json are written.",
    ),
    local_viz_links: bool = typer.Option(
        False,
        "--local-viz-links",
        help="Use relative file paths for PDF links instead of jsDelivr CDN. Use when viewing docs locally so PDFs open from disk (avoids 50 MB CDN limit).",
    ),
) -> None:
    """
    Rebuild the on-repo 'database' of submissions and leaderboards.

    This scans submissions_dir for results.json files, aggregates them into
    docs/leaderboard.json, and writes per-surface leaderboards in docs/leaderboards/.
    """
    from ..cli_helpers import _cli_error
    from ..update_db import update_database

    repo_root = Path.cwd()
    try:
        summary = update_database(
            repo_root=repo_root,
            submissions_root=submissions_dir,
            docs_dir=docs_dir,
            use_local_viz_links=local_viz_links,
        )
        n_surf = summary.get("surfaces_updated", 0)
        n_sub = summary.get("submissions_count", 0)
        typer.echo(f"Updated {n_surf} surfaces from {n_sub} submissions.")
    except (RuntimeError, OSError) as e:
        _cli_error(f"Update failed: {e}")


def register(app: typer.Typer) -> None:
    app.command("update-db")(update_db_cmd)
