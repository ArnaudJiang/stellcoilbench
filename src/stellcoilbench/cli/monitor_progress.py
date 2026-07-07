"""CLI wrapper for the lightweight progress monitor."""

from __future__ import annotations

import typer

from ..monitor_progress import monitor_progress_cmd


def register(app: typer.Typer) -> None:
    """Register the monitor-progress command with the Typer app."""

    app.command("monitor-progress")(monitor_progress_cmd)
