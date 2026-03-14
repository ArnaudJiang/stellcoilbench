"""list-cases CLI command.

Scans cases/ for *.yaml files and displays a table of available benchmark cases.
Skips cases/pending/ and cases/done/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from ..path_utils import load_yaml


def _get_description(data: dict[str, Any]) -> str:
    """Extract description from case YAML, truncated for display."""
    desc = data.get("description") or ""
    if isinstance(desc, str) and len(desc) > 60:
        return desc[:57] + "..."
    return str(desc)


def _get_surface(data: dict[str, Any]) -> str:
    """Extract surface identifier from case YAML."""
    sp = data.get("surface_params")
    if isinstance(sp, dict) and "surface" in sp:
        return str(sp["surface"])
    return "—"


def list_cases_cmd(
    cases_dir: Path = typer.Argument(
        Path("cases"),
        help="Directory containing case YAML files.",
    ),
) -> None:
    """List available benchmark cases from cases/*.yaml."""
    skip_subdirs = {"pending", "done"}
    if not cases_dir.exists():
        typer.echo(f"Directory not found: {cases_dir}", err=True)
        raise typer.Exit(1)
    rows: list[tuple[str, str, str]] = []
    for yaml_path in sorted(cases_dir.glob("*.yaml")):
        data: dict[str, Any] | None = {}
        try:
            data = load_yaml(path=yaml_path)
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        rows.append(
            (
                yaml_path.name,
                _get_surface(data),
                _get_description(data),
            )
        )
    for sub in sorted(cases_dir.iterdir()):
        if sub.is_dir() and sub.name not in skip_subdirs:
            for yaml_path in sorted(sub.glob("*.yaml")):
                rel = str(yaml_path.relative_to(cases_dir))
                data = {}
                try:
                    data = load_yaml(path=yaml_path) or {}
                except Exception:
                    pass
                if not isinstance(data, dict):
                    data = {}
                rows.append(
                    (
                        rel,
                        _get_surface(data),
                        _get_description(data),
                    )
                )
    if not rows:
        typer.echo("No case YAML files found.")
        return
    col_widths = [
        max(len(r[0]) for r in rows) + 2,
        max(len(r[1]) for r in rows) + 2,
        max(min(len(r[2]), 60) for r in rows) + 2,
    ]
    header = ("Filename", "Surface", "Description")
    typer.echo(
        header[0].ljust(col_widths[0]) + header[1].ljust(col_widths[1]) + header[2]
    )
    typer.echo("-" * (sum(col_widths) + 10))
    for name, surf, desc in rows:
        d = desc[:60] + "..." if len(desc) > 60 else desc
        typer.echo(name.ljust(col_widths[0]) + surf.ljust(col_widths[1]) + d)


def register(app: typer.Typer) -> None:
    """Register the list-cases command with the Typer app."""
    app.command("list-cases")(list_cases_cmd)
