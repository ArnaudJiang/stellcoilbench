"""validate-config CLI command.

Validates case YAML files and optionally emits a JSON schema summary
for editor support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ..cli_helpers import _cli_error
from ..validate_config import validate_case_yaml_file


def _emit_schema_summary() -> None:
    """Print JSON schema of CaseConfig for editor autocomplete."""
    schema = {
        "description": "str, required",
        "surface_params": {
            "surface": "str, required (e.g. input.LandremanPaul2021_QA)",
            "range": "str, optional: 'half period' | 'full torus'",
            "virtual_casing": "bool, optional",
        },
        "coils_params": {
            "ncoils": "int, required, positive",
            "order": "int, required, positive",
            "coil_type": "str, optional",
            "target_B": "float, optional",
            "coil_width": "float, optional",
        },
        "optimizer_params": {
            "algorithm": "str, required (e.g. L-BFGS-B, BFGS, SLSQP)",
            "max_iterations": "int, required, positive",
            "max_iter_subopt": "int, optional, positive",
            "verbose": "bool, optional",
            "algorithm_options": "dict, optional",
        },
        "coil_objective_terms": "dict, optional",
        "fourier_continuation": "dict, optional",
    }
    typer.echo(json.dumps(schema, indent=2))


def validate_config_cmd(
    case_path: Optional[Path] = typer.Argument(
        None,
        help="Path to case.yaml (e.g., cases/basic_tokamak.yaml).",
    ),
    schema: bool = typer.Option(
        False,
        "--schema",
        help="Emit JSON schema summary for editor support.",
    ),
    plasma_surfaces_dir: Optional[Path] = typer.Option(
        None,
        "--plasma-surfaces-dir",
        help="Directory containing plasma surface files.",
    ),
) -> None:
    """Validate a case YAML configuration file."""
    if schema:
        _emit_schema_summary()
        return
    if case_path is None:
        _cli_error("case_path is required when not using --schema")
    if not case_path.exists():
        _cli_error(f"File not found: {case_path}")
    errors = validate_case_yaml_file(case_path, surfaces_dir=plasma_surfaces_dir)
    if errors:
        for err in errors:
            typer.echo(err, err=True)
        raise typer.Exit(1)
    typer.echo("Configuration valid.")


def register(app: typer.Typer) -> None:
    """Register the validate-config command with the Typer app."""
    app.command("validate-config")(validate_config_cmd)
