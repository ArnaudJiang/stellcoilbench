"""
Typer CLI package for StellCoilBench.

Provides commands: submit-case, run-case, run-ci-case, update-db,
generate-submission, post-process. Subcommand implementations
live in sibling modules; this module wires the app and exports the public API.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

import typer

from ..ci_autopilot import _write_autopilot_submission  # noqa: F401 — re-exported for tests
from ..cli_helpers import (  # noqa: F401 — re-exported for tests and external use
    NumpyJSONEncoder,
    _cli_error,
    _detect_github_username,
    _detect_hardware,
    _fmt_scalar,
    _resolve_github_username,
    _write_json,
    _zip_submission_directory,
)
from ..mpi_utils import is_proc0  # noqa: F401 — re-exported for tests
from ..version_utils import _get_version_info  # noqa: F401 — re-exported for tests

from .._optional_imports import require_reactor_scale_compute

_compute_reactor_scale_metrics = require_reactor_scale_compute()  # noqa: F401

from ..constants import DEFAULT_CI_TIMEOUT_MINUTES  # noqa: F401 — re-exported for submit_run

app = typer.Typer(
    help=(
        "StellCoilBench: benchmarking framework for stellarator coil optimization. "
        "Commands: validate-config, list-cases, submit-case, run-case, run-ci-case, "
        "update-db, generate-submission, post-process, sensitivity."
    ),
)


def _version_callback(value: bool) -> None:
    """Print package version and exit when --version is passed."""
    if value:
        try:
            v = importlib.metadata.version("stellcoilbench")
        except importlib.metadata.PackageNotFoundError:
            v = "0.0.0.dev"
        typer.echo(f"stellcoilbench {v}")
        raise typer.Exit(0)


@app.callback()
def _root_callback(
    version: bool = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Root callback; --version is handled by the callback."""


def _apply_all_post_processing_flags(
    *args: Any, **kwargs: Any
) -> tuple[bool, bool, bool, bool, bool, bool, bool]:
    """Backward-compat alias for tests that patch this."""
    from ._shared import apply_all_post_processing_flags

    return apply_all_post_processing_flags(*args, **kwargs)


def _register_commands() -> None:
    """Register all subcommands with the app."""
    from . import list_cases_cmd
    from . import post_process
    from . import sensitivity_cmd
    from . import submit_run
    from . import update_db_cmd
    from . import validate_cmd

    validate_cmd.register(app)
    list_cases_cmd.register(app)
    update_db_cmd.register(app)
    submit_run.register(app)
    post_process.register(app)
    sensitivity_cmd.register(app)


_register_commands()


def main() -> None:
    app()


# Re-export command functions for tests (imports after _register_commands to avoid circular deps)
from . import list_cases_cmd as _list_cases_mod  # noqa: E402
from . import post_process as _post_process_mod  # noqa: E402
from . import submit_run as _submit_run_mod  # noqa: E402
from . import update_db_cmd as _update_db_mod  # noqa: E402
from . import validate_cmd as _validate_mod  # noqa: E402

list_cases = _list_cases_mod.list_cases_cmd
update_db_cmd = _update_db_mod.update_db_cmd
validate_config_cmd = _validate_mod.validate_config_cmd
submit_case = _submit_run_mod.submit_case
run_case = _submit_run_mod.run_case
run_ci_case = _submit_run_mod.run_ci_case
generate_submission = _submit_run_mod.generate_submission
post_process = _post_process_mod.post_process
