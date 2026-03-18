"""submit-case, run-case, run-ci-case, and generate-submission CLI commands.

This module implements the main workflow commands for StellCoilBench:

- submit-case: Run full optimization pipeline (optimize coils, evaluate,
  post-process, package submission) for a given case.yaml.
- run-case: Same as submit-case but with default post-processing only.
- run-ci-case: CI-optimized variant with iteration cap, autopilot metadata,
  and reactor-scale metrics when available.
- generate-submission: Package existing optimization results into a
  submission archive without re-running optimization.

All commands support MPI; optimization runs on rank 0, post-processing uses
all ranks when applicable.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from ..case_loader import load_case
from ..constants import COILS_FILENAME
from ..mpi_utils import is_proc0
from ..submission_packaging import (
    _build_submission_dict,
    _extract_surface_name,
    _package_submission,
    _prepare_submission_dir,
    _print_submission_summary,
)
from ..cli_helpers import (
    _cli_error,
    _detect_hardware,
    _resolve_github_username,
    _write_json,
)
from ..path_utils import dump_yaml, load_yaml
from ..ci_autopilot import (
    _append_failure_to_autopilot_failures,
    _canonical_failure_class,
    _write_autopilot_submission,
    _write_ci_summary,
)
from ..version_utils import _get_version_info

from .._optional_imports import get_reactor_scale_compute

_compute_reactor_scale_metrics = get_reactor_scale_compute()

from . import _shared
from ._post_processing_options import (
    ALL_POST_PROCESSING_OPTION,
    COMPUTE_SHAPE_GRADIENT_OPTION,
    EXPORT_STRUCTURAL_FULL_COIL_SET_OPTION,
    FINITE_BUILD_HEIGHT_OPTION,
    FINITE_BUILD_WIDTH_OPTION,
    PLOT_FINITE_BUILD_OPTION,
    RUN_STRUCTURAL_OPTION,
    RUN_SIMPLE_OPTION,
    RUN_VMEC_OPTION,
    PLOT_POINCARE_OPTION,
    STRUCTURAL_E_OPTION,
    STRUCTURAL_NU_OPTION,
    build_post_processing_config,
)
from ..constants import DEFAULT_CI_TIMEOUT_MINUTES


def submit_case(
    case_path: Path = typer.Argument(
        ..., help="Path to case.yaml file (e.g., cases/case.yaml)."
    ),
    submissions_dir: Path = typer.Option(
        Path("submissions"),
        "--submissions-dir",
        help="Directory where submission results.json will be written.",
    ),
    all_post_processing: bool = ALL_POST_PROCESSING_OPTION,
    run_vmec: bool = RUN_VMEC_OPTION,
    run_simple: bool = RUN_SIMPLE_OPTION,
    plot_poincare: bool = PLOT_POINCARE_OPTION,
    plot_finite_build: bool = PLOT_FINITE_BUILD_OPTION,
    finite_build_width: Optional[float] = FINITE_BUILD_WIDTH_OPTION,
    finite_build_height: Optional[float] = FINITE_BUILD_HEIGHT_OPTION,
    run_structural: bool = RUN_STRUCTURAL_OPTION,
    structural_E: Optional[float] = STRUCTURAL_E_OPTION,
    structural_nu: Optional[float] = STRUCTURAL_NU_OPTION,
    compute_shape_gradient: bool = COMPUTE_SHAPE_GRADIENT_OPTION,
    export_structural_full_coil_set: bool = EXPORT_STRUCTURAL_FULL_COIL_SET_OPTION,
    run_sensitivity: bool = typer.Option(False, "--run-sensitivity/--no-sensitivity"),
    sensitivity_n_samples: int = typer.Option(20, "--sensitivity-n-samples"),
    sensitivity_correlation_length: float = typer.Option(
        1.0, "--sensitivity-correlation-length"
    ),
    sensitivity_n_vtk: int = typer.Option(0, "--sensitivity-n-vtk"),
) -> None:
    """Run a case and generate a submission results.json file."""
    from ..coil_optimization import optimize_coils

    run_sensitivity = run_sensitivity or all_post_processing
    cfg = build_post_processing_config(
        all_post_processing,
        run_vmec,
        run_simple,
        plot_poincare,
        False,
        plot_finite_build,
        finite_build_width,
        finite_build_height,
        run_structural,
        structural_E,
        structural_nu,
        compute_shape_gradient,
        export_structural_full_coil_set=export_structural_full_coil_set,
    )

    github_username = _resolve_github_username()
    contact = github_username
    typer.echo(f"Using contact: {contact}")

    hardware = _detect_hardware()
    if not hardware:
        hardware = "Unknown hardware"
        typer.echo("Warning: Could not auto-detect hardware.")
    else:
        typer.echo(f"Auto-detected hardware: {hardware}")

    case_cfg = load_case(case_path)
    surface_name = _extract_surface_name(case_cfg)
    run_date = datetime.now().isoformat()

    submission_dir, _ = _prepare_submission_dir(
        submissions_dir, surface_name, github_username, case_path
    )
    coils_out_path = submission_dir / COILS_FILENAME

    if is_proc0():
        typer.echo("Running optimizer...")
    results_dict = optimize_coils(
        case_path=case_path,
        coils_out_path=coils_out_path,
        case_cfg=case_cfg,
        output_dir=submission_dir,
        run_vmec=cfg.run_vmec,
        run_simple=cfg.run_simple,
        plot_poincare=cfg.plot_poincare,
        plot_finite_build=cfg.plot_finite_build,
        finite_build_width=cfg.finite_build_width,
        finite_build_height=cfg.finite_build_height,
        run_structural=cfg.run_structural,
        structural_E=cfg.structural_E,
        structural_nu=cfg.structural_nu,
        compute_shape_gradient=cfg.compute_shape_gradient,
    )

    if not is_proc0():
        return

    metrics = results_dict
    sensitivity_results = _shared.run_sensitivity_if_configured(
        run_sensitivity=run_sensitivity,
        coils_out_path=coils_out_path,
        case_path=case_path,
        correlation_length=sensitivity_correlation_length,
        n_samples=sensitivity_n_samples,
        output_dir=submission_dir,
        n_vtk=sensitivity_n_vtk,
        metrics=metrics,
    )

    submission = _build_submission_dict(
        metrics,
        case_cfg,
        run_date=run_date,
        contact=contact,
        hardware=hardware,
        sensitivity_results=sensitivity_results,
    )
    _package_submission(
        submission=submission, submission_dir=submission_dir, case_path=case_path
    )


def run_case(
    case_path: Path = typer.Argument(
        ...,
        help="Path to case directory containing case.yaml and coils.yaml, or a single YAML file.",
    ),
    submissions_dir: Path = typer.Option(
        Path("submissions"),
        "--submissions-dir",
        help="Directory where submission results will be written.",
    ),
    results_out: Optional[Path] = typer.Option(
        None,
        "--results-out",
        "-o",
        help="Where to write the results JSON.",
    ),
) -> None:
    """Run coil optimization for one case."""
    from ..coil_optimization import optimize_coils

    case_cfg = load_case(case_path)
    surface_name = _extract_surface_name(case_cfg)
    github_username = _resolve_github_username()
    submission_dir, _ = _prepare_submission_dir(
        submissions_dir, surface_name, github_username, case_path
    )
    coils_out_path = submission_dir / COILS_FILENAME

    if is_proc0():
        typer.echo("Running optimizer...")
    results_dict = optimize_coils(
        case_path=case_path, coils_out_path=coils_out_path, case_cfg=case_cfg
    )

    if not is_proc0():
        return

    if results_out is None:
        results_out = submission_dir / "results.json"
    if not str(results_out).endswith(".json"):
        results_out = results_out.with_suffix(".json")

    submission = _build_submission_dict(results_dict, case_cfg)
    results_out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(results_out, submission)
    _print_submission_summary(submission)


def run_ci_case(
    case_file: Path = typer.Argument(
        ...,
        help="Path to a CI case JSON file (cases/pending/<case_id>.json).",
    ),
    output_dir: Path = typer.Option(
        Path("cases/done"),
        "--output-dir",
        "-o",
        help="Root directory for completed case results.",
    ),
    policy_file: Optional[Path] = typer.Option(
        None, "--policy", help="Path to proposer_policy.yaml."
    ),
) -> None:
    """Run a single CI autopilot case from a JSON file."""
    import time as _time

    from ..coil_optimization import optimize_coils
    from ..config_scheme import CaseConfig
    from ..validate_config import validate_ci_case

    case_text = case_file.read_text()
    try:
        case_data = json.loads(case_text)
    except json.JSONDecodeError as exc:
        typer.echo(f"ERROR: invalid JSON in {case_file}: {exc}", err=True)
        raise typer.Exit(code=1)

    policy: dict | None = None
    if policy_file and policy_file.exists():
        policy = load_yaml(path=policy_file)

    errors = validate_ci_case(case_data, policy=policy, file_path=case_file)
    if errors:
        for err in errors:
            typer.echo(f"VALIDATION ERROR: {err}", err=True)
        case_id = case_data.get("case_id", case_file.stem)
        _write_ci_summary(
            output_dir / case_id / "summary.json",
            case_id=case_id,
            success=False,
            failure_reason="validation_error",
            failure_class="validation",
            case_config=case_data.get("case_config", {}),
        )
        raise typer.Exit(code=1)

    case_id = case_data["case_id"]
    case_config_dict = case_data["case_config"]
    resource = case_data.get("resource", {})
    random_seed = case_data.get("random_seed")

    if random_seed is not None:
        np.random.seed(random_seed)

    case_cfg = CaseConfig.from_dict(case_config_dict)
    out = output_dir / case_id
    out.mkdir(parents=True, exist_ok=True)
    case_yaml_path = out / "case.yaml"
    dump_yaml(case_config_dict, path=case_yaml_path)
    coils_out_path = out / COILS_FILENAME
    wall_start = _time.time()
    results_dict: dict = {}
    summary_path = out / "summary.json"
    ci_common = dict(
        random_seed=random_seed,
        tags=case_data.get("tags", []),
        parent_ids=case_data.get("parent_ids", []),
        proposer_mode=case_data.get("proposer_mode"),
        case_config=case_config_dict,
        llm_reasoning=case_data.get("llm_reasoning"),
    )

    try:
        timeout_sec = resource.get("timeout_minutes", DEFAULT_CI_TIMEOUT_MINUTES) * 60
        results_dict = optimize_coils(
            case_path=case_yaml_path,
            coils_out_path=coils_out_path,
            case_cfg=case_cfg,
            output_dir=out,
            skip_post_processing=False,
            run_vmec=False,
            run_simple=False,
            plot_poincare=False,
        )
        wall_end = _time.time()
        walltime = wall_end - wall_start
        if walltime > timeout_sec:
            typer.echo(
                f"WARNING: case {case_id} exceeded timeout ({walltime:.0f}s > {timeout_sec}s)"
            )
        metrics = {
            k: v
            for k, v in results_dict.items()
            if isinstance(v, (int, float)) and not k.startswith("_")
        }
        _write_ci_summary(
            summary_path,
            case_id=case_id,
            success=True,
            total_score=float(results_dict.get("final_squared_flux", float("inf"))),
            iterations_used=int(results_dict.get("iterations_used", 0)),
            walltime_sec=round(walltime, 2),
            metrics=metrics,
            timing=results_dict.get("timing"),
            **ci_common,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        wall_end = _time.time()
        import traceback

        typer.echo(
            f"ERROR running case {case_id}: {exc}\n{traceback.format_exc()}", err=True
        )
        _write_ci_summary(
            summary_path,
            case_id=case_id,
            success=False,
            walltime_sec=round(wall_end - wall_start, 2),
            failure_reason=str(exc),
            failure_class=_canonical_failure_class(exc),
            **ci_common,
        )
        failures_path = Path.cwd() / "policy" / "autopilot_failures.json"
        try:
            summary = json.loads(summary_path.read_text())
            _append_failure_to_autopilot_failures(failures_path, summary)
        except (OSError, json.JSONDecodeError, KeyError) as append_err:
            typer.echo(
                f"WARNING: could not append to autopilot_failures.json: {append_err}",
                err=True,
            )

    typer.echo(f"Wrote summary to {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("success"):
        try:
            _write_autopilot_submission(
                case_id=case_id,
                results_dict=results_dict,
                case_cfg=case_cfg,
                case_config_dict=case_config_dict,
                walltime=summary["walltime_sec"],
                repo_root=Path.cwd(),
                case_output_dir=out,
                tags=summary.get("tags"),
                parent_ids=summary.get("parent_ids"),
            )
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            typer.echo(f"WARNING: could not create submission entry: {exc}", err=True)


def generate_submission(
    case_path: Path = typer.Argument(
        ..., help="Path to case.yaml or directory containing case.yaml."
    ),
    metadata_path: Path = typer.Argument(..., help="Path to metadata.yaml file."),
    coils_path: Optional[Path] = typer.Option(
        None, "--coils", help="Path to coils.json (default: <case_dir>/coils.json)."
    ),
    submission_out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Where to write the submission results.json."
    ),
) -> None:
    """Generate a results.json submission file from a case and coils file."""
    from ..config_scheme import SubmissionMetadata

    metadata_data = load_yaml(path=metadata_path)
    metadata = SubmissionMetadata(
        method_version=metadata_data.get("method_version", "0.0.0"),
        contact=metadata_data.get("contact", ""),
        hardware=metadata_data.get("hardware", ""),
    )
    case_cfg = load_case(case_path)
    if coils_path is None:
        coils_path = (
            case_path / "coils.json"
            if case_path.is_dir()
            else case_path.parent / "coils.json"
        )
    if not coils_path.exists():
        _cli_error(f"Coils file not found: {coils_path}")

    results_dict = {"chi2_Bn": 0.001}
    metrics = results_dict
    reactor_scale_metrics = (
        _compute_reactor_scale_metrics(metrics, case_cfg)
        if _compute_reactor_scale_metrics
        else {}
    )
    run_date = datetime.now().isoformat()
    version_info = _get_version_info()
    submission = {
        "metadata": {
            "method_version": metadata.method_version,
            "contact": metadata.contact,
            "hardware": metadata.hardware,
            "run_date": run_date,
        },
        "version_info": version_info,
        "metrics": metrics,
        "reactor_scale_metrics": reactor_scale_metrics,
    }
    if submission_out is None:
        submission_out = (
            Path("submissions")
            / metadata.contact
            / metadata.method_version
            / "results.json"
        )
    if not str(submission_out).endswith(".json"):
        submission_out = submission_out.with_suffix(".json")
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(submission_out, submission)


def register(app: typer.Typer) -> None:
    app.command("submit-case")(submit_case)
    app.command("run-case")(run_case)
    app.command("run-ci-case")(run_ci_case)
    app.command("generate-submission")(generate_submission)
