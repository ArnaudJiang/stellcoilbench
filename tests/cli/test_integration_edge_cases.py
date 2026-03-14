"""Edge-case and cross-cutting integration tests for CLI commands."""

from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from tests.assert_helpers import assert_single_result
from stellcoilbench.cli import app, run_ci_case
from tests.cli.conftest import (
    _install_stub_modules,
    _make_integration_metrics,
    write_case_yaml,
)


runner = CliRunner()


class TestCLIEdgeCases:
    """Edge-case and cross-cutting integration tests."""

    def test_numpy_types_in_ci_summary(self, tmp_path, monkeypatch):
        """Verify NumpyJSONEncoder handles numpy types in CI summary output."""
        metrics = _make_integration_metrics(
            final_squared_flux=np.float64(2e-7),
            iterations_used=np.int64(5),
        )
        _install_stub_modules(monkeypatch, metrics=metrics)
        monkeypatch.setattr(
            "stellcoilbench.validate_config.validate_ci_case",
            lambda *a, **kw: [],
        )
        monkeypatch.setattr(
            "stellcoilbench.cli_helpers._detect_github_username", lambda: "ci"
        )
        monkeypatch.setattr(
            "stellcoilbench.cli._zip_submission_directory",
            lambda p: p / "all_files.zip",
        )

        ci_case = {
            "case_id": "numpy_test",
            "random_seed": None,
            "tags": [],
            "parent_ids": [],
            "resource": {"timeout_minutes": 5},
            "case_config": {
                "description": "numpy test",
                "surface_params": {"surface": "input.TestSurface"},
                "coils_params": {"ncoils": 2},
                "optimizer_params": {"max_iterations": 1},
            },
        }
        case_json = tmp_path / "case.json"
        case_json.write_text(json.dumps(ci_case))

        run_ci_case(
            case_file=case_json,
            output_dir=tmp_path / "done",
            policy_file=None,
        )

        summary_path = tmp_path / "done" / "numpy_test" / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["success"] is True

    def test_run_case_via_cli_runner(self, tmp_path, monkeypatch):
        """Exercise run-case through the Typer CLI runner."""
        _install_stub_modules(monkeypatch)
        monkeypatch.setattr(
            "stellcoilbench.cli_helpers._detect_github_username", lambda: "runner_user"
        )

        case_yaml = tmp_path / "case.yaml"
        write_case_yaml(case_yaml)

        submissions = tmp_path / "subs"
        result = runner.invoke(
            app,
            [
                "run-case",
                str(case_yaml),
                "--submissions-dir",
                str(submissions),
            ],
        )
        assert result.exit_code == 0
        assert_single_result(submissions)
