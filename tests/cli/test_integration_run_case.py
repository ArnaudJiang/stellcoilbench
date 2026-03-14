"""Integration tests for run_case and run_ci_case CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from tests.assert_helpers import assert_single_result
from stellcoilbench.cli import run_case, run_ci_case
from tests.cli.conftest import _install_stub_modules, _make_integration_metrics


# ============================================================================
# Task 8.1a: run_case command flow
# ============================================================================


class TestRunCaseIntegration:
    """Integration tests for the run_case CLI command."""

    def test_run_case_full_pipeline(self, run_case_integration_env):
        """Verify run_case loads config, dispatches optimisation, and writes results.json."""
        case_yaml, submissions_dir = run_case_integration_env(
            metrics=_make_integration_metrics(final_squared_flux=3e-7),
        )

        run_case(
            case_path=case_yaml,
            submissions_dir=submissions_dir,
            results_out=None,
        )

        results_path = assert_single_result(submissions_dir)
        data = json.loads(results_path.read_text())
        assert "metrics" in data
        assert "version_info" in data
        assert "reactor_scale_metrics" in data
        assert data["metrics"]["final_squared_flux"] == 3e-7

    def test_run_case_custom_results_out(self, tmp_path, run_case_integration_env):
        """Verify run_case writes to a custom results_out path."""
        case_yaml, _ = run_case_integration_env(username="bob")

        custom_out = tmp_path / "custom_results.json"
        run_case(
            case_path=case_yaml,
            submissions_dir=tmp_path / "subs",
            results_out=custom_out,
        )
        assert custom_out.exists()
        data = json.loads(custom_out.read_text())
        assert "metrics" in data

    def test_run_case_missing_case_file(self, tmp_path, monkeypatch):
        """Verify run_case fails gracefully with a missing YAML file."""
        _install_stub_modules(monkeypatch)

        def _exploding_load(_path):
            raise FileNotFoundError("case.yaml not found")

        monkeypatch.setattr("stellcoilbench.cli.submit_run.load_case", _exploding_load)

        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            run_case(
                case_path=missing,
                submissions_dir=tmp_path / "subs",
                results_out=None,
            )

    def test_run_case_directory_structure(self, run_case_integration_env):
        """Verify run_case creates the correct directory hierarchy."""
        case_yaml, submissions_dir = run_case_integration_env(
            surface="wout.QA_Surface",
            username="carol",
        )
        run_case(
            case_path=case_yaml,
            submissions_dir=submissions_dir,
            results_out=None,
        )

        results_path = assert_single_result(submissions_dir)
        results_path_str = str(results_path)
        assert "QA_Surface" in results_path_str
        assert "carol" in results_path_str

    def test_run_case_unknown_user_fallback(self, run_case_integration_env):
        """Verify run_case uses 'unknown_user' when GitHub username is empty."""
        case_yaml, submissions_dir = run_case_integration_env(username="")
        run_case(
            case_path=case_yaml,
            submissions_dir=submissions_dir,
            results_out=None,
        )

        results_path = assert_single_result(submissions_dir)
        assert "unknown_user" in str(results_path)


# ============================================================================
# Task 8.1b: run_ci_case command flow
# ============================================================================


class TestRunCiCaseIntegration:
    """Integration tests for the run_ci_case CLI command."""

    def _write_ci_case_json(
        self,
        path: Path,
        *,
        case_id: str = "test_ci_001",
        surface: str = "input.TestSurface",
        random_seed: int | None = 42,
    ) -> Path:
        """Write a minimal CI case JSON file and return its path."""
        ci_case = {
            "case_id": case_id,
            "random_seed": random_seed,
            "tags": ["integration-test"],
            "parent_ids": [],
            "resource": {"timeout_minutes": 5},
            "case_config": {
                "description": "CI integration test",
                "surface_params": {"surface": surface},
                "coils_params": {"ncoils": 2, "order": 2},
                "optimizer_params": {"max_iterations": 1},
            },
        }
        path.write_text(json.dumps(ci_case, indent=2))
        return path

    def test_run_ci_case_success(self, tmp_path, monkeypatch):
        """Verify run_ci_case writes summary.json on successful optimisation."""
        metrics = _make_integration_metrics(final_squared_flux=5e-8, iterations_used=10)
        _install_stub_modules(monkeypatch, metrics=metrics)

        monkeypatch.setattr(
            "stellcoilbench.cli_helpers._detect_github_username", lambda: "ci-bot"
        )
        monkeypatch.setattr(
            "stellcoilbench.cli._zip_submission_directory",
            lambda p: p / "all_files.zip",
        )

        case_json = self._write_ci_case_json(tmp_path / "case.json")
        output_dir = tmp_path / "done"

        monkeypatch.setattr(
            "stellcoilbench.validate_config.validate_ci_case",
            lambda *a, **kw: [],
        )

        run_ci_case(
            case_file=case_json,
            output_dir=output_dir,
            policy_file=None,
        )

        summary_path = output_dir / "test_ci_001" / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["success"] is True
        assert summary["case_id"] == "test_ci_001"
        assert summary["total_score"] == pytest.approx(5e-8)
        assert "metrics" in summary
        assert "case_config" in summary
        assert summary["random_seed"] == 42

    def test_run_ci_case_validation_failure(self, tmp_path, monkeypatch):
        """Verify run_ci_case handles validation errors and writes failure summary."""
        _install_stub_modules(monkeypatch)
        monkeypatch.setattr(
            "stellcoilbench.validate_config.validate_ci_case",
            lambda *a, **kw: ["max_iterations exceeds cap"],
        )

        case_json = self._write_ci_case_json(tmp_path / "bad_case.json")
        output_dir = tmp_path / "done"

        with pytest.raises(typer.Exit) as exc_info:
            run_ci_case(
                case_file=case_json,
                output_dir=output_dir,
                policy_file=None,
            )
        assert exc_info.value.exit_code == 1

        summary_path = output_dir / "test_ci_001" / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["success"] is False
        assert summary["failure_reason"] == "validation_error"

    def test_run_ci_case_optimisation_error(self, tmp_path, monkeypatch):
        """Verify run_ci_case handles optimisation exceptions gracefully."""

        def _boom(**kwargs):
            raise RuntimeError("NaN in objective function")

        monkeypatch.setattr(
            "stellcoilbench.coil_optimization.optimize_coils",
            _boom,
        )

        monkeypatch.setattr(
            "stellcoilbench.validate_config.validate_ci_case",
            lambda *a, **kw: [],
        )

        case_json = self._write_ci_case_json(tmp_path / "crash.json")
        output_dir = tmp_path / "done"

        run_ci_case(
            case_file=case_json,
            output_dir=output_dir,
            policy_file=None,
        )

        summary_path = output_dir / "test_ci_001" / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["success"] is False
        assert "NaN" in summary["failure_reason"]
        assert summary["failure_class"] == "nan_in_objective"

    def test_run_ci_case_invalid_json(self, tmp_path, monkeypatch):
        """Verify run_ci_case exits with code 1 for malformed JSON."""
        _install_stub_modules(monkeypatch)

        bad_json = tmp_path / "broken.json"
        bad_json.write_text("{not valid json!!")

        with pytest.raises(typer.Exit) as exc_info:
            run_ci_case(
                case_file=bad_json,
                output_dir=tmp_path / "done",
                policy_file=None,
            )
        assert exc_info.value.exit_code == 1

    def test_run_ci_case_summary_contains_config_hash(self, tmp_path, monkeypatch):
        """Verify summary.json includes a deterministic config_hash."""
        _install_stub_modules(monkeypatch, metrics=_make_integration_metrics())
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

        case_json = self._write_ci_case_json(tmp_path / "case.json")
        output_dir = tmp_path / "done"

        run_ci_case(
            case_file=case_json,
            output_dir=output_dir,
            policy_file=None,
        )

        summary = json.loads((output_dir / "test_ci_001" / "summary.json").read_text())
        assert "config_hash" in summary
        assert len(summary["config_hash"]) == 16

    def test_run_ci_case_with_policy(self, tmp_path, monkeypatch):
        """Verify run_ci_case passes policy to validation."""
        _install_stub_modules(monkeypatch, metrics=_make_integration_metrics())
        monkeypatch.setattr(
            "stellcoilbench.cli_helpers._detect_github_username", lambda: "ci"
        )
        monkeypatch.setattr(
            "stellcoilbench.cli._zip_submission_directory",
            lambda p: p / "all_files.zip",
        )

        captured_policy = {}

        def capture_validate(data, *, policy=None, file_path=None):
            captured_policy["policy"] = policy
            return []

        monkeypatch.setattr(
            "stellcoilbench.validate_config.validate_ci_case",
            capture_validate,
        )

        policy_yaml = tmp_path / "policy.yaml"
        policy_yaml.write_text("resource_caps:\n  max_iterations: 100\n")

        case_json = self._write_ci_case_json(tmp_path / "case.json")
        run_ci_case(
            case_file=case_json,
            output_dir=tmp_path / "done",
            policy_file=policy_yaml,
        )

        assert captured_policy["policy"] is not None
        assert "resource_caps" in captured_policy["policy"]
