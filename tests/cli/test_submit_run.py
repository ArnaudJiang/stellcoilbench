"""Tests for submit-case, run-case, generate-submission, and main CLI."""

import json

import pytest
import typer

from tests.assert_helpers import assert_single_result
from tests.cli.conftest import (
    _install_stub_modules,
    _make_case_config,
    _make_results_dict,
    write_case_yaml,
)
from stellcoilbench.cli import (
    generate_submission,
    run_case,
    submit_case,
)


def test_run_case_writes_results_json(tmp_path, monkeypatch):
    _install_stub_modules(
        monkeypatch,
        metrics={"final_normalized_squared_flux": 0.123},
        surface="input.TestSurface",
    )
    case_path = tmp_path / "case.yaml"
    write_case_yaml(case_path)
    submissions_dir = tmp_path / "submissions"
    monkeypatch.setattr(
        "stellcoilbench.cli_helpers._detect_github_username", lambda: "testuser"
    )

    run_case(case_path=case_path, submissions_dir=submissions_dir, results_out=None)

    results_path = assert_single_result(submissions_dir)
    assert "TestSurface" in str(results_path)
    assert "testuser" in str(results_path)
    data = json.loads(results_path.read_text())
    assert "metrics" in data
    assert "version_info" in data
    assert "reactor_scale_metrics" in data
    assert data["metrics"]["final_normalized_squared_flux"] == 0.123


def test_run_case_ensures_json_extension(tmp_path, monkeypatch):
    _install_stub_modules(monkeypatch, surface="input.TestSurface")
    case_path = tmp_path / "case.yaml"
    write_case_yaml(case_path)
    submissions_dir = tmp_path / "submissions"
    results_out = tmp_path / "metrics.txt"
    monkeypatch.setattr(
        "stellcoilbench.cli_helpers._detect_github_username", lambda: "testuser"
    )

    run_case(
        case_path=case_path, submissions_dir=submissions_dir, results_out=results_out
    )

    assert not results_out.exists()
    assert (tmp_path / "metrics.json").exists()


def test_generate_submission_writes_results(tmp_path, monkeypatch):
    _install_stub_modules(monkeypatch)
    monkeypatch.chdir(tmp_path)

    case_dir = tmp_path / "case"
    case_dir.mkdir()
    write_case_yaml(case_dir / "case.yaml")
    (case_dir / "coils.json").write_text("{}")

    metadata_yaml = tmp_path / "metadata.yaml"
    metadata_yaml.write_text("method_version: v1\ncontact: demo\nhardware: CPU\n")

    generate_submission(
        case_path=case_dir,
        metadata_path=metadata_yaml,
        coils_path=None,
        submission_out=None,
    )

    submission_path = tmp_path / "submissions" / "demo" / "v1" / "results.json"
    assert submission_path.exists()
    data = json.loads(submission_path.read_text())
    assert data["metadata"]["contact"] == "demo"
    assert "metrics" in data
    assert data["metrics"]["chi2_Bn"] == 0.001


def test_generate_submission_missing_coils_file(tmp_path, monkeypatch):
    _install_stub_modules(monkeypatch)
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    write_case_yaml(case_dir / "case.yaml")
    metadata_yaml = tmp_path / "metadata.yaml"
    metadata_yaml.write_text("contact: demo\nmethod_version: v1\n")

    with pytest.raises(typer.Exit):
        generate_submission(
            case_path=case_dir,
            metadata_path=metadata_yaml,
            coils_path=None,
            submission_out=None,
        )


def test_submit_case_creates_submission(tmp_path, monkeypatch):
    _install_stub_modules(monkeypatch, metrics={"final_normalized_squared_flux": 0.004})
    monkeypatch.chdir(tmp_path)

    case_path = tmp_path / "case.yaml"
    write_case_yaml(case_path)

    monkeypatch.setattr(
        "stellcoilbench.cli_helpers._detect_github_username", lambda: "user1"
    )
    monkeypatch.setattr(
        "stellcoilbench.cli_helpers._detect_hardware", lambda: "CPU: Test"
    )
    monkeypatch.setattr(
        "stellcoilbench.submission_packaging._zip_submission_directory",
        lambda path: path.with_suffix(".zip"),
    )

    submissions_dir = tmp_path / "submissions"
    submit_case(
        case_path=case_path,
        submissions_dir=submissions_dir,
    )

    results_path = assert_single_result(submissions_dir)
    results_data = json.loads(results_path.read_text())
    assert results_data["metadata"]["contact"] == "user1"

    case_copies = list(submissions_dir.rglob("case.yaml"))
    assert len(case_copies) == 1
    case_data = case_copies[0].read_text()
    assert "source_case_file: case.yaml" in case_data


def test_submit_case_unknown_user_and_hardware(tmp_path, monkeypatch):
    _install_stub_modules(monkeypatch, surface="wout.TestSurface")
    monkeypatch.chdir(tmp_path)

    case_path = tmp_path / "case.yaml"
    write_case_yaml(case_path, surface="wout.TestSurface")

    monkeypatch.setattr(
        "stellcoilbench.cli_helpers._detect_github_username", lambda: ""
    )
    monkeypatch.setattr("stellcoilbench.cli_helpers._detect_hardware", lambda: "")
    monkeypatch.setattr(
        "stellcoilbench.submission_packaging._zip_submission_directory",
        lambda path: path.with_suffix(".zip"),
    )

    submissions_dir = tmp_path / "submissions"
    submit_case(
        case_path=case_path,
        submissions_dir=submissions_dir,
    )

    results_path = assert_single_result(submissions_dir)
    assert "unknown_user" in str(results_path.parent)
    assert "TestSurface" in str(results_path.parent.parent)


def test_main_calls_app(monkeypatch):
    import stellcoilbench.cli as cli_module

    called = {"count": 0}

    def fake_app():
        called["count"] += 1

    monkeypatch.setattr(cli_module, "app", fake_app)
    cli_module.main()
    assert called["count"] == 1


def test_run_case_wout_surface_prefix(tmp_path, monkeypatch):
    """run_case strips 'wout.' prefix from surface name for directory."""
    _install_stub_modules(
        monkeypatch,
        metrics={"final_normalized_squared_flux": 0.1},
        surface="wout.TestSurface",
    )
    case_path = tmp_path / "case.yaml"
    write_case_yaml(case_path, surface="wout.TestSurface")
    submissions_dir = tmp_path / "submissions"
    monkeypatch.setattr(
        "stellcoilbench.cli_helpers._detect_github_username", lambda: "testuser"
    )

    run_case(case_path=case_path, submissions_dir=submissions_dir, results_out=None)

    results_path = assert_single_result(submissions_dir)
    assert "TestSurface" in str(results_path)
    assert "wout." not in str(results_path)


def test_run_case_surface_with_extension(tmp_path, monkeypatch):
    """run_case strips file extensions like .focus from surface name."""
    _install_stub_modules(
        monkeypatch,
        metrics={"final_normalized_squared_flux": 0.05},
        surface="c09r00_NCSX.focus",
    )
    case_path = tmp_path / "case.yaml"
    write_case_yaml(case_path, surface="c09r00_NCSX.focus")
    submissions_dir = tmp_path / "submissions"
    monkeypatch.setattr(
        "stellcoilbench.cli_helpers._detect_github_username", lambda: "testuser"
    )

    run_case(case_path=case_path, submissions_dir=submissions_dir, results_out=None)

    results_path = assert_single_result(submissions_dir)
    assert "c09r00_NCSX" in str(results_path)
    assert ".focus" not in str(results_path)


def test_submit_case_surface_with_extension(tmp_path, monkeypatch):
    """submit_case strips file extensions from surface name in directory path."""
    _install_stub_modules(
        monkeypatch,
        metrics={"final_normalized_squared_flux": 0.004},
        surface="plasma.focus",
    )
    monkeypatch.chdir(tmp_path)

    case_path = tmp_path / "case.yaml"
    write_case_yaml(case_path, surface="plasma.focus")

    monkeypatch.setattr(
        "stellcoilbench.cli_helpers._detect_github_username", lambda: "user1"
    )
    monkeypatch.setattr(
        "stellcoilbench.cli_helpers._detect_hardware", lambda: "CPU: Test"
    )
    monkeypatch.setattr(
        "stellcoilbench.submission_packaging._zip_submission_directory",
        lambda path: path.with_suffix(".zip"),
    )

    submissions_dir = tmp_path / "submissions"
    submit_case(
        case_path=case_path,
        submissions_dir=submissions_dir,
    )

    results_path = assert_single_result(submissions_dir)
    assert "/plasma/" in str(results_path)


def test_submit_case_relative_path_fallback(tmp_path, monkeypatch):
    """submit_case handles ValueError when computing relative path for case.yaml."""
    _install_stub_modules(
        monkeypatch,
        metrics={"final_normalized_squared_flux": 0.004},
        surface="input.Test",
    )
    other_root = tmp_path / "other_root"
    other_root.mkdir()
    monkeypatch.chdir(other_root)

    case_path = tmp_path / "case.yaml"
    write_case_yaml(case_path, surface="input.Test")

    monkeypatch.setattr(
        "stellcoilbench.cli_helpers._detect_github_username", lambda: "user1"
    )
    monkeypatch.setattr(
        "stellcoilbench.cli_helpers._detect_hardware", lambda: "CPU: Test"
    )
    monkeypatch.setattr(
        "stellcoilbench.submission_packaging._zip_submission_directory",
        lambda path: path.with_suffix(".zip"),
    )

    submissions_dir = tmp_path / "submissions"
    submit_case(
        case_path=case_path,
        submissions_dir=submissions_dir,
    )

    assert_single_result(submissions_dir)


class TestWriteAutopilotSubmission:
    """Tests for _write_autopilot_submission."""

    def test_basic_submission_created(self, tmp_path):
        from stellcoilbench.cli import _write_autopilot_submission

        case_output = tmp_path / "case_output"
        case_output.mkdir()
        (case_output / "poincare_plot.png").write_bytes(b"png")

        _write_autopilot_submission(
            case_id="test_001",
            results_dict=_make_results_dict(),
            case_cfg=None,
            case_config_dict=_make_case_config(),
            walltime=10.0,
            repo_root=tmp_path,
            case_output_dir=case_output,
        )

        sub_dir = (
            tmp_path / "submissions" / "LandremanPaul2021_QA" / "auto" / "test_001"
        )
        assert (sub_dir / "results.json").exists()
        assert (sub_dir / "case.yaml").exists()
        assert (sub_dir / "poincare_plot.png").exists()

    def test_fc_plots_copied_from_order_dirs(self, tmp_path):
        from stellcoilbench.cli import _write_autopilot_submission

        case_output = tmp_path / "case_output"
        case_output.mkdir()
        (case_output / "poincare_plot.png").write_bytes(b"pp")
        for order in [4, 8, 16]:
            d = case_output / f"order_{order}"
            d.mkdir()
            (d / "bn_error_3d_plot.pdf").write_bytes(f"bn_{order}".encode())
            (d / "bn_error_3d_plot_initial.pdf").write_bytes(f"bni_{order}".encode())
            (d / "biot_savart_optimized.json").write_text("{}")

        _write_autopilot_submission(
            case_id="fc_test",
            results_dict=_make_results_dict(),
            case_cfg=None,
            case_config_dict=_make_case_config(fourier_continuation=True),
            walltime=10.0,
            repo_root=tmp_path,
            case_output_dir=case_output,
        )

        sub_dir = tmp_path / "submissions" / "LandremanPaul2021_QA" / "auto" / "fc_test"
        assert (sub_dir / "bn_error_3d_plot.pdf").exists()
        assert (sub_dir / "bn_error_3d_plot.pdf").read_bytes() == b"bn_16"
        assert (sub_dir / "poincare_plot.png").exists()
        assert (sub_dir / "order_4" / "bn_error_3d_plot.pdf").exists()
        assert (sub_dir / "order_8" / "bn_error_3d_plot.pdf").exists()
        assert (sub_dir / "order_16" / "bn_error_3d_plot.pdf").exists()

    def test_fc_no_order_dirs_falls_back(self, tmp_path):
        from stellcoilbench.cli import _write_autopilot_submission

        case_output = tmp_path / "case_output"
        case_output.mkdir()
        (case_output / "bn_error_3d_plot.pdf").write_bytes(b"top")

        _write_autopilot_submission(
            case_id="fc_fallback",
            results_dict=_make_results_dict(),
            case_cfg=None,
            case_config_dict=_make_case_config(fourier_continuation=True),
            walltime=10.0,
            repo_root=tmp_path,
            case_output_dir=case_output,
        )

        sub_dir = (
            tmp_path / "submissions" / "LandremanPaul2021_QA" / "auto" / "fc_fallback"
        )
        assert (sub_dir / "bn_error_3d_plot.pdf").read_bytes() == b"top"


class TestAppendFailureToAutopilotFailures:
    """Tests for _append_failure_to_autopilot_failures."""

    def test_append_creates_file(self, tmp_path):
        from stellcoilbench.ci_autopilot import _append_failure_to_autopilot_failures

        failures_path = tmp_path / "autopilot_failures.json"
        summary = {
            "case_id": "fail_001",
            "success": False,
            "failure_class": "timeout",
            "failure_reason": "timed out after 3600s",
            "config_hash": "abc123",
            "case_config": {"surface_params": {"surface": "input.Test"}},
            "tags": [],
            "parent_ids": [],
        }
        _append_failure_to_autopilot_failures(failures_path, summary)
        assert failures_path.exists()
        data = json.loads(failures_path.read_text())
        assert "failures" in data
        assert len(data["failures"]) == 1
        assert data["failures"][0]["case_id"] == "fail_001"
        assert data["failures"][0]["failure_class"] == "timeout"
        assert data["failures"][0]["success"] is False

    def test_appends_to_existing_file(self, tmp_path):
        from stellcoilbench.ci_autopilot import _append_failure_to_autopilot_failures

        failures_path = tmp_path / "autopilot_failures.json"
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        failures_path.write_text(
            json.dumps({"failures": [{"case_id": "old_001", "success": False}]})
        )
        _append_failure_to_autopilot_failures(
            failures_path,
            {"case_id": "new_002", "success": False, "failure_class": "nan"},
        )
        data = json.loads(failures_path.read_text())
        assert len(data["failures"]) == 2
        assert data["failures"][0]["case_id"] == "old_001"
        assert data["failures"][1]["case_id"] == "new_002"

    def test_append_trims_to_max_entries(self, tmp_path):
        from stellcoilbench.ci_autopilot import _append_failure_to_autopilot_failures

        failures_path = tmp_path / "autopilot_failures.json"
        for i in range(10):
            _append_failure_to_autopilot_failures(
                failures_path,
                {"case_id": f"fail_{i:03d}", "success": False, "failure_class": "x"},
                max_entries=5,
            )
        data = json.loads(failures_path.read_text())
        assert len(data["failures"]) == 5
        assert data["failures"][0]["case_id"] == "fail_005"
        assert data["failures"][-1]["case_id"] == "fail_009"
