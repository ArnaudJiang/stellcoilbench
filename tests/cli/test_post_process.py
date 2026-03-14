"""Tests for post-process CLI command."""

import sys
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from stellcoilbench.cli import app
from tests.cli.conftest import write_case_yaml
from tests.conftest import minimal_coils_json


def test_apply_all_post_processing_flags():
    """Test apply_all_post_processing_flags branches."""
    from stellcoilbench.cli._shared import apply_all_post_processing_flags

    flags = dict(
        run_vmec=False,
        run_simple=False,
        plot_poincare=False,
        plot_boozer=False,
        plot_finite_build=False,
        run_structural=False,
        compute_shape_gradient=False,
    )
    assert apply_all_post_processing_flags(True, **flags) == (True,) * 7
    assert apply_all_post_processing_flags(False, **flags) == (False,) * 7


def test_run_sensitivity_if_configured_early_return(tmp_path):
    """Test run_sensitivity_if_configured returns None when run_sensitivity=False."""
    from stellcoilbench.cli._shared import run_sensitivity_if_configured

    out = run_sensitivity_if_configured(
        run_sensitivity=False,
        coils_out_path=tmp_path / "coils.json",
        case_path=tmp_path,
        correlation_length=1.0,
        n_samples=10,
        output_dir=tmp_path,
        n_vtk=3,
        metrics={},
    )
    assert out is None


def test_run_sensitivity_if_configured_exception(tmp_path):
    """Test run_sensitivity_if_configured returns None on exception."""
    from stellcoilbench.cli._shared import run_sensitivity_if_configured

    minimal_coils_json(tmp_path)
    write_case_yaml(tmp_path / "case.yaml", surface="input.test")
    with patch(
        "stellcoilbench.sensitivity.run_sensitivity_analysis",
        side_effect=ValueError("x"),
    ):
        out = run_sensitivity_if_configured(
            run_sensitivity=True,
            coils_out_path=tmp_path / "coils.json",
            case_path=tmp_path,
            correlation_length=1.0,
            n_samples=10,
            output_dir=tmp_path,
            n_vtk=3,
            metrics={},
        )
    assert out is None


@pytest.mark.parametrize(
    "fake_results,expect_quasisymmetry_in_output",
    [
        (
            {
                "BdotN": 1.23e-4,
                "BdotN_over_B": 5.67e-5,
                "quasisymmetry_average": 0.01,
            },
            True,
        ),
        ({"some_metric": 42}, False),
    ],
)
def test_post_process_output(
    post_process_capture,
    monkeypatch,
    tmp_path,
    fake_results,
    expect_quasisymmetry_in_output,
):
    """post_process calls run_post_processing and prints results; quasisymmetry presence varies."""
    captured, coils_path = post_process_capture
    pp_mod = sys.modules["stellcoilbench.post_processing"]
    monkeypatch.setattr(pp_mod, "run_post_processing", lambda **kwargs: fake_results)
    output_dir = tmp_path / "output"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["post-process", str(coils_path), "--output-dir", str(output_dir)],
    )
    assert result.exit_code == 0
    assert "Post-processing complete!" in result.output
    assert ("quasisymmetry" in result.output) == expect_quasisymmetry_in_output


def test_post_process_exception(post_process_capture, monkeypatch, tmp_path):
    """post_process handles exceptions from run_post_processing."""
    captured, coils_path = post_process_capture

    def boom(**kwargs):
        raise RuntimeError("post-processing failed")

    pp_mod = sys.modules["stellcoilbench.post_processing"]
    monkeypatch.setattr(pp_mod, "run_post_processing", boom)
    output_dir = tmp_path / "output"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["post-process", str(coils_path), "--output-dir", str(output_dir)],
    )
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "cli_args,expected_flags",
    [
        (
            [],
            {
                "run_vmec": False,
                "run_simple": False,
                "plot_poincare": False,
                "plot_boozer": False,
                "plot_finite_build": False,
                "run_structural": False,
            },
        ),
        (
            ["--all-post-processing"],
            {
                "run_vmec": True,
                "run_simple": True,
                "plot_poincare": True,
                "plot_boozer": True,
                "plot_finite_build": True,
                "run_structural": True,
            },
        ),
    ],
)
def test_post_process_flag_defaults_and_all(
    post_process_capture, tmp_path, cli_args, expected_flags
):
    """All post-processing flags default to False; --all-post-processing enables them."""
    captured, coils_path = post_process_capture
    output_dir = tmp_path / "output"
    runner = CliRunner()
    cmd = ["post-process", str(coils_path), "--output-dir", str(output_dir)] + cli_args
    result = runner.invoke(app, cmd)
    assert result.exit_code == 0
    for flag, expected in expected_flags.items():
        assert captured[flag] is expected, f"{flag} should be {expected}"


def test_post_process_individual_overrides_still_work(post_process_capture, tmp_path):
    """Individual flags still work without --all-post-processing."""
    captured, coils_path = post_process_capture
    output_dir = tmp_path / "output"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "post-process",
            str(coils_path),
            "--output-dir",
            str(output_dir),
            "--run-vmec",
            "--plot-poincare",
        ],
    )
    assert result.exit_code == 0
    assert captured["run_vmec"] is True
    assert captured["plot_poincare"] is True
    assert captured["run_simple"] is False
    assert captured["plot_boozer"] is False
    assert captured["plot_finite_build"] is False
    assert captured["run_structural"] is False
