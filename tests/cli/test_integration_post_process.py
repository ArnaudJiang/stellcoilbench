"""Integration tests for the post-process CLI command."""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

from stellcoilbench.cli import app
from tests.cli.conftest import write_case_yaml


runner = CliRunner()


class TestPostProcessIntegration:
    """Integration tests for the post-process CLI command."""

    def test_post_process_dispatches_correctly(
        self, post_process_capture, monkeypatch, tmp_path
    ):
        """Verify post-process dispatches to run_post_processing with correct args."""
        captured, coils_path = post_process_capture
        fake_results = {"BdotN": 1.5e-4, "quasisymmetry_average": 0.02}

        def capture_and_return(**kw):
            captured.update(kw)
            return fake_results

        pp_mod = sys.modules["stellcoilbench.post_processing"]
        monkeypatch.setattr(pp_mod, "run_post_processing", capture_and_return)
        output_dir = tmp_path / "pp_out"
        result = runner.invoke(
            app,
            [
                "post-process",
                str(coils_path),
                "--output-dir",
                str(output_dir),
                "--run-vmec",
                "--plot-poincare",
                "--nfieldlines",
                "10",
            ],
        )
        assert result.exit_code == 0
        assert "Post-processing complete!" in result.output
        assert "quasisymmetry" in result.output

        assert captured["run_vmec"] is True
        assert captured["plot_poincare"] is True
        assert captured["nfieldlines"] == 10
        assert captured["run_simple"] is False

    def test_post_process_all_flags_via_cli(self, post_process_capture, tmp_path):
        """Verify --all-post-processing sets every flag to True."""
        captured, coils_path = post_process_capture
        result = runner.invoke(
            app,
            [
                "post-process",
                str(coils_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--all-post-processing",
            ],
        )
        assert result.exit_code == 0
        for flag in [
            "run_vmec",
            "run_simple",
            "plot_poincare",
            "plot_boozer",
            "plot_finite_build",
            "run_structural",
        ]:
            assert captured[flag] is True, f"{flag} should be True"

    def test_post_process_error_handling(
        self, post_process_capture, monkeypatch, tmp_path
    ):
        """Verify post-process command exits with code 1 on RuntimeError."""
        captured, coils_path = post_process_capture

        def boom(**kwargs):
            raise RuntimeError("VMEC failed to converge")

        pp_mod = sys.modules["stellcoilbench.post_processing"]
        monkeypatch.setattr(pp_mod, "run_post_processing", boom)
        result = runner.invoke(
            app,
            [
                "post-process",
                str(coils_path),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 1

    def test_post_process_finite_build_params(self, post_process_capture, tmp_path):
        """Verify finite-build width/height are forwarded correctly."""
        captured, coils_path = post_process_capture
        result = runner.invoke(
            app,
            [
                "post-process",
                str(coils_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--plot-finite-build",
                "--finite-build-width",
                "0.05",
                "--finite-build-height",
                "0.03",
            ],
        )
        assert result.exit_code == 0
        assert captured["plot_finite_build"] is True
        assert captured["finite_build_width"] == pytest.approx(0.05)
        assert captured["finite_build_height"] == pytest.approx(0.03)

    def test_post_process_structural_params(self, post_process_capture, tmp_path):
        """Verify structural analysis params are forwarded correctly."""
        captured, coils_path = post_process_capture
        result = runner.invoke(
            app,
            [
                "post-process",
                str(coils_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--run-structural",
                "--structural-E",
                "1e11",
                "--structural-nu",
                "0.3",
            ],
        )
        assert result.exit_code == 0
        assert captured["run_structural"] is True
        assert captured["structural_E"] == pytest.approx(1e11)
        assert captured["structural_nu"] == pytest.approx(0.3)

    def test_post_process_case_yaml_forwarded(self, post_process_capture, tmp_path):
        """Verify --case-yaml is forwarded to run_post_processing."""
        captured, coils_path = post_process_capture
        case_yaml = tmp_path / "case.yaml"
        write_case_yaml(case_yaml)
        result = runner.invoke(
            app,
            [
                "post-process",
                str(coils_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--case-yaml",
                str(case_yaml),
            ],
        )
        assert result.exit_code == 0
        assert captured["case_yaml_path"] == case_yaml

    def test_post_process_helicity_and_ns(self, post_process_capture, tmp_path):
        """Verify helicity_m, helicity_n, and ns are forwarded correctly."""
        captured, coils_path = post_process_capture
        result = runner.invoke(
            app,
            [
                "post-process",
                str(coils_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--helicity-m",
                "2",
                "--helicity-n",
                "1",
                "--ns",
                "100",
            ],
        )
        assert result.exit_code == 0
        assert captured["helicity_m"] == 2
        assert captured["helicity_n"] == 1
        assert captured["ns"] == 100
