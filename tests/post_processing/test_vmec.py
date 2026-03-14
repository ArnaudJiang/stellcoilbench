"""Tests for VMEC equilibrium, Boozer surface, iota, quasisymmetry, run_post_processing, and timing."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from stellcoilbench.post_processing import (
    clear_timing_results,
    get_timing_results,
    plot_boozer_surface,
    plot_iota_profile,
    plot_quasisymmetry_profile,
    run_post_processing,
    run_vmec_equilibrium,
    timed_section,
)


class TestRunVmecEquilibrium:
    """VMEC mock smoke test."""

    def test_run_vmec_equilibrium_success(self, tmp_path: Path) -> None:
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        from simsopt.geo import SurfaceRZFourier

        mock_surface = Mock(spec=SurfaceRZFourier)
        mock_equil = Mock()
        mock_equil.as_dict.return_value = {"iota": [0.5, 1.0]}
        mock_equil.wout.iotas = np.array([0.5, 1.0])
        template_file = tmp_path / "input.template"
        template_file.write_text("dummy")

        with patch("stellcoilbench.post_processing._vmec.Vmec") as mock_vmec_class:
            mock_vmec_class.return_value = mock_equil
            mock_equil.run.return_value = None
            with patch(
                "stellcoilbench.post_processing._vmec._find_vmec_template_input",
                return_value=template_file,
            ):
                result = run_vmec_equilibrium(mock_surface, tmp_path)
                assert result == mock_equil
                mock_equil.run.assert_called_once()


class TestPlotBoozerSurface:
    """Smoke test for plot_boozer_surface."""

    def test_plot_boozer_surface_2x2_grid(self, tmp_path: Path) -> None:
        pytest.importorskip("booz_xform", reason="booz_xform not available")
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")

        mock_equil = Mock()
        mock_equil.wout.iotas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        mock_equil.output_file = str(tmp_path / "wout.test.nc")
        mock_b2 = Mock()
        mock_b2.wout.iotas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        mock_b2.read_wout = Mock()
        mock_b2.run = Mock()

        mock_bx_module = Mock()
        mock_bx_module.Booz_xform = Mock(return_value=mock_b2)
        mock_bx_module.surfplot = Mock()

        output_path = tmp_path / "boozer_plot.pdf"
        with patch(
            "stellcoilbench.post_processing._boozer_plots.booz_xform_mod",
            mock_bx_module,
        ):
            plot_boozer_surface(mock_equil, output_path)
        assert mock_bx_module.surfplot.call_count == 4
        assert output_path.exists()


class TestPlotIotaProfile:
    """Smoke test for plot_iota_profile."""

    def test_plot_iota_profile_success(self, tmp_path: Path) -> None:
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        mock_equil = Mock()
        mock_equil.wout.iotas = np.array([0.0, 0.5, 1.0, 1.5])
        mock_equil.ds = 0.01
        output_path = tmp_path / "iota_profile.pdf"
        plot_iota_profile(mock_equil, output_path, sign=1, equil_original=None)
        assert output_path.exists()


class TestPlotQuasisymmetryProfile:
    """Smoke test for plot_quasisymmetry_profile."""

    def test_plot_quasisymmetry_profile_success(self, tmp_path: Path) -> None:
        mock_qs_profile = np.array([0.001, 0.002, 0.003])
        mock_radii = np.array([0.0, 0.5, 1.0])
        output_path = tmp_path / "qs_profile.pdf"
        plot_quasisymmetry_profile(mock_qs_profile, mock_radii, output_path)
        assert output_path.exists()


class TestRunPostProcessing:
    """Smoke tests for run_post_processing."""

    def test_run_post_processing_no_vmec_no_plots(self, tmp_path: Path) -> None:
        pytest.importorskip("simsopt.geo", reason="simsopt not available")
        from simsopt.field import BiotSavart
        from simsopt.geo import SurfaceRZFourier

        coils_json = tmp_path / "coils.json"
        coils_json.write_text(json.dumps({"simsopt_version": "1.0", "coils": []}))
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text(
            """
surface_params:
  surface: input.test
coils_params:
  ncoils: 2
  order: 2
optimizer_params:
  algorithm: l-bfgs
"""
        )
        plasma_dir = tmp_path / "plasma_surfaces"
        plasma_dir.mkdir()
        (plasma_dir / "input.test").write_text("&INDATA\nNFP=2\n/")

        mock_surface = Mock(spec=SurfaceRZFourier)
        gamma_array = np.random.rand(10, 10, 3)
        normal_array = np.random.rand(10, 10, 3)
        mock_surface.gamma = lambda: gamma_array
        mock_surface.normal = lambda: normal_array
        mock_surface.unitnormal = lambda: normal_array
        mock_surface.quadpoints_phi = np.linspace(0, 1, 10)
        mock_surface.quadpoints_theta = np.linspace(0, 1, 10)

        mock_bfield = Mock(spec=BiotSavart)
        mock_bfield.B.return_value = np.random.rand(100, 3)
        mock_bfield.AbsB.return_value = np.random.rand(100) + 0.1

        with patch(
            "stellcoilbench.post_processing.load_coils_and_surface"
        ) as mock_load:
            mock_load.return_value = (mock_bfield, mock_surface)
            with patch(
                "stellcoilbench.post_processing.compute_qfm_surface"
            ) as mock_qfm:
                mock_qfm.return_value = mock_surface
                results = run_post_processing(
                    coils_json_path=coils_json,
                    output_dir=tmp_path / "output",
                    case_yaml_path=case_yaml,
                    plasma_surfaces_dir=plasma_dir,
                    run_vmec=False,
                    plot_boozer=False,
                    plot_poincare=False,
                )
                assert results is not None


class TestTimingUtilities:
    """Tests for timing utilities."""

    def test_timed_section(self) -> None:
        clear_timing_results()
        with timed_section("test_section", print_time=False):
            pass
        results = get_timing_results()
        assert "test_section" in results
        assert results["test_section"] >= 0
