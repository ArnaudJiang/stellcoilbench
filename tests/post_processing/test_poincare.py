"""Tests for trace_fieldlines and run_simple_particle_tracing."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from stellcoilbench.post_processing import (
    TRACING_AVAILABLE,
    run_simple_particle_tracing,
    trace_fieldlines,
)


class TestTraceFieldlines:
    """Smoke tests for trace_fieldlines."""

    def test_trace_fieldlines_not_available(self) -> None:
        from simsopt.field import BiotSavart
        from simsopt.geo import SurfaceRZFourier

        mock_bfield = Mock(spec=BiotSavart)
        mock_surface = Mock(spec=SurfaceRZFourier)
        with patch(
            "stellcoilbench.post_processing._fieldlines.TRACING_AVAILABLE",
            False,
        ):
            with pytest.raises(ImportError, match="Fieldline tracing"):
                trace_fieldlines(mock_bfield, mock_surface, Path("/tmp/test.pdf"))

    @pytest.mark.skipif(not TRACING_AVAILABLE, reason="Tracing not available")
    def test_trace_fieldlines_with_mock(self, tmp_path: Path) -> None:
        pytest.importorskip("simsopt.field.tracing")
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import BiotSavart

        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        mock_bfield = Mock(spec=BiotSavart)
        mock_bfield.B.return_value = np.random.rand(100, 3)
        mock_bfield.AbsB.return_value = np.random.rand(100)
        mock_fieldlines_tys = [np.random.rand(100, 3) for _ in range(5)]
        mock_fieldlines_phi_hits = [np.random.rand(50, 2) for _ in range(5)]

        with patch(
            "stellcoilbench.post_processing._fieldlines.compute_fieldlines"
        ) as mock_compute:
            mock_compute.return_value = (
                mock_fieldlines_tys,
                mock_fieldlines_phi_hits,
            )
            with patch("stellcoilbench.post_processing._fieldlines.plot_poincare_data"):
                output_path = tmp_path / "poincare.png"
                result = trace_fieldlines(
                    mock_bfield,
                    surface,
                    output_path,
                    nfieldlines=5,
                    tmax=10000,
                    use_interpolated_field=False,
                )
                assert result is not None
                assert "poincare_plot_path" in result


class TestRunSimpleParticleTracing:
    """Smoke tests for run_simple_particle_tracing."""

    def test_simple_success(self, tmp_path: Path) -> None:
        mock_equil = Mock()
        vmec_output = tmp_path / "wout_test.nc"
        vmec_output.write_text("dummy")
        mock_equil.output_file = str(vmec_output)
        simple_x = tmp_path / "simple.x"
        simple_x.write_text("#!/bin/bash\necho done\n")
        simple_x.chmod(0o755)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "confined_fraction.dat").write_text(
            "0.0 0.5 0.3 1000\n0.2 0.4 0.2 1000\n"
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="Success", stderr="")
            results = run_simple_particle_tracing(
                mock_equil,
                output_dir,
                simple_executable_path=simple_x,
            )
            assert results is not None
            assert "simple_output_dir" in results
            assert "confined_fraction_file" in results

    def test_simple_executable_not_found(self, tmp_path: Path) -> None:
        mock_equil = Mock()
        vmec_output = tmp_path / "wout_test.nc"
        vmec_output.write_text("dummy")
        mock_equil.output_file = str(vmec_output)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        results = run_simple_particle_tracing(
            mock_equil,
            output_dir,
            simple_executable_path=None,
        )
        assert results == {}


def test_tracing_available_flag_exists() -> None:
    """TRACING_AVAILABLE is a bool."""
    assert isinstance(TRACING_AVAILABLE, bool)
