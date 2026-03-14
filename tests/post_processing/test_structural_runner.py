"""Tests for structural analysis, shape gradient, and finite-build runners."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stellcoilbench.post_processing._finite_build_runner import _run_finite_build_vtk
from stellcoilbench.post_processing._structural_runner import (
    _run_shape_gradient_analysis,
    _run_structural,
)


def _make_mock_surface(nfp: int = 5, stellsym: bool = False) -> MagicMock:
    """Minimal mock surface with nfp and stellsym."""
    s = MagicMock()
    s.nfp = nfp
    s.stellsym = stellsym
    return s


class TestRunStructural:
    """Tests for _run_structural."""

    def test_run_structural_success_adds_metrics(self, tmp_path: Path) -> None:
        """Successful run adds structural_metrics to results."""
        mock_bfield = MagicMock()
        mock_surface = _make_mock_surface()
        results: dict = {}

        with patch(
            "stellcoilbench.structural_analysis.run_structural_analysis",
        ) as mock_run:
            mock_run.return_value = {"max_von_mises_stress_Pa": 1e8}
            with patch(
                "stellcoilbench.post_processing._structural_runner._get_coils_from_bfield",
                return_value=[MagicMock()],
            ):
                with patch(
                    "stellcoilbench.post_processing._structural_runner.get_unique_coils",
                    return_value=[MagicMock()],
                ):
                    _run_structural(
                        bfield=mock_bfield,
                        surface=mock_surface,
                        output_dir=tmp_path,
                        results=results,
                        finite_build_width=0.35,
                        finite_build_height=0.35,
                        structural_E=None,
                        structural_nu=None,
                    )

        assert results["structural_metrics"] == {"max_von_mises_stress_Pa": 1e8}

    def test_run_structural_exception_handled(self, tmp_path: Path) -> None:
        """Exception from run_structural_analysis is caught and does not propagate."""
        mock_bfield = MagicMock()
        mock_surface = _make_mock_surface()
        results: dict = {}

        with patch(
            "stellcoilbench.structural_analysis.run_structural_analysis",
            side_effect=RuntimeError("FEM solve failed"),
        ):
            with patch(
                "stellcoilbench.post_processing._structural_runner._get_coils_from_bfield",
                return_value=[MagicMock()],
            ):
                with patch(
                    "stellcoilbench.post_processing._structural_runner.get_unique_coils",
                    return_value=[MagicMock()],
                ):
                    _run_structural(
                        bfield=mock_bfield,
                        surface=mock_surface,
                        output_dir=tmp_path,
                        results=results,
                        finite_build_width=0.35,
                        finite_build_height=0.35,
                        structural_E=None,
                        structural_nu=None,
                    )

        assert "structural_metrics" not in results

    def test_run_structural_skips_when_no_coils(self, tmp_path: Path) -> None:
        """When get_unique_coils returns [], run_structural_analysis is not called."""
        mock_bfield = MagicMock()
        mock_surface = _make_mock_surface()
        results: dict = {}

        with patch(
            "stellcoilbench.structural_analysis.run_structural_analysis",
        ) as mock_run:
            with patch(
                "stellcoilbench.post_processing._structural_runner._get_coils_from_bfield",
                return_value=[],
            ):
                with patch(
                    "stellcoilbench.post_processing._structural_runner.get_unique_coils",
                    return_value=[],
                ):
                    _run_structural(
                        bfield=mock_bfield,
                        surface=mock_surface,
                        output_dir=tmp_path,
                        results=results,
                        finite_build_width=0.35,
                        finite_build_height=0.35,
                        structural_E=None,
                        structural_nu=None,
                    )

        mock_run.assert_not_called()
        assert "structural_metrics" not in results

    @pytest.mark.parametrize("width,height", [(0.42, 0.28), (0.35, 0.35)])
    def test_run_structural_passes_width_height_when_coils_exist(
        self, tmp_path: Path, width: float, height: float
    ) -> None:
        """run_structural_analysis is called with correct finite_build width/height."""
        mock_bfield = MagicMock()
        mock_surface = _make_mock_surface()
        results: dict = {}
        mock_coil = MagicMock()

        with patch(
            "stellcoilbench.structural_analysis.run_structural_analysis",
        ) as mock_run:
            mock_run.return_value = {"max_von_mises_stress_Pa": 1e8}
            with patch(
                "stellcoilbench.post_processing._structural_runner._get_coils_from_bfield",
                return_value=[mock_coil],
            ):
                with patch(
                    "stellcoilbench.post_processing._structural_runner.get_unique_coils",
                    return_value=[mock_coil],
                ):
                    _run_structural(
                        bfield=mock_bfield,
                        surface=mock_surface,
                        output_dir=tmp_path,
                        results=results,
                        finite_build_width=width,
                        finite_build_height=height,
                        structural_E=None,
                        structural_nu=None,
                    )

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["width"] == width
        assert call_kwargs["height"] == height

    def test_run_structural_passes_export_full_coil_set_nfp_stellsym(
        self, tmp_path: Path
    ) -> None:
        """run_structural_analysis receives export_full_coil_set, nfp, stellsym."""
        mock_bfield = MagicMock()
        mock_surface = _make_mock_surface(nfp=5, stellsym=True)
        results: dict = {}

        with patch(
            "stellcoilbench.structural_analysis.run_structural_analysis",
        ) as mock_run:
            mock_run.return_value = {"max_von_mises_stress_Pa": 1e8}
            with patch(
                "stellcoilbench.post_processing._structural_runner._get_coils_from_bfield",
                return_value=[MagicMock()],
            ):
                with patch(
                    "stellcoilbench.post_processing._structural_runner.get_unique_coils",
                    return_value=[MagicMock()],
                ):
                    _run_structural(
                        bfield=mock_bfield,
                        surface=mock_surface,
                        output_dir=tmp_path,
                        results=results,
                        finite_build_width=0.35,
                        finite_build_height=0.35,
                        structural_E=None,
                        structural_nu=None,
                        export_full_coil_set=True,
                    )

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["export_full_coil_set"] is True
        assert call_kwargs["nfp"] == 5
        assert call_kwargs["stellsym"] is True


def _make_mock_coil(gamma_return: list | None = None, gamma_raises: Exception | None = None) -> MagicMock:
    """Mock coil with optional curve.gamma() return or raise."""
    c = MagicMock()
    if gamma_raises is not None:
        c.curve.gamma.side_effect = gamma_raises
    else:
        c.curve.gamma.return_value = gamma_return or [[0.0, 0.0, 0.0]]
    return c


class TestRunFiniteBuildVtk:
    """Tests for _run_finite_build_vtk from _finite_build_runner."""

    @pytest.mark.parametrize(
        "width,height",
        [(0.35, 0.28), (0.35, 0.35), (None, None)],
    )
    def test_finite_build_vtk_called_with_width_height_when_coils_exist(
        self, tmp_path: Path, width: float | None, height: float | None
    ) -> None:
        """When coils exist, finite_build_coils_to_vtk is called with correct width/height."""
        mock_bfield = MagicMock()
        mock_surface = _make_mock_surface()
        results: dict = {}
        mock_coil = _make_mock_coil()

        with patch(
            "stellcoilbench.post_processing._finite_build_runner._get_coils_from_bfield",
            return_value=[mock_coil],
        ):
            with patch(
                "stellcoilbench.post_processing._finite_build_runner.get_unique_coils",
                return_value=[mock_coil],
            ):
                with patch(
                    "stellcoilbench.finite_build.finite_build_coils_to_vtk",
                ) as mock_fb_vtk:
                    mock_fb_vtk.return_value = tmp_path / "coils.vtk"
                    _run_finite_build_vtk(
                        bfield=mock_bfield,
                        surface=mock_surface,
                        output_dir=tmp_path,
                        width=width,
                        height=height,
                        results=results,
                    )

        mock_fb_vtk.assert_called_once()
        call_kwargs = mock_fb_vtk.call_args[1]
        assert call_kwargs["width"] is width
        assert call_kwargs["height"] is height
        assert call_kwargs["min_mesh_size"] == 0.02
        assert call_kwargs["max_mesh_size"] == 0.02
        assert "finite_build_vtk_path" in results

    def test_finite_build_vtk_skips_when_no_coils(self, tmp_path: Path) -> None:
        """When get_unique_coils returns [], finite_build_coils_to_vtk is not called."""
        mock_bfield = MagicMock()
        mock_surface = _make_mock_surface()
        results: dict = {}

        with patch(
            "stellcoilbench.post_processing._finite_build_runner._get_coils_from_bfield",
            return_value=[],
        ):
            with patch(
                "stellcoilbench.post_processing._finite_build_runner.get_unique_coils",
                return_value=[],
            ):
                with patch(
                    "stellcoilbench.finite_build.finite_build_coils_to_vtk",
                ) as mock_fb_vtk:
                    _run_finite_build_vtk(
                        bfield=mock_bfield,
                        surface=mock_surface,
                        output_dir=tmp_path,
                        width=0.35,
                        height=0.35,
                        results=results,
                    )

        mock_fb_vtk.assert_not_called()
        assert "finite_build_vtk_path" not in results

    def test_finite_build_vtk_exception_handled_with_diagnostic(
        self, tmp_path: Path
    ) -> None:
        """Exception from finite_build_coils_to_vtk is caught; on_catch diagnostic runs."""
        mock_bfield = MagicMock()
        mock_surface = _make_mock_surface()
        results: dict = {}
        mock_coil = _make_mock_coil()

        with patch(
            "stellcoilbench.post_processing._finite_build_runner.proc0_print",
        ) as mock_print:
            with patch(
                "stellcoilbench.post_processing._finite_build_runner._get_coils_from_bfield",
                return_value=[mock_coil],
            ):
                with patch(
                    "stellcoilbench.post_processing._finite_build_runner.get_unique_coils",
                    return_value=[mock_coil],
                ):
                    with patch(
                        "stellcoilbench.finite_build.finite_build_coils_to_vtk",
                        side_effect=OSError("Gmsh failed"),
                    ):
                        _run_finite_build_vtk(
                            bfield=mock_bfield,
                            surface=mock_surface,
                            output_dir=tmp_path,
                            width=0.35,
                            height=0.35,
                            results=results,
                        )

        assert "finite_build_vtk_path" not in results
        # on_catch calls proc0_print with diagnostic
        assert any(
            "finite-build diagnostic" in str(c) for c in mock_print.call_args_list
        )

    def test_finite_build_vtk_coil_gamma_error_logged(self, tmp_path: Path) -> None:
        """When coil.curve.gamma() raises, error is logged and processing continues."""
        mock_bfield = MagicMock()
        mock_surface = _make_mock_surface()
        results: dict = {}
        mock_coil_ok = _make_mock_coil()
        mock_coil_bad = _make_mock_coil(gamma_raises=ValueError("bad curve"))

        with patch(
            "stellcoilbench.post_processing._finite_build_runner.proc0_print",
        ) as mock_print:
            with patch(
                "stellcoilbench.post_processing._finite_build_runner._get_coils_from_bfield",
                return_value=[mock_coil_ok],
            ):
                with patch(
                    "stellcoilbench.post_processing._finite_build_runner.get_unique_coils",
                    return_value=[mock_coil_bad, mock_coil_ok],
                ):
                    with patch(
                        "stellcoilbench.finite_build.finite_build_coils_to_vtk",
                    ) as mock_fb_vtk:
                        mock_fb_vtk.return_value = tmp_path / "coils.vtk"
                        _run_finite_build_vtk(
                            bfield=mock_bfield,
                            surface=mock_surface,
                            output_dir=tmp_path,
                            width=0.35,
                            height=0.28,
                            results=results,
                        )

        mock_fb_vtk.assert_called_once()
        assert "finite_build_vtk_path" in results
        assert any("ERROR(ValueError)" in str(c) for c in mock_print.call_args_list)


class TestRunShapeGradientAnalysis:
    """Tests for _run_shape_gradient_analysis."""

    def test_run_shape_gradient_no_coils_warning(self, tmp_path: Path) -> None:
        """When no coils in bfield, proc0_warning is called and no exception."""
        mock_bfield = MagicMock()
        mock_surface = MagicMock()
        results: dict = {}

        with patch(
            "stellcoilbench.post_processing._structural_runner._get_coils_from_bfield",
            return_value=[],
        ):
            with patch(
                "stellcoilbench.post_processing._structural_runner.proc0_warning",
            ) as mock_warning:
                _run_shape_gradient_analysis(
                    bfield=mock_bfield,
                    surface=mock_surface,
                    output_dir=tmp_path,
                    results=results,
                )

        mock_warning.assert_called_once()
        assert "no coils found" in mock_warning.call_args[0][0].lower()
