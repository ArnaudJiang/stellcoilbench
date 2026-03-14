"""Tests for compute_qfm_surface."""

from unittest.mock import Mock, patch

import pytest

from stellcoilbench.post_processing import compute_qfm_surface


class TestComputeQfmSurface:
    """Smoke tests for compute_qfm_surface."""

    def test_compute_qfm_surface_success(self) -> None:
        pytest.importorskip("simsopt.util.permanent_magnet_helper_functions")
        from simsopt.field import BiotSavart
        from simsopt.geo import SurfaceRZFourier

        mock_surface = Mock(spec=SurfaceRZFourier)
        mock_bfield = Mock(spec=BiotSavart)
        mock_qfm_result = Mock()
        mock_qfm_surface = Mock(spec=SurfaceRZFourier)
        mock_qfm_result.surface = mock_qfm_surface

        with patch("stellcoilbench.post_processing._qfm.make_qfm") as mock_make_qfm:
            mock_make_qfm.return_value = mock_qfm_result
            result = compute_qfm_surface(mock_surface, mock_bfield)
            assert result == mock_qfm_surface
            mock_make_qfm.assert_called_once()
