"""Tests for structural_analysis backend detection."""

from __future__ import annotations

import pytest

from stellcoilbench.constants import WP_POISSON_RATIO, WP_YOUNGS_MODULUS_PA


class TestBackendDetection:
    """Tests for _require_backend and import guards."""

    def test_require_backend_returns_string(self):
        """_require_backend should return 'dolfinx' or 'skfem' or raise."""
        from stellcoilbench.structural_analysis import (
            _DOLFINX_AVAILABLE,
            _SKFEM_AVAILABLE,
        )

        if _DOLFINX_AVAILABLE or _SKFEM_AVAILABLE:
            from stellcoilbench.structural_analysis import _require_backend

            backend = _require_backend()
            assert backend in ("dolfinx", "skfem")
        else:
            from stellcoilbench.structural_analysis import _require_backend

            with pytest.raises(ImportError, match="Structural analysis requires"):
                _require_backend()

    def test_constants_exported(self):
        """Default material properties should be accessible."""
        assert WP_YOUNGS_MODULUS_PA == 100.0e9
        assert WP_POISSON_RATIO == 0.3


class TestCommonHelpers:
    """Tests for _common shared helpers."""

    def test_compute_z_threshold_for_fixed_support(self):
        """_compute_z_threshold_for_fixed_support uses BC_Z_FRACTION correctly."""
        from stellcoilbench.structural_analysis._common import (
            BC_Z_FRACTION,
            _compute_z_threshold_for_fixed_support,
        )

        z_min, z_max = 0.0, 1.0
        th = _compute_z_threshold_for_fixed_support(z_min, z_max)
        expected = z_min + BC_Z_FRACTION * (z_max - z_min)
        assert th == pytest.approx(expected)
        assert th == pytest.approx(0.15)

        th_custom = _compute_z_threshold_for_fixed_support(
            z_min, z_max, bc_z_fraction=0.2
        )
        assert th_custom == pytest.approx(0.2)

        th_global = _compute_z_threshold_for_fixed_support(1.0, 1.0)
        assert th_global == pytest.approx(1.0)

        th_per_block_flat = _compute_z_threshold_for_fixed_support(
            1.0, 1.0, range_if_zero=1.0
        )
        assert th_per_block_flat == pytest.approx(1.0 + BC_Z_FRACTION)

    def test_prepare_structural_output_dir(self, tmp_path):
        """_prepare_structural_output_dir creates directory and returns Path."""
        from stellcoilbench.structural_analysis._common import (
            _prepare_structural_output_dir,
        )

        out = _prepare_structural_output_dir(tmp_path / "nested" / "dir")
        assert out.is_dir()
        assert str(out).endswith("nested/dir")
