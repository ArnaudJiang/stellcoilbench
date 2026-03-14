"""Tests for Fourier continuation in coil optimization.

Covers _extend_coils_to_higher_order (early return, CurveXYZFourier branch,
non-CurveXYZFourier padded/truncated branch) and optimize_coils_with_fourier_continuation
(single order, continuation, skip_post_processing).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from simsopt.geo import (
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt.field import Current, coils_via_symmetries

from stellcoilbench.coil_optimization._fourier_continuation import (
    _extend_coils_to_higher_order,
    optimize_coils_with_fourier_continuation,
)
from stellcoilbench.path_utils import load_surface_with_range

REPO_ROOT = Path(__file__).parent.parent.parent


# ---- Fixtures ----


@pytest.fixture
def circular_tokamak_surface() -> SurfaceRZFourier:
    """Load the circular_tokamak plasma surface from plasma_surfaces."""
    surf_path = REPO_ROOT / "plasma_surfaces" / "input.circular_tokamak"
    if not surf_path.exists():
        pytest.skip(f"Surface file not found: {surf_path}")
    return load_surface_with_range(
        surf_path,
        surface_range="half period",
        nphi=64,
        ntheta=64,
    )


@pytest.fixture
def simple_surface() -> SurfaceRZFourier:
    """Create a minimal SurfaceRZFourier for tests that don't need a real VMEC file."""
    s = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
    s.set_rc(0, 0, 1.0)
    s.set_rc(1, 0, 0.1)
    s.set_zs(0, 0, 0.0)
    return s


@pytest.fixture
def coils_order_4(simple_surface: SurfaceRZFourier):
    """Create coils at order 4 via create_equally_spaced_curves and coils_via_symmetries."""
    R0 = simple_surface.major_radius()
    R1 = simple_surface.get_rc(1, 0) * 3.5
    base_curves = create_equally_spaced_curves(
        4,
        simple_surface.nfp,
        stellsym=simple_surface.stellsym,
        R0=R0,
        R1=R1,
        order=4,
        numquadpoints=64,
    )
    base_currents = [Current(1e6), Current(-1e6), Current(1e6), Current(-1e6)]
    return coils_via_symmetries(
        base_curves,
        base_currents,
        simple_surface.nfp,
        simple_surface.stellsym,
    )


# ---- _extend_coils_to_higher_order ----


class TestExtendCoilsToHigherOrder:
    """Tests for _extend_coils_to_higher_order."""

    def test_new_order_less_than_or_equal_old_order_returns_unchanged(
        self,
        coils_order_4,
        simple_surface: SurfaceRZFourier,
    ) -> None:
        """When new_order <= old_order, coils are returned unchanged."""
        coils = coils_order_4
        # old order is 4; new_order=4 should return same coils
        result = _extend_coils_to_higher_order(
            coils, new_order=4, s=simple_surface, ncoils=4
        )
        assert result is coils

        # new_order=2 < 4
        result2 = _extend_coils_to_higher_order(
            coils, new_order=2, s=simple_surface, ncoils=4
        )
        assert result2 is coils

    def test_curve_xyz_fourier_dofs_copied_correctly(
        self,
        coils_order_4,
        simple_surface: SurfaceRZFourier,
    ) -> None:
        """When CurveXYZFourier, Fourier coefficients are copied to higher-order curves."""
        coils = coils_order_4
        old_dofs = [coil.curve.get_dofs().copy() for coil in coils[:4]]

        extended = _extend_coils_to_higher_order(
            coils, new_order=8, s=simple_surface, ncoils=4
        )

        assert len(extended) >= 4
        for i, coil in enumerate(extended[:4]):
            assert coil.curve.order == 8
            new_dofs = coil.curve.get_dofs()
            old = old_dofs[i]
            # Old order 4: 2*4+1 = 9 dofs per component, 27 total
            # New order 8: 2*8+1 = 17 dofs per component, 51 total
            for comp in range(3):
                old_per_comp = 2 * 4 + 1
                new_per_comp = 2 * 8 + 1
                for j in range(old_per_comp):
                    assert new_dofs[comp * new_per_comp + j] == pytest.approx(
                        old[comp * old_per_comp + j]
                    )

    def test_non_curve_xyz_fourier_padded_dofs(
        self,
        simple_surface: SurfaceRZFourier,
    ) -> None:
        """When old curve is not CurveXYZFourier and has fewer DOFs, pad with zeros."""
        # Build coils with non-CurveXYZFourier curves (Coil.curve is read-only, so we
        # use a minimal coil-like container). new_order=8 -> new curves have 51 dofs.
        old_dofs_short = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        class FakeCurve:
            order = 2  # Required for old_order; new_order=8 > 2 so we proceed

            def get_dofs(self):
                return old_dofs_short.copy()

        class CoilLike:
            def __init__(self, curve, current):
                self._curve = curve
                self.current = current

            @property
            def curve(self):
                return self._curve

        fake = FakeCurve()
        coils = [
            CoilLike(fake, Current(1e6)),
            CoilLike(fake, Current(-1e6)),
        ]

        extended = _extend_coils_to_higher_order(
            coils, new_order=8, s=simple_surface, ncoils=2
        )

        # Extended coils use new CurveXYZFourier curves; their dofs were set via
        # padded_dofs (old_dofs in front, zeros after)
        new_dofs_0 = extended[0].curve.get_dofs()
        new_dofs_1 = extended[1].curve.get_dofs()
        assert len(new_dofs_0) == 51  # 3 * (2*8+1) for order 8
        np.testing.assert_array_almost_equal(new_dofs_0[:5], [1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.all(new_dofs_0[5:] == 0.0)
        np.testing.assert_array_almost_equal(new_dofs_1[:5], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_non_curve_xyz_fourier_truncated_dofs(
        self,
        simple_surface: SurfaceRZFourier,
    ) -> None:
        """When old curve has more DOFs than new, truncate to fit."""
        # Need old_order < new_order (so we don't early-return) and len(old_dofs) >=
        # len(new_dofs). Use .order=2 so old_order=2, new_order=4, and 80 dofs (>=51).
        class FakeCurveWithLowOrderAndManyDofs:
            order = 2

            def get_dofs(self):
                return np.arange(80, dtype=float)

        class CoilLike:
            def __init__(self, curve, current):
                self._curve = curve
                self.current = current

            @property
            def curve(self):
                return self._curve

        fake = FakeCurveWithLowOrderAndManyDofs()
        coils = [
            CoilLike(fake, Current(1e6)),
            CoilLike(fake, Current(-1e6)),
        ]

        extended = _extend_coils_to_higher_order(
            coils, new_order=4, s=simple_surface, ncoils=2
        )

        # Should have truncated: first 27 dofs from old (order 4 -> 3*(2*4+1)=27)
        new_dofs = extended[0].curve.get_dofs()
        assert len(new_dofs) == 27
        np.testing.assert_array_almost_equal(new_dofs, np.arange(27, dtype=float))


# ---- optimize_coils_with_fourier_continuation ----


class TestOptimizeCoilsWithFourierContinuation:
    """Tests for optimize_coils_with_fourier_continuation."""

    @pytest.mark.parametrize(
        "orders,match",
        [([], "non-empty list"), ([4, 0, 8], "positive integers"), ([8, 4, 16], "ascending order")],
        ids=["empty", "non_positive", "not_ascending"],
    )
    def test_invalid_fourier_orders_raises(
        self, simple_surface: SurfaceRZFourier, orders: list, match: str
    ) -> None:
        """Invalid fourier_orders raises ValueError."""
        with pytest.raises(ValueError, match=match):
            optimize_coils_with_fourier_continuation(
                s=simple_surface,
                fourier_orders=orders,
                out_dir="/tmp/test",
                skip_post_processing=True,
            )

    def test_single_fourier_order(
        self,
        tmp_path: Path,
        simple_surface: SurfaceRZFourier,
    ) -> None:
        """With fourier_orders=[8], optimize_coils_loop is called once."""
        out_dir = tmp_path / "fourier_out"
        mock_coils = [Mock()]
        mock_results = {"J": 1.0, "n_iter": 5}

        with patch(
            "stellcoilbench.coil_optimization._optimization_loop.optimize_coils_loop"
        ) as mock_loop:
            mock_loop.return_value = (mock_coils, mock_results)
            coils, results = optimize_coils_with_fourier_continuation(
                s=simple_surface,
                fourier_orders=[8],
                out_dir=out_dir,
                max_iterations=2,
                ncoils=2,
                skip_post_processing=True,
            )

        assert coils is mock_coils
        assert results["fourier_continuation"] is True
        assert results["fourier_orders"] == [8]
        assert results["final_order"] == 8
        mock_loop.assert_called_once()
        call_kw = mock_loop.call_args[1]
        assert call_kw["order"] == 8
        assert call_kw["skip_post_processing"] is True

    def test_fourier_continuation_multi_step(
        self,
        tmp_path: Path,
        circular_tokamak_surface: SurfaceRZFourier,
    ) -> None:
        """With fourier_orders=[4,8,16], optimize_coils_loop is called three times."""
        out_dir = tmp_path / "fourier_cont"
        base_curves_4 = create_equally_spaced_curves(
            2,
            circular_tokamak_surface.nfp,
            stellsym=circular_tokamak_surface.stellsym,
            R0=circular_tokamak_surface.major_radius(),
            R1=circular_tokamak_surface.get_rc(1, 0) * 3.5,
            order=4,
            numquadpoints=64,
        )
        coils_4 = coils_via_symmetries(
            base_curves_4,
            [Current(1e6), Current(-1e6)],
            circular_tokamak_surface.nfp,
            circular_tokamak_surface.stellsym,
        )
        base_curves_8 = create_equally_spaced_curves(
            2,
            circular_tokamak_surface.nfp,
            stellsym=circular_tokamak_surface.stellsym,
            R0=circular_tokamak_surface.major_radius(),
            R1=circular_tokamak_surface.get_rc(1, 0) * 3.5,
            order=8,
            numquadpoints=64,
        )
        coils_8 = coils_via_symmetries(
            base_curves_8,
            [Current(1e6), Current(-1e6)],
            circular_tokamak_surface.nfp,
            circular_tokamak_surface.stellsym,
        )
        base_curves_16 = create_equally_spaced_curves(
            2,
            circular_tokamak_surface.nfp,
            stellsym=circular_tokamak_surface.stellsym,
            R0=circular_tokamak_surface.major_radius(),
            R1=circular_tokamak_surface.get_rc(1, 0) * 3.5,
            order=16,
            numquadpoints=64,
        )
        coils_16 = coils_via_symmetries(
            base_curves_16,
            [Current(1e6), Current(-1e6)],
            circular_tokamak_surface.nfp,
            circular_tokamak_surface.stellsym,
        )

        call_results = [
            (coils_4, {"J": 10.0, "n_iter": 2}),
            (coils_8, {"J": 5.0, "n_iter": 2}),
            (coils_16, {"J": 2.0, "n_iter": 2}),
        ]

        with patch(
            "stellcoilbench.coil_optimization._optimization_loop.optimize_coils_loop"
        ) as mock_loop:
            mock_loop.side_effect = call_results
            coils, results = optimize_coils_with_fourier_continuation(
                s=circular_tokamak_surface,
                fourier_orders=[4, 8, 16],
                out_dir=out_dir,
                max_iterations=2,
                ncoils=2,
                skip_post_processing=True,
            )

        assert coils is coils_16
        assert results["fourier_continuation"] is True
        assert results["final_order"] == 16
        assert len(results["continuation_results"]) == 3
        assert mock_loop.call_count == 3

        # First call: no initial_coils
        first_call = mock_loop.call_args_list[0]
        assert first_call[1]["order"] == 4
        assert "initial_coils" not in first_call[1] or first_call[1]["initial_coils"] is None

        # Second and third: have initial_coils (from extend)
        second_call = mock_loop.call_args_list[1]
        assert second_call[1]["order"] == 8
        assert second_call[1]["initial_coils"] is not None

        third_call = mock_loop.call_args_list[2]
        assert third_call[1]["order"] == 16
        assert third_call[1]["initial_coils"] is not None

    def test_skip_post_processing_no_pp_call(
        self,
        tmp_path: Path,
        simple_surface: SurfaceRZFourier,
    ) -> None:
        """When skip_post_processing=True, _run_post_processing is not invoked."""
        mock_coils = [Mock()]
        mock_results = {"J": 1.0}

        with patch(
            "stellcoilbench.coil_optimization._optimization_loop.optimize_coils_loop",
            return_value=(mock_coils, mock_results),
        ):
            with patch(
                "stellcoilbench.coil_optimization._post_opt_processing._run_post_processing_after_optimization"
            ) as mock_pp:
                coils, results = optimize_coils_with_fourier_continuation(
                    s=simple_surface,
                    fourier_orders=[4],
                    out_dir=tmp_path / "out",
                    skip_post_processing=True,
                )
        mock_pp.assert_not_called()
