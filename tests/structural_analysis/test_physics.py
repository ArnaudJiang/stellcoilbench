"""Tests for structural analysis physics helpers."""

from __future__ import annotations

import numpy as np

from stellcoilbench.structural_analysis._physics import (
    _G_helper,
    _lame_parameters,
)


class TestLameParameters:
    """Tests for _lame_parameters."""

    def test_known_values(self) -> None:
        E = 200e9
        nu = 0.3
        lam, mu = _lame_parameters(E, nu)
        expected_lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        expected_mu = E / (2 * (1 + nu))
        assert np.isclose(lam, expected_lam)
        assert np.isclose(mu, expected_mu)

    def test_shear_modulus_independent_of_nu_bound(self) -> None:
        """mu = E/(2(1+nu)) is finite even when lam diverges."""
        E = 100e9
        nu = 0.49
        lam, mu = _lame_parameters(E, nu)
        assert lam > 0
        assert 0 < mu < E


class TestGHelper:
    """Tests for _G_helper (Landreman Eq 17-18)."""

    def test_both_zero(self) -> None:
        assert _G_helper(0.0, 0.0) == 0.0
        x = np.array([0.0, 0.0])
        y = np.array([0.0, 0.0])
        np.testing.assert_array_equal(_G_helper(x, y), [0.0, 0.0])

    def test_x_zero_y_nonzero(self) -> None:
        result = _G_helper(0.0, 1.0)
        assert result == 0.0

    def test_general_case(self) -> None:
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 1.0])
        out = _G_helper(x, y)
        expected = y * np.arctan(x / y) + (x / 2) * np.log(1.0 + (y**2) / (x**2))
        np.testing.assert_allclose(out, expected)
