"""Tests for constraint scaling in coil optimization.

Verifies _compute_constraint_scaling_for_term, especially force/torque
scaling that accounts for simsopt's MN (meganewton) conversion.
"""

from __future__ import annotations

import pytest

from stellcoilbench.coil_optimization._constraint_builders import (
    _compute_constraint_scaling_for_term,
)


class TestComputeConstraintScalingForceTorque:
    """Tests for force and torque constraint scaling with MN conversion."""

    def test_force_scaling_includes_1e6_factor(self) -> None:
        """Force scaling uses R^(p-1) * 1e6^p / I^(2p) for scale invariance."""
        R, I_amp, p = 1.0, 1e6, 2
        scale = _compute_constraint_scaling_for_term(
            "coil_coil_force", "lp", R, I_amp, p, base_scaling=1.0
        )
        expected = (R ** (p - 1)) * (1e6**p) / (I_amp ** (2 * p))
        assert scale == pytest.approx(expected)

    def test_torque_scaling_same_formula_as_force(self) -> None:
        """Torque uses same scaling as force (both MN-converted in simsopt)."""
        R, I_amp, p = 2.0, 5e5, 2
        force_scale = _compute_constraint_scaling_for_term(
            "coil_coil_force", "lp_threshold", R, I_amp, p, base_scaling=1.0
        )
        torque_scale = _compute_constraint_scaling_for_term(
            "coil_coil_torque", "lp_threshold", R, I_amp, p, base_scaling=1.0
        )
        assert torque_scale == pytest.approx(force_scale)

    def test_force_scaling_scale_invariance(self) -> None:
        """Scaling * J ∝ const: doubling I should multiply scaling by 1/16 for p=2."""
        R, I_amp, p = 1.0, 1e6, 2
        s1 = _compute_constraint_scaling_for_term(
            "coil_coil_force", "lp", R, I_amp, p, base_scaling=1.0
        )
        s2 = _compute_constraint_scaling_for_term(
            "coil_coil_force", "lp", R, 2 * I_amp, p, base_scaling=1.0
        )
        # J ∝ I^(2p), so scaling ∝ 1/I^(2p); doubling I -> scaling / 2^(2p) = /16 for p=2
        assert s2 == pytest.approx(s1 / 16.0)

    @pytest.mark.parametrize("p", [1, 2], ids=["p1", "p2"])
    def test_force_scaling_p_values(self, p: int) -> None:
        """Scaling formula works for p=1 and p=2."""
        R, I_amp = 1.0, 1e6
        scale = _compute_constraint_scaling_for_term(
            "coil_coil_force", "lp", R, I_amp, p, base_scaling=1.0
        )
        expected = (R ** (p - 1)) * (1e6**p) / (I_amp ** (2 * p))
        assert scale == pytest.approx(expected)
