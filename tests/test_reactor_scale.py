"""Tests for stellcoilbench.reactor_scale module.

Covers REBCO critical-current model, turn-count computation,
reactor scaling, and the top-level compute_reactor_scale_metrics.
"""

from __future__ import annotations

import pytest

from stellcoilbench.reactor_scale import (
    STELLARIS_T_OP,
    _apply_reactor_scaling,
    _compute_turn_counts,
    _rebco_jc_tape_stack,
    compute_N_turns_critical_current,
)


@pytest.mark.parametrize(
    "B,T_op,check_type,check_args",
    [
        (0.0, 20.0, "approx", (5.0e9, 0.01)),
        (10.0, 20.0, "decreases_with_B", ()),
        (20.0, 20.0, "range", (1.5e9, 3.5e9)),
        (-5.0, 20.0, "same_as_B0", ()),
        (10.0, 40.0, "decreases_with_T", (20.0,)),
        (10.0, 92.0, "exact", (0.0,)),
        (None, None, "grid_non_negative", ([0, 5, 10, 20, 50, 100], [4.2, 20, 40, 77, 92])),
    ],
    ids=[
        "zero_field",
        "positive_field_decreases_jc",
        "high_field",
        "negative_field_treated_as_zero",
        "temperature_scaling",
        "critical_temperature_gives_zero",
        "always_non_negative",
    ],
)
def test_rebco_jc_tape_stack(B, T_op, check_type, check_args) -> None:
    """REBCO Kim-like Jc parametrization: field/temperature scaling and bounds."""
    if check_type == "grid_non_negative":
        B_vals, T_vals = check_args
        for b in B_vals:
            for t in T_vals:
                assert _rebco_jc_tape_stack(float(b), float(t)) >= 0.0
        return
    jc = _rebco_jc_tape_stack(B, T_op=T_op)
    if check_type == "approx":
        expected, rel = check_args
        assert jc == pytest.approx(expected, rel=rel)
    elif check_type == "exact":
        assert jc == check_args[0]
    elif check_type == "range":
        lo, hi = check_args
        assert lo < jc < hi
    elif check_type == "decreases_with_B":
        jc_0 = _rebco_jc_tape_stack(0.0)
        assert jc < jc_0 and jc > 0
    elif check_type == "same_as_B0":
        jc_zero = _rebco_jc_tape_stack(0.0)
        assert jc == pytest.approx(jc_zero)
    elif check_type == "decreases_with_T":
        T_ref = check_args[0]
        jc_ref = _rebco_jc_tape_stack(B, T_op=T_ref)
        assert jc < jc_ref and jc > 0


class TestComputeNTurnsCriticalCurrent:
    """Tests for compute_N_turns_critical_current."""

    def test_basic_single_coil(self) -> None:
        """Single coil should return sensible N_turns and model data."""
        result = compute_N_turns_critical_current(
            per_coil_forces=[1e6],
            per_coil_currents=[1e5],
            per_coil_lengths=[20.0],
            L_scale=2.0,
            B_scale=1.5,
            target_B=5.0,
        )
        assert "N_turns_jc" in result
        assert len(result["N_turns_jc"]) == 1
        assert result["N_turns_jc"][0] >= 1
        assert "model_params" in result
        assert result["model_params"]["T_op_K"] == STELLARIS_T_OP

    def test_multiple_coils(self) -> None:
        """Multiple coils should return per-coil lists of equal length."""
        n = 4
        result = compute_N_turns_critical_current(
            per_coil_forces=[1e6] * n,
            per_coil_currents=[1e5] * n,
            per_coil_lengths=[20.0] * n,
            L_scale=1.0,
            B_scale=1.0,
            target_B=5.7,
        )
        for key in ("N_turns_jc", "NI_reactor", "I_turn", "B_peak_estimate"):
            assert len(result[key]) == n

    def test_none_currents_fallback(self) -> None:
        """When per_coil_currents is None, should still compute N_turns."""
        result = compute_N_turns_critical_current(
            per_coil_forces=[1e6],
            per_coil_currents=None,
            per_coil_lengths=None,
            L_scale=1.0,
            B_scale=1.0,
            target_B=5.0,
        )
        assert result["N_turns_jc"][0] >= 1

    def test_zero_force(self) -> None:
        """Zero force should still produce a valid (minimum) turn count."""
        result = compute_N_turns_critical_current(
            per_coil_forces=[0.0],
            per_coil_currents=[1e5],
            per_coil_lengths=[20.0],
            L_scale=1.0,
            B_scale=1.0,
            target_B=5.0,
        )
        assert result["N_turns_jc"][0] >= 1


class TestApplyReactorScaling:
    """Tests for _apply_reactor_scaling."""

    @pytest.mark.parametrize(
        "metrics,L_scale,B_scale,assertions",
        [
            (
                {"final_min_cs_separation": 0.5, "final_total_length": 100.0},
                3.0,
                1.0,
                [
                    ("reactor_scale_min_cs_separation", 1.5),
                    ("reactor_scale_total_length", 300.0),
                ],
            ),
            (
                {"final_max_curvature": 2.0},
                4.0,
                1.0,
                [("reactor_scale_max_curvature", 0.5)],
            ),
            (
                {"final_mean_squared_curvature": 4.0},
                2.0,
                1.0,
                [("reactor_scale_mean_squared_curvature", 1.0)],
            ),
            (
                {"final_max_max_coil_force": 1e6},
                2.0,
                3.0,
                [("reactor_scale_max_max_coil_force", 18.0)],
            ),
            (
                {"final_max_max_coil_torque": 1e6},
                2.0,
                3.0,
                [("reactor_scale_max_max_coil_torque", 36.0)],
            ),
            (
                {"final_squared_flux": 1.0},
                2.0,
                3.0,
                [("reactor_scale_squared_flux", 36.0)],
            ),
            ({}, 1.0, 1.0, []),
            (
                {"final_total_length": 50.0},
                1.0,
                1.0,
                [("reactor_scale_total_length", 50.0)],
            ),
        ],
        ids=[
            "length_scaling",
            "curvature_scaling",
            "mean_squared_curvature_scaling",
            "force_scaling",
            "torque_scaling",
            "flux_scaling",
            "empty_metrics",
            "identity_scaling",
        ],
    )
    def test_apply_reactor_scaling(
        self, metrics: dict, L_scale: float, B_scale: float, assertions: list
    ) -> None:
        """_apply_reactor_scaling correctly scales lengths, curvature, force, torque, flux."""
        scaled = _apply_reactor_scaling(metrics, L_scale=L_scale, B_scale=B_scale)
        if not assertions:
            assert scaled == {}
        else:
            for key, val in assertions:
                assert scaled[key] == pytest.approx(val)


class TestComputeTurnCounts:
    """Tests for _compute_turn_counts (in-place reactor_metrics update)."""

    def test_populates_turn_count_keys(self) -> None:
        """Should add N_turns_per_coil and related keys."""
        metrics = {
            "final_max_force_per_coil": [1e6, 2e6],
            "final_current_per_coil": [1e5, 1.5e5],
            "final_length_per_coil": [20.0, 25.0],
        }
        reactor_metrics: dict = {}
        _compute_turn_counts(
            metrics, reactor_metrics, L_scale=2.0, B_scale=1.5, target_B=5.0
        )

        assert "N_turns_per_coil" in reactor_metrics
        assert len(reactor_metrics["N_turns_per_coil"]) == 2
        assert all(n >= 1 for n in reactor_metrics["N_turns_per_coil"])

    def test_no_force_data_returns_early(self) -> None:
        """Without force or current data, should not populate turn counts."""
        reactor_metrics: dict = {}
        _compute_turn_counts(
            {}, reactor_metrics, L_scale=1.0, B_scale=1.0, target_B=5.0
        )
        assert "N_turns_per_coil" not in reactor_metrics

    def test_winding_pack_width(self) -> None:
        """Should compute winding_pack_width_per_coil."""
        metrics = {
            "final_max_force_per_coil": [1e6],
            "final_current_per_coil": [1e5],
            "final_length_per_coil": [20.0],
        }
        reactor_metrics: dict = {}
        _compute_turn_counts(
            metrics, reactor_metrics, L_scale=2.0, B_scale=1.5, target_B=5.0
        )
        assert "winding_pack_width_per_coil" in reactor_metrics
        assert "max_winding_pack_width" in reactor_metrics
        assert reactor_metrics["max_winding_pack_width"] > 0
