"""Tests for backfill_reactor_scale_metrics."""

from __future__ import annotations

import numpy as np
import pytest

from stellcoilbench.reactor_scale import STELLARIS_A_TURN
from stellcoilbench.update_db._backfill import backfill_reactor_scale_metrics


class TestBackfillReactorScaleMetrics:
    """Tests for backfill_reactor_scale_metrics."""

    def test_backfill_winding_pack_width(self) -> None:
        """Backfill sets winding_pack_width_per_coil and max_winding_pack_width."""
        reactor_scale: dict = {"N_turns_per_coil": [4, 9]}
        metrics: dict = {}
        backfill_reactor_scale_metrics(reactor_scale, metrics)

        turn_side = np.sqrt(STELLARIS_A_TURN)
        expected_wp = [
            float(np.sqrt(4) * turn_side),
            float(np.sqrt(9) * turn_side),
        ]
        assert reactor_scale["winding_pack_width_per_coil"] == expected_wp
        assert reactor_scale["max_winding_pack_width"] == float(max(expected_wp))

    def test_backfill_finite_build_cc_clearance(self) -> None:
        """Backfill sets finite_build_cc_clearance from d_cc - max_wp."""
        reactor_scale: dict = {
            "N_turns_per_coil": [4],
            "max_winding_pack_width": 0.3,
            "reactor_scale_min_cc_separation": 0.5,
        }
        metrics: dict = {}
        backfill_reactor_scale_metrics(reactor_scale, metrics)

        assert reactor_scale["finite_build_cc_clearance"] == 0.2

    def test_backfill_per_turn_force(self) -> None:
        """Backfill sets per_turn_max_force from reactor_scale_force_per_coil_MN_per_m."""
        reactor_scale: dict = {
            "N_turns_per_coil": [10, 8],
            "reactor_scale_force_per_coil_MN_per_m": [0.5, 0.4],
        }
        metrics: dict = {}
        backfill_reactor_scale_metrics(reactor_scale, metrics)

        per_turn_f = [0.5 / 10, 0.4 / 8]
        assert reactor_scale["per_turn_max_force"] == float(max(per_turn_f))

    def test_backfill_total_superconductor_length_km(self) -> None:
        """Backfill sets total_superconductor_length_km from per-coil lengths."""
        reactor_scale: dict = {
            "N_turns_per_coil": [100, 80],
            "reactor_scale_total_length": 1e6,  # 1000 km
        }
        metrics: dict = {
            "final_length_per_coil": [1000.0, 800.0],
            "final_total_length": 1800.0,
        }
        backfill_reactor_scale_metrics(reactor_scale, metrics)

        L_scale = 1e6 / 1800
        reactor_lengths = [1000 * L_scale, 800 * L_scale]
        expected = sum(n * ln for n, ln in zip([100, 80], reactor_lengths)) / 1e3
        assert reactor_scale["total_superconductor_length_km"] == pytest.approx(
            expected
        )
