"""Tests for per-coil length balance metrics and objective."""

from __future__ import annotations

import numpy as np
import pytest

from simsopt.objectives import Weight
from simsopt._core.optimizable import Optimizable

from stellcoilbench.coil_optimization._length_balance import (
    CoilLengthVariance,
    coil_length_distribution_metrics,
)


class _FakeLengthObjective(Optimizable):
    def __init__(self, length: float, grad: np.ndarray) -> None:
        super().__init__(x0=np.zeros_like(np.asarray(grad, dtype=float)))
        self.length = float(length)
        self.grad = np.asarray(grad, dtype=float)

    def J(self) -> float:
        return self.length

    def dJ(self, **_kwargs) -> np.ndarray:
        return self.grad


def test_coil_length_distribution_metrics() -> None:
    metrics = coil_length_distribution_metrics([5.0, 6.0, 7.0, 6.0])

    assert metrics["final_mean_coil_length"] == pytest.approx(6.0)
    assert metrics["final_length_variance"] == pytest.approx(0.5)
    assert metrics["final_length_std"] == pytest.approx(np.sqrt(0.5))
    assert metrics["final_length_cv"] == pytest.approx(np.sqrt(0.5) / 6.0)
    assert metrics["final_length_ratio"] == pytest.approx(7.0 / 5.0)
    assert metrics["final_max_length"] == pytest.approx(7.0)
    assert metrics["final_min_length"] == pytest.approx(5.0)


def test_coil_length_variance_objective_and_gradient() -> None:
    objectives = [
        _FakeLengthObjective(5.0, np.array([1.0, 0.0])),
        _FakeLengthObjective(6.0, np.array([0.0, 1.0])),
        _FakeLengthObjective(7.0, np.array([1.0, 1.0])),
    ]
    objective = CoilLengthVariance(objectives)

    assert objective.J() == pytest.approx(2.0 / 3.0)
    np.testing.assert_allclose(
        objective.dJ(),
        np.array([-2.0 / 3.0, 0.0, 0.0, 0.0, 2.0 / 3.0, 2.0 / 3.0]),
    )


def test_coil_length_variance_supports_simsopt_weight() -> None:
    objective = CoilLengthVariance(
        [
            _FakeLengthObjective(5.0, np.array([1.0])),
            _FakeLengthObjective(7.0, np.array([0.0])),
        ]
    )

    weighted = Weight(3.0) * objective

    assert weighted.J() == pytest.approx(3.0)
    np.testing.assert_allclose(weighted.dJ(), np.array([-3.0, 0.0]))
