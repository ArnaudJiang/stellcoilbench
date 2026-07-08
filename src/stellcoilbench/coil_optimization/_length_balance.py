"""Objectives and metrics for balancing per-coil lengths."""

from __future__ import annotations

from typing import Any

import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec


def coil_length_distribution_metrics(lengths: list[float]) -> dict[str, Any]:
    """Return robust summary metrics for per-base-coil lengths."""
    values = np.asarray(lengths, dtype=float)
    if values.size == 0:
        return {
            "final_mean_coil_length": 0.0,
            "final_length_variance": 0.0,
            "final_length_std": 0.0,
            "final_length_cv": 0.0,
            "final_length_ratio": 0.0,
            "final_max_length": 0.0,
            "final_min_length": 0.0,
        }
    mean = float(np.mean(values))
    variance = float(np.mean((values - mean) ** 2))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    return {
        "final_mean_coil_length": mean,
        "final_length_variance": variance,
        "final_length_std": float(np.sqrt(variance)),
        "final_length_cv": float(np.sqrt(variance) / mean) if mean > 0 else 0.0,
        "final_length_ratio": float(maximum / minimum) if minimum > 0 else float("inf"),
        "final_max_length": maximum,
        "final_min_length": minimum,
    }


class CoilLengthVariance(Optimizable):
    """Simsopt-compatible objective for variance of per-coil lengths.

    The objective is ``mean((L_i - mean(L))**2)`` in m^2. Its gradient is
    computed from the child ``CurveLength`` gradients via the chain rule.
    """

    def __init__(self, length_objectives: list[Any]) -> None:
        if not length_objectives:
            raise ValueError("CoilLengthVariance requires at least one CurveLength objective")
        super().__init__(depends_on=length_objectives)
        self.length_objectives = list(length_objectives)

    @property
    def x(self) -> np.ndarray:
        return self.length_objectives[0].x

    @x.setter
    def x(self, value: np.ndarray) -> None:
        for obj in self.length_objectives:
            obj.x = value

    def J(self) -> float:
        lengths = np.asarray([float(obj.J()) for obj in self.length_objectives])
        mean = float(np.mean(lengths))
        return float(np.mean((lengths - mean) ** 2))

    @derivative_dec
    def dJ(self, **kwargs: Any) -> Derivative:
        lengths = np.asarray([float(obj.J()) for obj in self.length_objectives])
        mean = float(np.mean(lengths))
        n = float(len(lengths))
        derivative = Derivative({})
        for length, obj in zip(lengths, self.length_objectives):
            coeff = (2.0 / n) * float(length - mean)
            child_derivative = obj.dJ(partials=True)
            if isinstance(child_derivative, Derivative):
                derivative += coeff * child_derivative
            else:
                derivative += Derivative({obj: coeff * np.asarray(child_derivative)})
        return derivative
