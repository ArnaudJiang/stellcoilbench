from __future__ import annotations

import numpy as np

from stellcoilbench.coil_optimization import _scipy_optimizer
from stellcoilbench.coil_optimization._early_stop import (
    EarlyStopController,
    EarlyStopTriggered,
)


class FakeDistance:
    def __init__(self, value: float) -> None:
        self.value = value

    def shortest_distance(self) -> float:
        return self.value


class FakeCurve:
    def __init__(self, kappa: float = 1.0, torsion: float = 1.0) -> None:
        self._kappa = kappa
        self._torsion = torsion

    def kappa(self) -> np.ndarray:
        return np.array([self._kappa])

    def torsion(self) -> np.ndarray:
        return np.array([self._torsion])


def test_early_stop_triggers_after_sustained_low_cs(tmp_path):
    controller = EarlyStopController(
        {
            "enabled": True,
            "min_eval": 10,
            "check_interval": 5,
            "hard_min_cs": 0.15,
            "sustained_bad_checks": 2,
        },
        base_curves=[FakeCurve()],
        Jccdist=FakeDistance(0.3),
        Jcsdist=FakeDistance(0.1),
        output_dir=tmp_path,
    )

    controller.maybe_check(10, 100.0)
    try:
        controller.maybe_check(15, 99.0)
    except EarlyStopTriggered as exc:
        assert "cs_shortest_distance<0.15" in exc.status["reason"]
    else:
        raise AssertionError("expected early stop")

    assert (tmp_path / "early_stop_history.csv").exists()
    assert (tmp_path / "early_stop_final.json").exists()


def test_scipy_minimize_converts_early_stop_to_result(monkeypatch, tmp_path):
    controller = EarlyStopController(
        {
            "enabled": True,
            "min_eval": 1,
            "check_interval": 1,
            "hard_min_cc": 0.2,
            "sustained_bad_checks": 1,
        },
        base_curves=[FakeCurve()],
        Jccdist=FakeDistance(0.1),
        Jcsdist=FakeDistance(0.3),
        output_dir=tmp_path,
    )

    class FakeJF:
        x = np.array([1.0, 2.0])

    def objective(_x):
        controller.maybe_check(1, 1.0)
        return 1.0

    def fake_minimize(**kwargs):
        kwargs["fun"](np.array([3.0, 4.0]))
        raise AssertionError("objective should have raised EarlyStopTriggered")

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    result, iterations = _scipy_optimizer._invoke_scipy_minimize(
        objective,
        lambda x: np.ones_like(x),
        FakeJF(),
        "L-BFGS-B",
        10,
        {},
        early_stop=controller,
    )

    assert result.success is False
    assert "early stop:" in result.message
    assert iterations == 1
    assert result.nfev == 1
