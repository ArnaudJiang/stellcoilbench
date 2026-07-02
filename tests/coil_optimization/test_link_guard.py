from __future__ import annotations

import numpy as np

from stellcoilbench.coil_optimization._link_guard import PairwiseLinkGuard
from stellcoilbench.coil_optimization import _scipy_optimizer


class FakeCurve:
    def __init__(self, points: np.ndarray) -> None:
        self._points = points

    def gamma(self) -> np.ndarray:
        return self._points


def _xy_circle(center=(0.0, 0.0, 0.0), radius=1.0, n=256) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack(
        [
            center[0] + radius * np.cos(t),
            center[1] + radius * np.sin(t),
            np.full_like(t, center[2]),
        ]
    )


def _xz_circle(center=(0.0, 0.0, 0.0), radius=1.0, n=256) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack(
        [
            center[0] + radius * np.cos(t),
            np.full_like(t, center[1]),
            center[2] + radius * np.sin(t),
        ]
    )


def test_pairwise_link_guard_no_penalty_for_unchanged_unlinked_curves(tmp_path):
    curves = [
        FakeCurve(_xy_circle(center=(0.0, 0.0, 0.0))),
        FakeCurve(_xy_circle(center=(4.0, 0.0, 0.0))),
    ]
    guard = PairwiseLinkGuard(curves, output_dir=tmp_path, penalty=123.0)

    assert guard.evaluate(1) == 0.0
    assert (tmp_path / "link_guard_initial.json").exists()
    assert (tmp_path / "link_guard_history.csv").exists()


def test_pairwise_link_guard_penalizes_topology_change():
    mutable = FakeCurve(_xy_circle(center=(4.0, 0.0, 0.0)))
    guard = PairwiseLinkGuard(
        [FakeCurve(_xy_circle()), mutable],
        penalty=123.0,
        tolerance=0.5,
    )

    mutable._points = _xz_circle(center=(0.5, 0.0, 0.0), radius=0.6)

    assert guard.evaluate(1) == 123.0


def test_pairwise_link_guard_tracks_and_restores_last_safe_point(tmp_path):
    mutable = FakeCurve(_xy_circle(center=(4.0, 0.0, 0.0)))
    guard = PairwiseLinkGuard(
        [FakeCurve(_xy_circle()), mutable],
        output_dir=tmp_path,
        penalty=123.0,
        tolerance=0.5,
        rollback=True,
    )

    safe_x = np.array([1.0, 2.0, 3.0])
    assert guard.evaluate(1, x=safe_x, objective=7.0) == 0.0
    mutable._points = _xz_circle(center=(0.5, 0.0, 0.0), radius=0.6)
    assert guard.evaluate(2, x=np.array([9.0, 9.0, 9.0]), objective=8.0) == 123.0

    class Target:
        x = np.zeros(3)

    target = Target()
    assert guard.restore_last_safe(target)
    np.testing.assert_allclose(target.x, safe_x)
    guard.write_final_audit(restored=True)
    assert (tmp_path / "link_guard_final.json").exists()


def test_scipy_minimize_restores_last_safe_when_final_state_is_linked(monkeypatch):
    mutable = FakeCurve(_xy_circle(center=(4.0, 0.0, 0.0)))
    guard = PairwiseLinkGuard(
        [FakeCurve(_xy_circle()), mutable],
        penalty=123.0,
        tolerance=0.5,
    )
    safe_x = np.array([1.0, 2.0, 3.0])
    guard.evaluate(1, x=safe_x, objective=7.0)
    mutable._points = _xz_circle(center=(0.5, 0.0, 0.0), radius=0.6)

    class FakeJF:
        x = np.array([9.0, 9.0, 9.0])

    class Result:
        x = np.array([9.0, 9.0, 9.0])
        success = True
        message = "done"
        nit = 2

    def fake_minimize(**_kwargs):
        return Result()

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)
    result, iterations = _scipy_optimizer._invoke_scipy_minimize(
        lambda x: float(np.sum(x)),
        lambda x: np.ones_like(x),
        FakeJF(),
        "L-BFGS-B",
        10,
        {},
        link_guard=guard,
    )

    assert iterations == 2
    assert result.success is False
    np.testing.assert_allclose(result.x, safe_x)
    assert "restored last no-link checkpoint" in result.message


def test_scipy_minimize_marks_failure_without_safe_checkpoint(monkeypatch):
    mutable = FakeCurve(_xz_circle(center=(0.5, 0.0, 0.0), radius=0.6))
    guard = PairwiseLinkGuard(
        [FakeCurve(_xy_circle()), FakeCurve(_xy_circle(center=(4.0, 0.0, 0.0)))],
        penalty=123.0,
        tolerance=0.5,
    )
    guard.curves[1] = mutable

    class FakeJF:
        x = np.array([9.0, 9.0, 9.0])

    class Result:
        x = np.array([9.0, 9.0, 9.0])
        success = True
        message = "done"
        nit = 2

    def fake_minimize(**_kwargs):
        return Result()

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)
    result, _iterations = _scipy_optimizer._invoke_scipy_minimize(
        lambda x: float(np.sum(x)),
        lambda x: np.ones_like(x),
        FakeJF(),
        "L-BFGS-B",
        10,
        {},
        link_guard=guard,
    )

    assert result.success is False
    assert "no no-link checkpoint available" in result.message
