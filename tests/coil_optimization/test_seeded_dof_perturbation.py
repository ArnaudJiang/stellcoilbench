from __future__ import annotations

import numpy as np

from stellcoilbench.coil_optimization._optimization_setup import (
    _initialize_coils_for_optimization,
)


class _FakeCurve:
    def __init__(self) -> None:
        self.x = np.linspace(0.1, 1.0, 12)


class _FakeCoil:
    def __init__(self) -> None:
        self.curve = _FakeCurve()


def _perturbed_x(seed: int) -> np.ndarray:
    coils = [_FakeCoil(), _FakeCoil()]
    out, _ = _initialize_coils_for_optimization(
        s=None,
        target_B=1.0,
        out_dir=None,
        ncoils=2,
        order=6,
        coil_width=0.1,
        numquadpoints=128,
        regularization=None,
        initial_coils=coils,
        is_continuation_step=False,
        kwargs={"dof_perturbation": 0.02, "random_seed": seed},
    )
    return out[0].curve.x.copy()


def test_dof_perturbation_is_reproducible_for_same_seed() -> None:
    assert np.allclose(_perturbed_x(4101), _perturbed_x(4101))


def test_dof_perturbation_changes_for_different_seed() -> None:
    assert not np.allclose(_perturbed_x(4101), _perturbed_x(4102))
