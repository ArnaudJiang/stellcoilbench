import pytest

from stellcoilbench.coil_optimization._adaptive_search import _make_base_currents


def test_make_base_currents_uniform_fixed_equal() -> None:
    total_current = 1.2e6
    currents = _make_base_currents(total_current, ncoils=3)

    values = [current.get_value() for current in currents]
    assert values == pytest.approx([4.0e5, 4.0e5, 4.0e5])
    assert sum(values) == pytest.approx(total_current)
    assert all(current.local_full_dof_size == 0 for current in currents)


def test_make_base_currents_weighted_fixed() -> None:
    total_current = 1.2e6
    currents = _make_base_currents(total_current, ncoils=3, current_weights=[1, 2, 3])

    values = [current.get_value() for current in currents]
    assert values == pytest.approx([2.0e5, 4.0e5, 6.0e5])
    assert sum(values) == pytest.approx(total_current)
    assert all(current.local_full_dof_size == 0 for current in currents)
