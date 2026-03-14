"""Tests for verbose iteration output formatting."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
from simsopt.geo import create_equally_spaced_curves

from stellcoilbench.coil_optimization._iteration_output import (
    _format_verbose_iteration_output,
)


def test_format_verbose_iteration_output_structure() -> None:
    """Main and contrib lines have expected structure."""
    Jls = [Mock(J=Mock(return_value=10.0))]
    Jccdist = Mock(shortest_distance=Mock(return_value=0.5))
    Jcsdist = Mock(shortest_distance=Mock(return_value=0.3))
    base_curves = create_equally_spaced_curves(
        1, 1, stellsym=False, R0=1.0, R1=0.1, order=2, numquadpoints=32
    )

    grad = np.array([1e-4, 2e-4])
    weights = [1.0, 0.1]
    c0 = Mock(J=Mock(return_value=5.0))
    c1 = Mock(J=Mock(return_value=0.2))
    c_list = [c0, c1]
    constraint_names_and_thresholds = [("CC Distance", 0.5)]

    main_line, contrib_line = _format_verbose_iteration_output(
        iteration=0,
        Jls=Jls,
        Jccdist=Jccdist,
        Jcsdist=Jcsdist,
        base_curves=base_curves,
        Jlink=None,
        grad=grad,
        weights=weights,
        c_list=c_list,
        constraint_names_and_thresholds=constraint_names_and_thresholds,
        J_total=5.02,
    )

    assert main_line.startswith("[0]")
    assert "L=" in main_line
    assert "d_cc=" in main_line
    assert "d_cs=" in main_line
    assert "‖∇J‖" in main_line
    assert ", F=" not in main_line and ", Tq=" not in main_line
    assert "Objs:" in contrib_line
    assert "Total=" in contrib_line
    # Contributions should sum to J_total
    assert "J_f=5.0e+00" in contrib_line
    assert "d_cc=2.0e-02" in contrib_line  # 0.1 * 0.2 = 0.02
    assert "Total=5.0e+00" in contrib_line  # 5.02 rounded to 1 decimal


def test_format_verbose_iteration_output_with_max_force_torque() -> None:
    """When max_force and max_torque are provided, F= and Tq= appear on main line."""
    Jls = [Mock(J=Mock(return_value=10.0))]
    Jccdist = Mock(shortest_distance=Mock(return_value=0.5))
    Jcsdist = Mock(shortest_distance=Mock(return_value=0.3))
    base_curves = create_equally_spaced_curves(
        1, 1, stellsym=False, R0=1.0, R1=0.1, order=2, numquadpoints=32
    )
    grad = np.array([1e-4, 2e-4])
    weights = [1.0, 0.1]
    c_list = [Mock(J=Mock(return_value=5.0)), Mock(J=Mock(return_value=0.2))]
    constraint_names_and_thresholds = [("CC Distance", 0.5)]

    main_line, _ = _format_verbose_iteration_output(
        iteration=1,
        Jls=Jls,
        Jccdist=Jccdist,
        Jcsdist=Jcsdist,
        base_curves=base_curves,
        Jlink=None,
        grad=grad,
        weights=weights,
        c_list=c_list,
        constraint_names_and_thresholds=constraint_names_and_thresholds,
        J_total=5.02,
        max_force=123.45,
        max_torque=67.89,
    )

    assert ", F=1.23e+02" in main_line
    assert ", Tq=6.79e+01" in main_line
