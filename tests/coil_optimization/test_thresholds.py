from __future__ import annotations

import pytest

from stellcoilbench.coil_optimization import _thresholds
from stellcoilbench.config_scheme import ARIES_CS_MINOR_RADIUS
from stellcoilbench.coil_optimization._thresholds import get_full_thresholds


class DummySurface:
    pass


def test_device_threshold_overrides_bypass_minor_radius_scaling(monkeypatch):
    minor_radius = ARIES_CS_MINOR_RADIUS / 4.0
    monkeypatch.setattr(
        _thresholds,
        "get_reference_radii",
        lambda _surface: (2.0, minor_radius),
    )

    th = get_full_thresholds(
        DummySurface(),
        {
            "length_threshold": 12.0,
            "length_threshold_device": 10.0,
            "cc_threshold": 0.25,
            "cc_threshold_device": 0.26,
            "cs_threshold": 0.25,
            "cs_threshold_device": 0.27,
            "curvature_threshold": 5.0,
            "curvature_threshold_device": 4.8,
            "torsion_threshold": 7.0,
            "torsion_threshold_device": 6.5,
            "msc_threshold": 5.0,
            "msc_threshold_device": 4.5,
            "arclength_variation_threshold": 0.5,
            "arclength_variation_threshold_device": 0.4,
            "force_threshold": 200.0,
            "force_threshold_device": 180.0,
            "torque_threshold": 200.0,
            "torque_threshold_device": 190.0,
        },
        coil_objective_terms={},
    )

    assert th["a0"] == pytest.approx(4.0)
    assert th["length_threshold"] == pytest.approx(10.0)
    assert th["cc_threshold"] == pytest.approx(0.26)
    assert th["cs_threshold"] == pytest.approx(0.27)
    assert th["curvature_threshold"] == pytest.approx(4.8)
    assert th["torsion_threshold"] == pytest.approx(6.5)
    assert th["msc_threshold"] == pytest.approx(4.5)
    assert th["arclength_variation_threshold"] == pytest.approx(0.4)
    assert th["force_threshold"] == pytest.approx(180.0)
    assert th["torque_threshold"] == pytest.approx(190.0)


def test_reactor_scale_thresholds_keep_existing_scaling(monkeypatch):
    minor_radius = ARIES_CS_MINOR_RADIUS / 4.0
    monkeypatch.setattr(
        _thresholds,
        "get_reference_radii",
        lambda _surface: (2.0, minor_radius),
    )

    th = get_full_thresholds(
        DummySurface(),
        {
            "length_threshold": 12.0,
            "cc_threshold": 0.25,
            "cs_threshold": 0.25,
            "curvature_threshold": 5.0,
            "torsion_threshold": 7.0,
            "msc_threshold": 5.0,
            "arclength_variation_threshold": 0.5,
            "force_threshold": 200.0,
            "torque_threshold": 200.0,
        },
        coil_objective_terms={},
    )

    assert th["length_threshold"] == pytest.approx(3.0)
    assert th["cs_threshold"] == pytest.approx(0.0625)
    assert th["curvature_threshold"] == pytest.approx(20.0)
    assert th["torsion_threshold"] == pytest.approx(28.0)
    assert th["msc_threshold"] == pytest.approx(20.0)
    assert th["arclength_variation_threshold"] == pytest.approx(8.0)
    assert th["force_threshold"] == pytest.approx(50.0)
    assert th["torque_threshold"] == pytest.approx(200.0)


def test_finite_build_force_threshold_uses_same_geometric_scaling(monkeypatch):
    minor_radius = ARIES_CS_MINOR_RADIUS / 4.0
    monkeypatch.setattr(
        _thresholds,
        "get_reference_radii",
        lambda _surface: (2.0, minor_radius),
    )

    th = get_full_thresholds(
        DummySurface(),
        {
            "finite_build_width": 0.1,
            "force_threshold": 200.0,
        },
        coil_objective_terms={},
    )

    assert th["force_threshold"] == pytest.approx(50.0)
