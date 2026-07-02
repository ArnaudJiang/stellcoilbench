"""Tests for finite-section magnetic-field bundle construction."""

from __future__ import annotations

import numpy as np
import pytest

from simsopt.field import Coil, Current
from simsopt.geo import CurveXYZFourier

from stellcoilbench.coil_optimization._finite_section_field import (
    FiniteSectionFieldConfig,
    build_finite_section_coil_bundle,
    parse_finite_section_field_config,
)


def _unit_circle_coil(current: float = 9.0) -> Coil:
    """Create a simple circular CurveXYZFourier coil in the xy plane."""
    curve = CurveXYZFourier(32, 1)
    dofs = np.zeros(9)
    dofs[2] = 1.0  # x cos(theta)
    dofs[4] = 1.0  # y sin(theta)
    curve.set_dofs(dofs)
    return Coil(curve, Current(current))


def test_parse_finite_section_field_config_defaults() -> None:
    """Missing optional fields use the smoke-test defaults."""
    config = parse_finite_section_field_config({"enabled": True, "width": 0.2})
    assert config.enabled is True
    assert config.width == pytest.approx(0.2)
    assert config.height == pytest.approx(0.2)
    assert config.n_width == 3
    assert config.n_height == 3
    assert config.n_filaments == 9


def test_build_finite_section_bundle_count_and_current_split() -> None:
    """A 3x3 native bundle creates nine offset coils with split current."""
    coil = _unit_circle_coil(current=9.0)
    config = FiniteSectionFieldConfig(
        enabled=True,
        width=0.1,
        height=0.2,
        n_width=3,
        n_height=3,
    )

    bundle = build_finite_section_coil_bundle([coil], config)

    assert len(bundle) == 9
    assert sum(c.current.get_value() for c in bundle) == pytest.approx(9.0)
    assert all(c.current.get_value() == pytest.approx(1.0) for c in bundle)
    assert {type(c.curve).__name__ for c in bundle} == {"CurveFilament"}
    assert any(
        not np.allclose(c.curve.gamma(), coil.curve.gamma()) for c in bundle
    )


def test_build_finite_section_disabled_returns_original_coils() -> None:
    """Disabled finite-section mode leaves the original coil list unchanged."""
    coils = [_unit_circle_coil()]
    config = FiniteSectionFieldConfig(enabled=False)
    assert build_finite_section_coil_bundle(coils, config) is coils
