"""Finite-section magnetic-field approximation for optimization.

This module approximates a rectangular winding pack by replacing each
centerline filament with a small bundle of offset filaments. The bundle uses
SIMSOPT's native ``CurveFilament`` objects so derivatives flow back to the
centerline curve and optional frame-rotation variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FiniteSectionFieldConfig:
    """Runtime finite-section field configuration."""

    enabled: bool = False
    width: float = 0.10
    height: float = 0.10
    n_width: int = 3
    n_height: int = 3
    current_distribution: str = "uniform"

    @property
    def n_filaments(self) -> int:
        """Total number of filaments per centerline coil."""
        return self.n_width * self.n_height


def parse_finite_section_field_config(raw: dict[str, Any] | None) -> FiniteSectionFieldConfig:
    """Parse a raw finite_section_field dictionary into a runtime config."""
    if not raw:
        return FiniteSectionFieldConfig()
    return FiniteSectionFieldConfig(
        enabled=bool(raw.get("enabled", False)),
        width=float(raw.get("width", 0.10)),
        height=float(raw.get("height", raw.get("width", 0.10))),
        n_width=int(raw.get("n_width", 3)),
        n_height=int(raw.get("n_height", 3)),
        current_distribution=str(raw.get("current_distribution", "uniform")),
    )


def finite_section_field_enabled(raw: dict[str, Any] | None) -> bool:
    """Return whether finite-section magnetic-field mode is enabled."""
    return parse_finite_section_field_config(raw).enabled


def _filament_gap(span: float, count: int) -> float:
    """Return spacing between filaments for a total rectangular span."""
    if count <= 1:
        return span
    return span / float(count - 1)


def build_finite_section_coil_bundle(
    coils: list[Any],
    config: FiniteSectionFieldConfig,
) -> list[Any]:
    """Build offset filament coils approximating rectangular finite cross-sections."""
    if not config.enabled:
        return coils
    if config.current_distribution != "uniform":
        raise ValueError("Only uniform finite-section current distribution is supported")

    from simsopt.field import Coil
    from simsopt.geo import create_multifilament_grid

    gap_width = _filament_gap(config.width, config.n_width)
    gap_height = _filament_gap(config.height, config.n_height)
    current_scale = 1.0 / float(config.n_filaments)
    bundle: list[Any] = []

    for coil in coils:
        filament_curves = create_multifilament_grid(
            coil.curve,
            numfilaments_n=config.n_width,
            numfilaments_b=config.n_height,
            gapsize_n=gap_width,
            gapsize_b=gap_height,
            rotation_order=None,
            frame="centroid",
        )
        for curve in filament_curves:
            bundle.append(Coil(curve, coil.current * current_scale))

    return bundle
