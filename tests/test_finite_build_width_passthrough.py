"""Test that --finite-build-width flows to structural stress objective."""

from __future__ import annotations

from pathlib import Path

from stellcoilbench.case_loader import load_case
from stellcoilbench.coil_optimization._config_parsing import (
    _prepare_optimization_config,
)
from stellcoilbench.coil_optimization._scipy_optimizer import _parse_optimizer_config


def test_finite_build_width_in_thresholds() -> None:
    """finite_build_width in kwargs produces finite_build_width in thresholds."""
    case_path = Path("cases/basic_LandremanPaulQA.yaml")
    case_cfg = load_case(case_path)
    case_yaml_abs = case_path.resolve()
    coils_path = Path("/tmp/test_coils.json")
    output_dir = Path("/tmp/test_output")

    config = _prepare_optimization_config(
        case_cfg,
        case_path,
        case_yaml_abs,
        coils_path,
        output_dir,
        surface_resolution=32,
    )
    surface = config["surface"]
    coil_objective_terms = config["coil_objective_terms"]

    # Simulate dispatch adding finite_build_width from CLI
    kwargs = dict(config["threshold_kwargs"])
    kwargs["finite_build_width"] = 0.1

    opt_config = _parse_optimizer_config(
        surface,
        kwargs,
        max_iterations=10,
        is_continuation_step=False,
        coil_objective_terms=coil_objective_terms,
    )
    th = opt_config["thresholds"]

    assert "finite_build_width" in th
    assert th["finite_build_width"] == 0.1
