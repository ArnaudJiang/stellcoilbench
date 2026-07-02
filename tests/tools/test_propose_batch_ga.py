"""Tests for GA proposer finite-section policy support."""

from __future__ import annotations

from tools.propose_batch.ga import _rng, explore_case, mutate_case


def _w7x_finite_section_policy() -> dict:
    """Return a minimal W7-X finite-section GA policy."""
    return {
        "resource_caps": {
            "max_total_iterations": 1000,
            "timeout_minutes_max": 45,
        },
        "mutation": {
            "surfaces": ["wout_w7x.nc"],
            "algorithms": ["L-BFGS-B"],
            "threshold_sigma": 0.1,
            "structural_mutation_prob": 0.0,
            "ncoils_choices": [4, 5, 6],
            "order_choices": [4],
            "max_iterations": 120,
            "dof_perturbation": 0.01,
        },
        "finite_section_field": {
            "enabled": True,
            "width": 0.10,
            "height": 0.10,
            "n_width": 3,
            "n_height": 3,
            "current_distribution": "uniform",
        },
        "fourier_continuation": {"enabled": False, "orders": [4]},
        "exploration": {
            "use_default_thresholds": False,
            "surfaces": ["wout_w7x.nc"],
            "algorithms": ["L-BFGS-B"],
            "ncoils_choices": [4, 5, 6],
            "order_choices": [4],
            "max_iterations": 120,
            "dof_perturbation": 0.01,
            "length_threshold_range": [140.0, 185.0],
            "cc_threshold_range": [1.10, 1.35],
            "cs_threshold_range": [1.10, 1.50],
            "curvature_threshold_range": [0.8, 1.3],
            "msc_threshold_range": [0.8, 1.6],
            "include_torsion": True,
            "torsion_threshold_range": [2.0, 25.0],
        },
    }


def test_explore_case_includes_finite_section_l_bfgs_and_seed() -> None:
    """Exploration cases inherit the W7-X finite-section policy block."""
    case = explore_case(_w7x_finite_section_policy(), _rng(1))
    cfg = case["case_config"]

    assert cfg["surface_params"]["surface"] == "wout_w7x.nc"
    assert cfg["optimizer_params"]["algorithm"] == "L-BFGS-B"
    assert cfg["optimizer_params"]["max_iterations"] == 120
    assert cfg["finite_section_field"]["enabled"] is True
    assert cfg["finite_section_field"]["n_width"] == 3
    assert cfg["finite_section_field"]["n_height"] == 3
    assert cfg["dof_perturbation"] == 0.01
    assert isinstance(case["random_seed"], int)


def test_mutate_case_preserves_finite_section_and_policy_algorithm() -> None:
    """Mutation cases apply finite-section and policy-selected optimizer."""
    parent = {
        "case_id": "parent",
        "case_config": {
            "description": "parent",
            "surface_params": {
                "surface": "input.LandremanPaul2021_QH_reactorScale_lowres",
                "range": "half period",
            },
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": {
                "algorithm": "augmented_lagrangian",
                "max_iterations": 20,
            },
            "coil_objective_terms": {
                "length_threshold": 151.7,
                "cc_threshold": 1.20,
                "cs_threshold": 1.18,
            },
            "fourier_continuation": {"enabled": True, "orders": [4, 8, 16]},
        },
    }

    case = mutate_case(parent, _w7x_finite_section_policy(), _rng(2))
    cfg = case["case_config"]

    assert cfg["surface_params"]["surface"] == "wout_w7x.nc"
    assert cfg["optimizer_params"]["algorithm"] == "L-BFGS-B"
    assert cfg["optimizer_params"]["max_iterations"] == 120
    assert cfg["finite_section_field"]["enabled"] is True
    assert cfg["dof_perturbation"] == 0.01
    assert "fourier_continuation" not in cfg
    assert case["parent_ids"] == ["parent"]
    assert isinstance(case["random_seed"], int)
