"""Shared fixtures for validate_config tests."""

from __future__ import annotations


def _base_config() -> dict:
    """Base config for edge case tests."""
    return {
        "description": "Test",
        "surface_params": {"surface": "input.LandremanPaul2021_QA"},
        "coils_params": {"ncoils": 4, "order": 16},
        "optimizer_params": {"algorithm": "l-bfgs"},
    }


def _merge(base: dict, updates: dict) -> dict:
    """Deep merge updates into base (shallow for nested dicts)."""
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


_EDGE_CASE_PARAMS = [
    (
        {"surface_params": {"virtual_casing": "yes"}},
        ["virtual_casing must be a boolean"],
    ),
    (
        {"coil_objective_terms": {"length_threshold": True}},
        ["must be a non-negative number"],
    ),
    (
        {"coil_objective_terms": {"length_threshold": [1, 2]}},
        ["must be a non-negative number"],
    ),
    (
        {"coil_objective_terms": {"length_threshold": -5}},
        ["must be a non-negative number"],
    ),
    ({"coil_objective_terms": {"curvature_p": True}}, ["must be a positive number"]),
    ({"coil_objective_terms": {"curvature_p": "abc"}}, ["must be a positive number"]),
    ({"coil_objective_terms": {"curvature_p": "0"}}, ["must be a positive number"]),
    (
        {"coil_objective_terms": {"coil_curvature": "invalid"}},
        ["coil_curvature must be one of"],
    ),
    (
        {"coil_objective_terms": {"coil_arclength_variation": "invalid"}},
        ["coil_arclength_variation", "must be one of"],
    ),
    (
        {"coil_objective_terms": {"coil_mean_squared_curvature": "invalid"}},
        ["coil_mean_squared_curvature must be one of"],
    ),
    (
        {"coil_objective_terms": {"linking_number": "soft"}},
        ["linking_number must be one of"],
    ),
    (
        {"coil_objective_terms": {"coil_coil_force": "invalid"}},
        ["coil_coil_force must be one of"],
    ),
    (
        {"coil_objective_terms": {"coil_coil_torque": "invalid"}},
        ["coil_coil_torque must be one of"],
    ),
    (
        {"coil_objective_terms": {"coil_torsion": "invalid"}},
        ["coil_torsion must be one of"],
    ),
    (
        {"fourier_continuation": "not a dict"},
        ["fourier_continuation must be a dictionary"],
    ),
    ({"fourier_continuation": {"enabled": "yes"}}, ["enabled must be a boolean"]),
    ({"fourier_continuation": {"orders": "3,5"}}, ["orders must be a list"]),
    ({"fourier_continuation": {"orders": []}}, ["must be non-empty"]),
    ({"fourier_continuation": {"orders": [0, 1, 2]}}, ["positive integers"]),
    ({"fourier_continuation": {"orders": [5, 3]}}, ["ascending order"]),
    ({"coil_objective_terms": {"cc_weight": False}}, ["must be a non-negative number"]),
    (
        {"coil_objective_terms": {"structural_use_cached_K": 1}},
        ["structural_use_cached_K must be a boolean"],
    ),
    (
        {"coil_objective_terms": {"structural_penalty_margin": 0.9}},
        ["Unknown coil_objective_terms key"],
    ),
]


_YAML_INVALID_CASE_PARAMS = [
    (
        """description: Bad case
surface_params:
  surface: nonexistent_surface_xyz
  range: half period
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
""",
        ["nonexistent_surface_xyz", "plasma_surfaces"],
    ),
    (
        """description: Bad range
surface_params:
  surface: input.LandremanPaul2021_QA
  range: invalid_range
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
""",
        ["range", "half period"],
    ),
    (
        """description: Bad ncoils
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: -1
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
""",
        ["ncoils"],
    ),
    (
        """description: Float ncoils
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4.0
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
""",
        ["ncoils", "integer"],
    ),
    (
        """description: Zero order
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4
  order: 0
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
""",
        ["order"],
    ),
    (
        """description: Bad max_iterations
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 0
""",
        ["max_iterations"],
    ),
    (
        """description: No ncoils
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
""",
        ["ncoils"],
    ),
    (
        """description: Bad objective term
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
coil_objective_terms:
  total_length: invalid_option
""",
        ["total_length", "must be one of"],
    ),
    (
        """description: Negative threshold
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
coil_objective_terms:
  length_threshold: -5.0
""",
        ["non-negative", "length_threshold"],
    ),
    (
        """description: Bad fourier_continuation
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
fourier_continuation:
  enabled: true
  orders: "4,8,16"
""",
        ["orders", "list"],
    ),
    (
        """description: Unsorted fourier_continuation orders
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
fourier_continuation:
  enabled: true
  orders: [8, 4, 16]
""",
        ["ascending"],
    ),
    (
        """description: Bad surface_params type
surface_params: "not a dict"
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
""",
        ["surface_params", "dictionary"],
    ),
]
