"""GA utilities: RNG, hashing, mutation, and exploration operators."""

from __future__ import annotations

import hashlib
import json
import math
import random as _random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    from stellcoilbench.path_utils import get_surface_filename
except ImportError:
    _repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_repo / "src"))
    from stellcoilbench.path_utils import get_surface_filename

__all__ = [
    "_rng",
    "_log_uniform",
    "_clamp",
    "_new_case_id",
    "_config_hash_short",
    "mutate_case",
    "explore_case",
]


def _rng(seed: int | None = None) -> _random.Random:
    """Return a seeded Random instance."""
    return _random.Random(seed)


def _log_uniform(rng: _random.Random, lo: float, hi: float) -> float:
    """Sample from a log-uniform distribution in [lo, hi]."""
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _new_case_id() -> str:
    """Generate a unique case_id string with timestamp + random suffix."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = _random.randint(10000, 99999)
    return f"{ts}_{suffix}"


def _config_hash_short(cfg: Dict[str, Any]) -> str:
    """Return short hash of config for novelty checking."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def mutate_case(
    parent: Dict[str, Any],
    policy: Dict[str, Any],
    rng: _random.Random,
    *,
    safe: bool = False,
) -> Dict[str, Any]:
    """Create a mutated child case from a parent's case_config."""
    import copy

    parent_cfg = parent.get("case_config", {})
    child_cfg = copy.deepcopy(parent_cfg)

    mut = policy.get("mutation", {})

    t_sigma = float(mut.get("threshold_sigma", 0.10))

    opt = child_cfg.get("optimizer_params", {})
    opt["algorithm"] = "augmented_lagrangian"

    obj = child_cfg.get("coil_objective_terms", {})
    if obj is None:
        obj = {}
    weight_keys = [k for k in obj if k.endswith("_weight")]
    for wk in weight_keys:
        del obj[wk]

    threshold_keys = [k for k in obj if k.endswith("_threshold")]
    if not threshold_keys:
        _REACTOR_DEFAULTS = {
            "length_threshold": 180.0,
            "cc_threshold": 0.8,
            "cs_threshold": 1.3,
            "msc_threshold": 1.0,
            "curvature_threshold": 1.0,
            "force_threshold": 1.0,
            "torque_threshold": 1.0,
            "torsion_threshold": 1.0,
        }
        for tname, default_val in _REACTOR_DEFAULTS.items():
            obj[tname] = default_val
        threshold_keys = [k for k in obj if k.endswith("_threshold")]

    for tk in threshold_keys:
        old = obj[tk]
        if isinstance(old, (int, float)) and not isinstance(old, bool) and old > 0:
            new_val = old * math.exp(rng.gauss(0, t_sigma))
            obj[tk] = round(max(1e-6, new_val), 6)

    child_cfg["coil_objective_terms"] = obj

    struct_prob = float(mut.get("structural_mutation_prob", 0.2))
    coils_params = child_cfg.get("coils_params", {})

    if rng.random() < struct_prob:
        ncoils_choices = mut.get("ncoils_choices", [3, 4, 5, 6, 7])
        current_ncoils = coils_params.get("ncoils", 4)
        adjacent = [n for n in ncoils_choices if abs(n - current_ncoils) == 1]
        if adjacent:
            coils_params["ncoils"] = rng.choice(adjacent)
        elif len(ncoils_choices) > 1:
            others = [n for n in ncoils_choices if n != current_ncoils]
            coils_params["ncoils"] = rng.choice(others)

    if rng.random() < struct_prob:
        order_choices = mut.get("order_choices", [4, 6, 8])
        current_order = coils_params.get("order", 4)
        adjacent = [
            o
            for o in order_choices
            if abs(o - current_order) <= 2 and o != current_order
        ]
        if adjacent:
            coils_params["order"] = rng.choice(adjacent)
        elif len(order_choices) > 1:
            others = [o for o in order_choices if o != current_order]
            coils_params["order"] = rng.choice(others)

    child_cfg["coils_params"] = coils_params

    surface = get_surface_filename(child_cfg) or "unknown"
    ncoils = coils_params.get("ncoils", 4)
    order = coils_params.get("order", 4)
    child_cfg["description"] = f"Mutation: {surface} ncoils={ncoils} order={order}"

    fc = policy.get("fourier_continuation", {})
    if fc and fc.get("enabled") and fc.get("orders"):
        child_cfg["fourier_continuation"] = {
            "enabled": True,
            "orders": list(fc["orders"]),
        }

    mut_policy = policy.get("mutation", {})
    opt["max_iterations"] = mut_policy.get("max_iterations", 1000)
    opt["verbose"] = True
    opt.pop("max_iter_subopt", None)
    child_cfg["optimizer_params"] = opt

    dof_perturbation = float(mut.get("dof_perturbation", 0.01))
    if dof_perturbation > 0:
        child_cfg["dof_perturbation"] = dof_perturbation

    new_seed = rng.randint(0, 2**31 - 1)

    caps = policy.get("resource_caps", {})
    max_iter = opt.get("max_iterations", 2000)
    resource = {
        "max_total_iterations": min(max_iter, caps.get("max_total_iterations", 10000)),
        "timeout_minutes": caps.get("timeout_minutes_max", 60),
    }

    return {
        "case_id": _new_case_id(),
        "parent_ids": [parent.get("case_id", "unknown")],
        "tags": ["exploit"],
        "proposer_mode": "ga",
        "resource": resource,
        "case_config": child_cfg,
        "random_seed": new_seed,
    }


def explore_case(
    policy: Dict[str, Any],
    rng: _random.Random,
    *,
    safe: bool = False,
) -> Dict[str, Any]:
    """Generate a random exploration case from the policy parameter ranges."""
    expl = policy.get("exploration", {})
    sm = policy.get("safe_mode", {})

    if safe:
        surfaces = sm.get(
            "preferred_surfaces", expl.get("surfaces", ["input.LandremanPaul2021_QA"])
        )
    else:
        surfaces = expl.get("surfaces", ["input.LandremanPaul2021_QA"])
    surface = rng.choice(surfaces)

    algorithms = expl.get("algorithms", ["augmented_lagrangian"])
    algorithm = rng.choice(algorithms)

    ncoils = rng.choice(expl.get("ncoils_choices", [4]))
    order = rng.choice(expl.get("order_choices", [8]))

    max_iterations = expl.get("max_iterations", 1000)

    coil_objective_terms: Dict[str, Any] = {
        "total_length": "l2_threshold",
        "coil_curvature": "lp_threshold",
        "coil_curvature_p": 2,
        "coil_mean_squared_curvature": "l2_threshold",
        "coil_arclength_variation": "l2_threshold",
        "linking_number": "",
    }

    include_force = expl.get("include_force", False)
    if include_force:
        coil_objective_terms["coil_coil_force"] = "lp_threshold"
    include_torque = expl.get("include_torque", False)
    if include_torque:
        coil_objective_terms["coil_coil_torque"] = "lp_threshold"
    include_torsion = expl.get("include_torsion", False)
    if include_torsion:
        coil_objective_terms["coil_torsion"] = "lp_threshold"
        coil_objective_terms["coil_torsion_p"] = 2

    use_defaults = expl.get("use_default_thresholds", True)
    if not use_defaults:
        coil_objective_terms["length_threshold"] = round(
            _log_uniform(rng, *expl.get("length_threshold_range", [100, 300])), 2
        )
        coil_objective_terms["cc_threshold"] = round(
            _log_uniform(rng, *expl.get("cc_threshold_range", [0.4, 1.5])), 3
        )
        coil_objective_terms["cs_threshold"] = round(
            _log_uniform(rng, *expl.get("cs_threshold_range", [0.5, 2.5])), 3
        )
        coil_objective_terms["curvature_threshold"] = round(
            _log_uniform(rng, *expl.get("curvature_threshold_range", [0.5, 5.0])), 3
        )
        coil_objective_terms["msc_threshold"] = round(
            _log_uniform(rng, *expl.get("msc_threshold_range", [0.1, 5.0])), 3
        )
        if include_force:
            coil_objective_terms["force_threshold"] = round(
                _log_uniform(rng, *expl.get("force_threshold_range", [50, 500])), 1
            )
        if include_torque:
            coil_objective_terms["torque_threshold"] = round(
                _log_uniform(rng, *expl.get("torque_threshold_range", [1, 500])), 1
            )
        if include_torsion:
            coil_objective_terms["torsion_threshold"] = round(
                _log_uniform(rng, *expl.get("torsion_threshold_range", [0.2, 5.0])), 3
            )

    case_config: Dict[str, Any] = {
        "description": f"Exploration case: {surface} ncoils={ncoils} order={order}",
        "surface_params": {
            "surface": surface,
            "range": "half period",
        },
        "coils_params": {
            "ncoils": ncoils,
            "order": order,
        },
        "optimizer_params": {
            "algorithm": algorithm,
            "max_iterations": max_iterations,
            "verbose": True,
        },
        "coil_objective_terms": coil_objective_terms,
    }

    dof_perturbation = float(expl.get("dof_perturbation", 0.0))
    if dof_perturbation > 0:
        case_config["dof_perturbation"] = dof_perturbation

    fc = policy.get("fourier_continuation", {})
    if fc and fc.get("enabled") and fc.get("orders"):
        case_config["fourier_continuation"] = {
            "enabled": True,
            "orders": list(fc["orders"]),
        }

    new_seed = rng.randint(0, 2**31 - 1)
    caps = policy.get("resource_caps", {})
    resource = {
        "max_total_iterations": min(
            max_iterations, caps.get("max_total_iterations", 10000)
        ),
        "timeout_minutes": caps.get("timeout_minutes_max", 60),
    }

    return {
        "case_id": _new_case_id(),
        "parent_ids": [],
        "tags": ["explore"],
        "proposer_mode": "ga",
        "resource": resource,
        "case_config": case_config,
        "random_seed": new_seed,
    }
