#!/usr/bin/env python3
"""Run policy-defined StellCoilBench Simsopt/FOCUS batch scans.

The policy owns the surface, queues, thresholds, weights, and output naming.
This runner generates StellCoilBench-compatible case.yaml files for Simsopt
and FOCUS backends, runs them in parallel, and records unified benchmark
metrics in a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SURFACE = "plasma_surfaces/wout_20260324.nc"
RESULTS_DIR = Path("results/simsopt_batch")
FOCUS_EXECUTABLE = "/home/jiangxm/.local/bin/xfocus"
DEFAULT_POLICY = Path("policy/simsopt_batch_policy.yaml")

TARGETS = {
    "avg_BdotN_over_B": 0.005,
    "avg_BdotN_over_target_B": None,
    "final_min_cc_separation": 0.25,
    "final_min_cs_separation": 0.25,
    "final_max_curvature": 5.0,
    "final_max_torsion": 15.0,
}

CSV_FIELDS = [
    "run_id",
    "backend",
    "status",
    "success",
    "failure_reason",
    "wave",
    "family",
    "init_family",
    "policy_family",
    "current_family",
    "geometry_weight_scale",
    "policy_label",
    "random_seed",
    "order",
    "algorithm",
    "max_iterations",
    "avg_BdotN_over_B",
    "max_BdotN_over_B",
    "avg_BdotN_over_target_B",
    "max_BdotN_over_target_B",
    "initial_R0",
    "initial_R1",
    "requested_major_radius_scale",
    "requested_minor_radius_scale",
    "requested_radial_offset",
    "requested_current_scale",
    "initial_min_cc_separation",
    "initial_min_cs_separation",
    "initial_total_length",
    "initial_length_variance",
    "initial_linking_number",
    "final_min_cc_separation",
    "final_min_cs_separation",
    "final_total_length",
    "final_max_length",
    "final_min_length",
    "final_mean_coil_length",
    "final_length_variance",
    "final_length_std",
    "final_length_cv",
    "final_length_ratio",
    "final_max_curvature",
    "final_mean_squared_curvature",
    "final_arclength_variation",
    "final_max_torsion",
    "final_linking_number",
    "optimization_time",
    "walltime_sec",
    "target_avg_BdotN_over_B",
    "target_avg_BdotN_over_target_B",
    "target_final_min_cc_separation",
    "target_final_min_cs_separation",
    "target_final_max_curvature",
    "target_final_max_torsion",
    "meets_targets",
    "run_dir",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--surface", default=SURFACE)
    parser.add_argument("--focus-executable", default=FOCUS_EXECUTABLE)
    parser.add_argument("--surface-resolution", type=int, default=32)
    parser.add_argument("--jobs-per-backend", type=int, default=None)
    parser.add_argument("--backend", choices=["both", "simsopt", "focus"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="Run only one generated job by zero-based index after backend filtering.",
    )
    parser.add_argument(
        "--job-indexes",
        default=None,
        help="Run selected zero-based job indexes, e.g. '0,2,5-8', after backend filtering.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--data-twin-campaign",
        default=None,
        help="Required for non-dry-run launches unless --allow-unregistered-launch is used.",
    )
    parser.add_argument(
        "--allow-unregistered-launch",
        action="store_true",
        help="Explicitly bypass the Data Twin launch gate. Use only for emergency/manual runs.",
    )
    parser.add_argument("--max-parallel-simsopt", type=int, default=None)
    parser.add_argument("--max-parallel-focus", type=int, default=None)
    parser.add_argument(
        "--submit-batch-size",
        type=int,
        default=None,
        help="Maximum queued futures per backend group. Defaults to 2 * max_parallel.",
    )
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _parse_job_indexes(value: str | None) -> set[int]:
    if not value:
        return set()
    indexes: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending job-index range: {part}")
            indexes.update(range(start, end + 1))
        else:
            indexes.add(int(part))
    return indexes


def _load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _policy_targets(policy: dict[str, Any] | None) -> dict[str, float]:
    """Return engineering screening targets, allowing policy-level overrides."""
    targets = dict(TARGETS)
    policy = policy or {}
    configured_targets = [
        policy.get("targets", {}) or {},
        (policy.get("common", {}) or {}).get("targets", {}) or {},
    ]
    for configured in configured_targets:
        for key, value in configured.items():
            if key in targets and value is not None:
                targets[key] = float(value)
    return targets


def _as_list(value: Any, default: list[Any] | None = None) -> list[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return value
    return [value]


def _grid_product(grid: dict[str, Any], keys: list[str]) -> list[dict[str, Any]]:
    values = [_as_list(grid[key]) if key in grid else [None] for key in keys]
    return [dict(zip(keys, item)) for item in itertools.product(*values)]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _current_initialization_from_family(
    current_family: str | None,
    ncoils: int,
) -> dict[str, Any]:
    """Map named current-initialization families to concrete initial values."""
    family = str(current_family or "uniform")
    if family == "uniform":
        return {}
    if family.startswith("scaled_uniform_"):
        scale_text = family.removeprefix("scaled_uniform_").replace("p", ".")
        return {"current_scale": float(scale_text)}
    if family == "mild_asymmetry":
        return {
            "current_weights": [
                1.05 if idx % 2 == 0 else 0.95 for idx in range(ncoils)
            ]
        }
    if family == "outer_inner_bias":
        if ncoils == 1:
            return {"current_weights": [1.0]}
        return {
            "current_weights": [
                0.9 + 0.2 * idx / (ncoils - 1) for idx in range(ncoils)
            ]
        }
    if family in {"warm_parent_current", "current_fixed_shape_free", "current_free"}:
        return {}
    return {}


def _write_yaml_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(_json_safe(data), indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _enforce_data_twin_launch_gate(
    campaign_id: str | None,
    *,
    allow_unregistered_launch: bool,
    expected_jobs: int,
    policy_path: Path,
    results_dir: Path,
) -> None:
    if allow_unregistered_launch:
        print(
            "WARNING: bypassing Data Twin launch gate via --allow-unregistered-launch",
            flush=True,
        )
        return
    if not campaign_id:
        raise SystemExit(
            "Refusing non-dry-run launch without --data-twin-campaign. "
            "Register planned cases/runs first, or pass --allow-unregistered-launch explicitly."
        )

    root = Path("experiments/data_twin") / campaign_id
    if not root.exists():
        raise SystemExit(f"Refusing launch: Data Twin campaign does not exist: {root}")

    cases = _read_jsonl(root / "cases.jsonl")
    runs = _read_jsonl(root / "runs.jsonl")
    events = _read_jsonl(root / "events.jsonl")
    planned_cases = [row for row in cases if row.get("case_id") != "campaign"]
    planned_runs = [row for row in runs if row.get("run_id") != "campaign"]
    if len(planned_cases) < expected_jobs or len(planned_runs) < expected_jobs:
        raise SystemExit(
            "Refusing launch: Data Twin campaign is not fully registered "
            f"({len(planned_cases)} cases, {len(planned_runs)} runs, expected {expected_jobs})."
        )

    board_events = [row for row in events if row.get("event_type") == "board_registered"]
    if not board_events:
        raise SystemExit("Refusing launch: Data Twin campaign has no board_registered event.")
    latest = board_events[-1]
    metadata = latest.get("metadata") or {}
    registered_policy = metadata.get("policy")
    registered_results = metadata.get("results_dir")
    if registered_policy and Path(str(registered_policy)) != policy_path:
        raise SystemExit(
            "Refusing launch: Data Twin registered policy does not match launch policy "
            f"({registered_policy} != {policy_path})."
        )
    if registered_results and Path(str(registered_results)) != results_dir:
        raise SystemExit(
            "Refusing launch: Data Twin registered results_dir does not match launch results_dir "
            f"({registered_results} != {results_dir})."
        )


def _base_case(
    surface: str,
    order: int,
    max_iterations: int,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    common = (policy or {}).get("common", {})
    thresholds = common.get("thresholds", {})
    optimizer_params = {
        "max_iterations": max_iterations,
        "verbose": True,
        "history_interval": 25,
    }
    if "plot_upsample_factor" in common:
        optimizer_params["plot_upsample_factor"] = int(common["plot_upsample_factor"])

    surface_params = {
        "surface": surface,
        "range": common.get("surface_range", "half period"),
    }
    if common.get("target_B") is not None:
        surface_params["target_B"] = float(common["target_B"])
    if common.get("virtual_casing") is not None:
        surface_params["virtual_casing"] = bool(common["virtual_casing"])

    return {
        "description": common.get("description", "Policy-defined Simsopt batch scan"),
        "surface_params": surface_params,
        "coils_params": {
            "coil_type": "modular",
            "ncoils": int(common.get("ncoils", 4)),
            "order": order,
            "numquadpoints": int(common.get("coil_quadpoints", 256)),
        },
        "optimizer_params": optimizer_params,
        "coil_objective_terms": {
            "total_length": "l2_threshold",
            "coil_curvature": "lp_threshold",
            "coil_curvature_p": 2,
            "coil_mean_squared_curvature": "l2_threshold",
            "coil_arclength_variation": "l2_threshold",
            "coil_torsion": "lp_threshold",
            "coil_torsion_p": 2,
            "linking_number": "",
            "length_threshold": thresholds.get("length_threshold", 12.0),
            "cc_threshold": thresholds.get("cc_threshold", 0.25),
            "cs_threshold": thresholds.get("cs_threshold", 0.25),
            "curvature_threshold": thresholds.get("curvature_threshold", 5.0),
            "torsion_threshold": thresholds.get("torsion_threshold", 15.0),
            "msc_threshold": thresholds.get("msc_threshold", 20.0),
        },
        "post_processing_params": {
            "run_vmec": False,
            "plot_poincare": False,
            "plot_boozer": False,
            "run_simple": False,
        },
    }


def _simsopt_jobs(
    surface: str,
    jobs_per_backend: int | None,
    smoke: bool,
    policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if smoke:
        combos = [
            {
                "queue": "smoke",
                "algorithm": "L-BFGS-B",
                "order": 4,
                "max_iterations": 30,
                "flux_weight": 10.0,
                "cc_weight": 30.0,
                "cs_weight": 30.0,
                "curvature_weight": 1.0,
                "torsion_weight": 1.0,
            }
        ]
    elif policy and policy.get("simsopt_queues"):
        combos = []
        keys = [
            "surface",
            "ncoils",
            "algorithm",
            "order",
            "max_iterations",
            "flux_weight",
            "cc_weight",
            "cs_weight",
            "curvature_weight",
            "torsion_weight",
            "length_weight",
            "length_variance_weight",
            "msc_weight",
            "arclength_variation_weight",
            "linking_weight",
            "length_threshold",
            "length_threshold_device",
            "cc_threshold",
            "cc_threshold_device",
            "cs_threshold",
            "cs_threshold_device",
            "curvature_threshold",
            "curvature_threshold_device",
            "torsion_threshold",
            "torsion_threshold_device",
            "msc_threshold",
            "msc_threshold_device",
            "arclength_variation_threshold",
            "arclength_variation_threshold_device",
            "length_variance_threshold",
            "length_variance_threshold_device",
            "force_threshold_device",
            "torque_threshold_device",
            "link_guard",
            "link_guard_interval",
            "link_guard_penalty",
            "link_guard_tolerance",
            "link_guard_rollback",
            "link_guard_sample_stride",
            "link_guard_record_interval",
            "cs_guard",
            "cs_guard_interval",
            "cs_guard_hard_min",
            "cs_guard_soft_min",
            "cs_guard_penalty",
            "cs_guard_rollback",
            "early_stop",
            "dof_perturbation",
            "initial_coils_path",
            "parent_run_id",
            "parent_case_id",
            "init_family",
            "policy_family",
            "current_family",
            "geometry_weight_scale",
            "major_radius_scale",
            "minor_radius_scale",
            "radial_offset",
            "current_scale",
            "current_weights",
            "random_seed",
            "family",
            "policy_label",
            "wave",
        ]
        for queue_name, queue in policy["simsopt_queues"].items():
            for combo in _grid_product(queue.get("grid", {}), keys):
                combo["queue"] = queue_name
                combo.update(queue.get("fixed", {}))
                combos.append(combo)
    else:
        combos = [
            {
                "queue": "legacy",
                "algorithm": algorithm,
                "order": order,
                "max_iterations": max_iter,
                "flux_weight": flux_w,
                "cc_weight": cc_w,
                "cs_weight": cs_w,
                "curvature_weight": curv_w,
                "torsion_weight": 1.0,
            }
            for algorithm, order, max_iter, flux_w, cc_w, cs_w, curv_w in itertools.product(
                ["L-BFGS-B", "augmented_lagrangian"],
                [4, 6, 8],
                [120, 240],
                [5.0, 15.0, 40.0],
                [20.0, 80.0],
                [20.0, 80.0],
                [0.5, 2.0],
            )
        ]
    jobs: list[dict[str, Any]] = []
    run_prefix = str((policy or {}).get("common", {}).get("run_prefix", "simsopt_batch"))
    for idx, combo in enumerate(combos):
        if jobs_per_backend is not None and idx >= jobs_per_backend:
            break
        algorithm = combo["algorithm"]
        order = int(combo["order"])
        max_iter = int(combo["max_iterations"])
        queue = str(combo.get("queue", "simsopt"))
        combo_surface = str(combo.get("surface") or surface)
        case = _base_case(combo_surface, order, max_iter, policy)
        if combo.get("ncoils") is not None:
            case["coils_params"]["ncoils"] = int(combo["ncoils"])
        case["optimizer_params"]["backend"] = "simsopt"
        case["optimizer_params"]["algorithm"] = algorithm
        if algorithm == "augmented_lagrangian":
            case["optimizer_params"]["max_iter_subopt"] = int(
                combo.get("max_iter_subopt", 10)
            )
        if combo.get("dof_perturbation") is not None:
            case["dof_perturbation"] = float(combo["dof_perturbation"])
        if combo.get("initial_coils_path") is not None:
            case["coils_params"]["initial_coils_path"] = str(combo["initial_coils_path"])
        current_defaults = _current_initialization_from_family(
            combo.get("current_family"), int(case["coils_params"].get("ncoils", 4))
        )
        for coil_key, value in current_defaults.items():
            case["coils_params"].setdefault(coil_key, value)
        for coil_key in (
            "init_family",
            "current_family",
            "major_radius_scale",
            "minor_radius_scale",
            "radial_offset",
            "current_scale",
            "current_weights",
        ):
            if combo.get(coil_key) is not None:
                case["coils_params"][coil_key] = combo[coil_key]
        if combo.get("random_seed") is not None:
            case["random_seed"] = int(combo["random_seed"])
        metadata = {
            key: combo.get(key)
            for key in (
                "queue",
                "family",
                "init_family",
                "policy_family",
                "current_family",
                "geometry_weight_scale",
                "policy_label",
                "wave",
                "parent_run_id",
                "parent_case_id",
                "surface",
                "ncoils",
            )
            if combo.get(key) is not None
        }
        if metadata:
            case["experiment_metadata"] = metadata
        terms = case["coil_objective_terms"]
        terms["flux_weight"] = float(combo["flux_weight"])
        terms["cc_weight"] = float(combo["cc_weight"])
        terms["cs_weight"] = float(combo["cs_weight"])
        terms["curvature_weight"] = float(combo["curvature_weight"])
        terms["torsion_weight"] = float(combo.get("torsion_weight", 1.0))
        for weight_key in (
            "length_weight",
            "length_variance_weight",
            "msc_weight",
            "arclength_variation_weight",
            "linking_weight",
        ):
            if weight_key in combo and combo[weight_key] is not None:
                terms[weight_key] = float(combo[weight_key])
        if float(terms.get("length_variance_weight", 0.0) or 0.0) > 0.0:
            terms["coil_length_variance"] = "l1"
        for threshold_key in (
            "length_threshold",
            "length_threshold_device",
            "cc_threshold",
            "cc_threshold_device",
            "cs_threshold",
            "cs_threshold_device",
            "curvature_threshold",
            "curvature_threshold_device",
            "torsion_threshold",
            "torsion_threshold_device",
            "msc_threshold",
            "msc_threshold_device",
            "arclength_variation_threshold",
            "arclength_variation_threshold_device",
            "length_variance_threshold",
            "length_variance_threshold_device",
            "force_threshold_device",
            "torque_threshold_device",
        ):
            if threshold_key in combo and combo[threshold_key] is not None:
                terms[threshold_key] = float(combo[threshold_key])
        for bool_key in ("link_guard", "link_guard_rollback"):
            if combo.get(bool_key) is not None:
                terms[bool_key] = bool(combo[bool_key])
        for bool_key in ("cs_guard", "cs_guard_rollback"):
            if combo.get(bool_key) is not None:
                terms[bool_key] = bool(combo[bool_key])
        for int_key in (
            "link_guard_interval",
            "link_guard_sample_stride",
            "link_guard_record_interval",
            "cs_guard_interval",
        ):
            if combo.get(int_key) is not None:
                terms[int_key] = int(combo[int_key])
        for link_key in ("link_guard_penalty", "link_guard_tolerance"):
            if combo.get(link_key) is not None:
                terms[link_key] = float(combo[link_key])
        for cs_key in (
            "cs_guard_hard_min",
            "cs_guard_soft_min",
            "cs_guard_penalty",
        ):
            if combo.get(cs_key) is not None:
                terms[cs_key] = float(combo[cs_key])
        if isinstance(combo.get("early_stop"), dict):
            terms["early_stop"] = dict(combo["early_stop"])
        jobs.append(
            {
                "backend": "simsopt",
                "run_id": (
                    f"{run_prefix}_{queue}_simsopt_{idx:04d}_"
                    f"{algorithm.lower().replace('-', '').replace('_', '')}_"
                    f"o{order}_it{max_iter}"
                ),
                "case": case,
            }
        )
    return jobs


def _focus_jobs(
    surface: str,
    executable: str,
    jobs_per_backend: int | None,
    smoke: bool,
    policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if smoke:
        combos = [
            {
                "queue": "smoke",
                "order": 4,
                "max_iterations": 30,
                "weight_bnorm": 30.0,
                "weight_ccsep": 20.0,
                "weight_cssep": 20.0,
                "weight_curv": 0.5,
                "target_length": 0.0,
                "init_radius": 0.5,
            }
        ]
    elif policy and policy.get("focus_queues"):
        combos = []
        keys = [
            "order",
            "max_iterations",
            "weight_bnorm",
            "weight_ccsep",
            "weight_cssep",
            "weight_curv",
            "target_length",
            "init_radius",
        ]
        for queue_name, queue in policy["focus_queues"].items():
            optional_keys = [
                "ccsep_alpha",
                "ccsep_beta",
                "cssep_factor",
                "curv_k0",
                "weight_ttlen",
            ]
            for combo in _grid_product(queue.get("grid", {}), keys):
                combo["queue"] = queue_name
                for key in optional_keys:
                    if key in queue.get("grid", {}):
                        # Expand optional keys separately by multiplying current combos.
                        pass
                combo.update(queue.get("fixed", {}))
                combos.append(combo)
            for key in optional_keys:
                if key in queue.get("grid", {}):
                    values = _as_list(queue["grid"][key])
                    expanded = []
                    for combo in combos:
                        if combo.get("queue") != queue_name:
                            expanded.append(combo)
                            continue
                        for value in values:
                            new_combo = dict(combo)
                            new_combo[key] = value
                            expanded.append(new_combo)
                    combos = expanded
    else:
        combos = [
            {
                "queue": "legacy",
                "order": order,
                "max_iterations": max_iter,
                "weight_bnorm": bnorm_w,
                "weight_ccsep": cc_w,
                "weight_cssep": cs_w,
                "weight_curv": curv_w,
                "target_length": target_len,
                "init_radius": init_radius,
            }
            for order, max_iter, bnorm_w, cc_w, cs_w, curv_w, target_len, init_radius in itertools.product(
                [4, 6, 8],
                [80, 160],
                [10.0, 50.0, 150.0],
                [5.0, 20.0, 80.0],
                [5.0, 20.0, 80.0],
                [0.2, 1.0, 5.0],
                [0.0, 3.0, 4.0],
                [0.45, 0.60, 0.80],
            )
        ]
    jobs: list[dict[str, Any]] = []
    run_prefix = str((policy or {}).get("common", {}).get("run_prefix", "simsopt_batch"))
    for idx, combo in enumerate(combos):
        if jobs_per_backend is not None and idx >= jobs_per_backend:
            break
        order = int(combo["order"])
        max_iter = int(combo["max_iterations"])
        bnorm_w = float(combo["weight_bnorm"])
        init_radius = float(combo["init_radius"])
        queue = str(combo.get("queue", "focus"))
        run_stem = f"focus_{idx:03d}_o{order}_it{max_iter}"
        case = _base_case(surface, order, max_iter, policy)
        case["optimizer_params"]["backend"] = "focus"
        case["optimizer_params"]["algorithm"] = "focus"
        case["focus_params"] = {
            "executable": executable,
            "run_stem": run_stem,
            "timeout_seconds": 7200,
            "nfp": 4,
            "stellsym": True,
            "weight_bnorm": bnorm_w,
            "weight_ccsep": float(combo["weight_ccsep"]),
            "weight_cssep": float(combo["weight_cssep"]),
            "weight_curv": float(combo["weight_curv"]),
            "weight_ttlen": float(combo.get("weight_ttlen", 1.0)),
            "target_length": float(combo["target_length"]),
            "init_radius": init_radius,
            "curv_k0": float(combo.get("curv_k0", 5.0)),
            "case_postproc": 0,
            "parser": "auto",
        }
        for key in ("ccsep_alpha", "ccsep_beta", "cssep_factor"):
            if key in combo:
                case["focus_params"][key] = float(combo[key])
        jobs.append(
            {
                "backend": "focus",
                "run_id": (
                    f"{run_prefix}_{queue}_focus_{idx:04d}_o{order}_it{max_iter}"
                    f"_bw{str(bnorm_w).replace('.', 'p')}"
                    f"_r{str(init_radius).replace('.', 'p')}"
                ),
                "case": case,
            }
        )
    return jobs


def _meets_targets(metrics: dict[str, Any], targets: dict[str, float]) -> bool:
    field_metric = "avg_BdotN_over_B"
    field_target = targets["avg_BdotN_over_B"]
    if targets.get("avg_BdotN_over_target_B") is not None:
        field_metric = "avg_BdotN_over_target_B"
        field_target = targets["avg_BdotN_over_target_B"]
    return (
        float(metrics.get(field_metric, 999.0)) <= field_target
        and float(metrics.get("final_min_cc_separation", 0.0))
        >= targets["final_min_cc_separation"]
        and float(metrics.get("final_min_cs_separation", 0.0))
        >= targets["final_min_cs_separation"]
        and float(metrics.get("final_max_curvature", 999.0))
        <= targets["final_max_curvature"]
        and float(metrics.get("final_max_torsion", 999.0))
        <= targets["final_max_torsion"]
    )


def _record_from_metrics(job: dict[str, Any], run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    case = job["case"]
    optimizer = case.get("optimizer_params", {})
    metadata = case.get("experiment_metadata", {})
    lengths = [float(x) for x in metrics.get("final_length_per_coil", []) or []]
    max_length = max(lengths) if lengths else None
    min_length = min(lengths) if lengths else None
    mean_length = sum(lengths) / len(lengths) if lengths else None
    length_variance = (
        sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
        if mean_length is not None
        else None
    )
    length_std = length_variance**0.5 if length_variance is not None else None
    length_cv = length_std / mean_length if mean_length and mean_length > 0 else None
    length_ratio = max_length / min_length if min_length and min_length > 0 else None
    targets = job.get("targets") or TARGETS
    return {
        "run_id": job["run_id"],
        "backend": job["backend"],
        "status": "completed",
        "success": bool(metrics.get("optimization_success", True)),
        "failure_reason": "",
        "wave": metadata.get("wave", ""),
        "family": metadata.get("family", ""),
        "init_family": metadata.get("init_family", ""),
        "policy_family": metadata.get("policy_family", ""),
        "current_family": metadata.get("current_family", ""),
        "geometry_weight_scale": metadata.get("geometry_weight_scale", ""),
        "policy_label": metadata.get("policy_label", ""),
        "random_seed": case.get("random_seed"),
        "order": case.get("coils_params", {}).get("order"),
        "algorithm": optimizer.get("algorithm", ""),
        "max_iterations": optimizer.get("max_iterations", ""),
        "avg_BdotN_over_B": metrics.get("avg_BdotN_over_B"),
        "max_BdotN_over_B": metrics.get("max_BdotN_over_B"),
        "avg_BdotN_over_target_B": metrics.get("avg_BdotN_over_target_B"),
        "max_BdotN_over_target_B": metrics.get("max_BdotN_over_target_B"),
        "initial_R0": metrics.get("initial_R0"),
        "initial_R1": metrics.get("initial_R1"),
        "requested_major_radius_scale": metrics.get("requested_major_radius_scale"),
        "requested_minor_radius_scale": metrics.get("requested_minor_radius_scale"),
        "requested_radial_offset": metrics.get("requested_radial_offset"),
        "requested_current_scale": metrics.get("requested_current_scale"),
        "initial_min_cc_separation": metrics.get("initial_min_cc_separation"),
        "initial_min_cs_separation": metrics.get("initial_min_cs_separation"),
        "initial_total_length": metrics.get("initial_total_length"),
        "initial_length_variance": metrics.get("initial_length_variance"),
        "initial_linking_number": metrics.get("initial_linking_number"),
        "initial_geometry": metrics.get("initial_geometry"),
        "final_min_cc_separation": metrics.get("final_min_cc_separation"),
        "final_min_cs_separation": metrics.get("final_min_cs_separation"),
        "final_total_length": metrics.get("final_total_length"),
        "final_max_length": max_length,
        "final_min_length": min_length,
        "final_mean_coil_length": mean_length,
        "final_length_variance": length_variance,
        "final_length_std": length_std,
        "final_length_cv": length_cv,
        "final_length_ratio": length_ratio,
        "final_max_curvature": metrics.get("final_max_curvature"),
        "final_mean_squared_curvature": metrics.get("final_mean_squared_curvature"),
        "final_arclength_variation": metrics.get("final_arclength_variation"),
        "final_max_torsion": metrics.get("final_max_torsion"),
        "final_linking_number": metrics.get("final_linking_number"),
        "optimization_time": metrics.get("optimization_time"),
        "walltime_sec": metrics.get("walltime_sec"),
        "target_avg_BdotN_over_B": targets["avg_BdotN_over_B"],
        "target_avg_BdotN_over_target_B": targets.get(
            "avg_BdotN_over_target_B", ""
        ),
        "target_final_min_cc_separation": targets["final_min_cc_separation"],
        "target_final_min_cs_separation": targets["final_min_cs_separation"],
        "target_final_max_curvature": targets["final_max_curvature"],
        "target_final_max_torsion": targets["final_max_torsion"],
        "meets_targets": _meets_targets(metrics, targets),
        "run_dir": str(run_dir),
    }


def _failure_record(job: dict[str, Any], run_dir: Path, exc: BaseException) -> dict[str, Any]:
    case = job["case"]
    optimizer = case.get("optimizer_params", {})
    metadata = case.get("experiment_metadata", {})
    targets = job.get("targets") or TARGETS
    return {
        "run_id": job["run_id"],
        "backend": job["backend"],
        "status": "failed",
        "success": False,
        "failure_reason": f"{type(exc).__name__}: {exc}",
        "wave": metadata.get("wave", ""),
        "family": metadata.get("family", ""),
        "init_family": metadata.get("init_family", ""),
        "policy_family": metadata.get("policy_family", ""),
        "current_family": metadata.get("current_family", ""),
        "geometry_weight_scale": metadata.get("geometry_weight_scale", ""),
        "policy_label": metadata.get("policy_label", ""),
        "random_seed": case.get("random_seed"),
        "order": case.get("coils_params", {}).get("order"),
        "algorithm": optimizer.get("algorithm", ""),
        "max_iterations": optimizer.get("max_iterations", ""),
        "target_avg_BdotN_over_B": targets["avg_BdotN_over_B"],
        "target_avg_BdotN_over_target_B": targets.get(
            "avg_BdotN_over_target_B", ""
        ),
        "target_final_min_cc_separation": targets["final_min_cc_separation"],
        "target_final_min_cs_separation": targets["final_min_cs_separation"],
        "target_final_max_curvature": targets["final_max_curvature"],
        "target_final_max_torsion": targets["final_max_torsion"],
        "meets_targets": False,
        "run_dir": str(run_dir),
    }


def _run_one(job: dict[str, Any], results_dir: str, surface_resolution: int, skip_existing: bool) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

    from stellcoilbench.coil_optimization import optimize_coils

    run_dir = Path(results_dir) / "runs" / job["run_id"]
    results_path = run_dir / "results.json"
    record_path = run_dir / "record.json"
    if skip_existing and record_path.exists():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if bool(existing.get("success")) and existing.get("status") == "completed":
            return existing

    if run_dir.exists() and not skip_existing:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    failure_path = run_dir / "failure.txt"
    if failure_path.exists():
        failure_path.unlink()
    case_path = run_dir / "case.yaml"
    _write_yaml_json(job["case"], case_path)

    start = time.perf_counter()
    try:
        metrics = optimize_coils(
            case_path=case_path,
            coils_out_path=run_dir / "coils.json",
            output_dir=run_dir,
            surface_resolution=surface_resolution,
            skip_post_processing=True,
        )
        metrics["walltime_sec"] = metrics.get("walltime_sec", time.perf_counter() - start)
        results_path.write_text(
            json.dumps({"metrics": _json_safe(metrics)}, indent=2),
            encoding="utf-8",
        )
        record = _record_from_metrics(job, run_dir, metrics)
    except Exception as exc:
        record = _failure_record(job, run_dir, exc)
        failure_path.write_text(record["failure_reason"], encoding="utf-8")
    record_path.write_text(json.dumps(_json_safe(record), indent=2), encoding="utf-8")
    return record


def _append_records(path: Path, records: list[dict[str, Any]]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for record in records:
            writer.writerow(record)


def _run_backend_group(
    jobs: list[dict[str, Any]],
    *,
    results_dir: Path,
    surface_resolution: int,
    max_parallel: int,
    skip_existing: bool,
    submit_batch_size: int | None = None,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if skip_existing:
        pending_jobs = []
        for job in jobs:
            record_path = results_dir / "runs" / job["run_id"] / "record.json"
            if not record_path.exists():
                pending_jobs.append(job)
                continue
            try:
                existing = json.loads(record_path.read_text(encoding="utf-8"))
            except Exception:
                pending_jobs.append(job)
                continue
            if not (bool(existing.get("success")) and existing.get("status") == "completed"):
                pending_jobs.append(job)
        jobs = pending_jobs
    if not jobs:
        return []
    if max_parallel <= 1:
        records: list[dict[str, Any]] = []
        for job in jobs:
            record = _run_one(
                job,
                str(results_dir),
                surface_resolution,
                skip_existing,
            )
            records.append(record)
            print(
                f"{record['backend']} {record['run_id']} {record['status']} "
                f"avgBn={record.get('avg_BdotN_over_B')} cc={record.get('final_min_cc_separation')} "
                f"cs={record.get('final_min_cs_separation')} feasible={record.get('meets_targets')}",
                flush=True,
            )
            _append_records(results_dir / "batch_records.csv", [record])
            _append_records(results_dir / "round1_records.csv", [record])
        return records
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=max(1, max_parallel),
        mp_context=mp.get_context("spawn"),
    ) as executor:
        window = max(
            1,
            int(submit_batch_size)
            if submit_batch_size is not None and submit_batch_size > 0
            else max_parallel * 2,
        )
        pending_iter = iter(jobs)
        future_to_job = {}

        def _submit_next() -> bool:
            try:
                job = next(pending_iter)
            except StopIteration:
                return False
            future = executor.submit(
                _run_one,
                job,
                str(results_dir),
                surface_resolution,
                skip_existing,
            )
            future_to_job[future] = job
            return True

        for _ in range(min(window, len(jobs))):
            _submit_next()
        while future_to_job:
            done, _pending = wait(
                future_to_job.keys(),
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                future_to_job.pop(future)
                record = future.result()
                records.append(record)
                print(
                    f"{record['backend']} {record['run_id']} {record['status']} "
                    f"avgBn={record.get('avg_BdotN_over_B')} cc={record.get('final_min_cc_separation')} "
                    f"cs={record.get('final_min_cs_separation')} feasible={record.get('meets_targets')}",
                    flush=True,
                )
                _append_records(results_dir / "batch_records.csv", [record])
                _append_records(results_dir / "round1_records.csv", [record])
                _submit_next()
    return records


def main() -> None:
    args = _parse_args()
    policy = _load_policy(args.policy)
    targets = _policy_targets(policy)
    if policy:
        args.surface = policy.get("surface", args.surface)
        args.focus_executable = policy.get("focus_executable", args.focus_executable)
        if args.results_dir == RESULTS_DIR:
            args.results_dir = Path(policy.get("results_dir", args.results_dir))
    resources = policy.get("resources", {}) if policy else {}
    max_parallel_simsopt = int(
        args.max_parallel_simsopt
        if args.max_parallel_simsopt is not None
        else resources.get("max_parallel_simsopt", 1)
    )
    max_parallel_focus = int(
        args.max_parallel_focus
        if args.max_parallel_focus is not None
        else resources.get("max_parallel_focus", 1)
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    if args.backend in {"both", "simsopt"}:
        jobs.extend(_simsopt_jobs(args.surface, args.jobs_per_backend, args.smoke, policy))
    if args.backend in {"both", "focus"}:
        jobs.extend(
            _focus_jobs(
                args.surface,
                args.focus_executable,
                args.jobs_per_backend,
                args.smoke,
                policy,
            )
        )
    selected_indexes = _parse_job_indexes(args.job_indexes)
    if args.job_index is not None:
        selected_indexes.add(args.job_index)
    if selected_indexes:
        indexed_jobs = list(enumerate(jobs))
        jobs = [job for idx, job in indexed_jobs if idx in selected_indexes]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    for job in jobs:
        job["targets"] = targets

    manifest_jobs = []
    for job in jobs:
        case = job["case"]
        metadata = case.get("experiment_metadata", {})
        manifest_jobs.append(
            {
                "run_id": job["run_id"],
                "backend": job["backend"],
                "surface": case.get("surface_params", {}).get("surface"),
                "ncoils": case.get("coils_params", {}).get("ncoils"),
                "order": case.get("coils_params", {}).get("order"),
                "random_seed": case.get("random_seed"),
                "length_variance_weight": case.get("coil_objective_terms", {}).get(
                    "length_variance_weight"
                ),
                "queue": metadata.get("queue", ""),
                "family": metadata.get("family", ""),
                "init_family": metadata.get("init_family", ""),
                "policy_family": metadata.get("policy_family", ""),
                "current_family": metadata.get("current_family", ""),
                "geometry_weight_scale": metadata.get("geometry_weight_scale", ""),
                "policy_label": metadata.get("policy_label", ""),
            }
        )

    manifest = {
        "surface": args.surface,
        "targets": targets,
        "policy": str(args.policy),
        "max_parallel_simsopt": max_parallel_simsopt,
        "max_parallel_focus": max_parallel_focus,
        "jobs": manifest_jobs,
    }
    manifest_text = json.dumps(manifest, indent=2)
    (args.results_dir / "batch_manifest.json").write_text(
        manifest_text,
        encoding="utf-8",
    )
    (args.results_dir / "round1_manifest.json").write_text(
        manifest_text,
        encoding="utf-8",
    )

    simsopt_jobs = [job for job in jobs if job["backend"] == "simsopt"]
    focus_jobs = [job for job in jobs if job["backend"] == "focus"]
    print(
        f"Prepared {len(simsopt_jobs)} simsopt jobs and {len(focus_jobs)} focus jobs "
        f"in {args.results_dir}",
        flush=True,
    )
    if args.dry_run:
        return
    _enforce_data_twin_launch_gate(
        args.data_twin_campaign,
        allow_unregistered_launch=args.allow_unregistered_launch,
        expected_jobs=len(jobs),
        policy_path=args.policy,
        results_dir=args.results_dir,
    )
    _run_backend_group(
        simsopt_jobs,
        results_dir=args.results_dir,
        surface_resolution=args.surface_resolution,
        max_parallel=max_parallel_simsopt,
        skip_existing=args.skip_existing,
        submit_batch_size=args.submit_batch_size,
    )
    _run_backend_group(
        focus_jobs,
        results_dir=args.results_dir,
        surface_resolution=args.surface_resolution,
        max_parallel=max_parallel_focus,
        skip_existing=args.skip_existing,
        submit_batch_size=args.submit_batch_size,
    )


if __name__ == "__main__":
    main()
