#!/usr/bin/env python3
"""Legacy board adapter for eval000030 optimization experiments.

Use `scripts/optimization_workflow.py --board ...` as the stable repo-level
entry point. This adapter remains here because it owns the eval000030 board
schema, generated policy layout, and screening vocabulary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = REPO / "experiments/wout_squid_eval_000030"
DEFAULT_BOARD = EXPERIMENT_DIR / "board.yaml"
GENERATOR = EXPERIMENT_DIR / "scripts/generate_industrial_round1_stageA_policy.py"
RUNNER = REPO / "scripts/run_simsopt_batch.py"
SCREEN_SCRIPT = EXPERIMENT_DIR / "scripts/screen_industrial_round1_stageA.py"

sys.path.insert(0, str(REPO))

from data_twin.core.hashing import parameter_hash
from data_twin.core.ids import make_id
from data_twin.core.models import (
    ArtifactRecord,
    CaseRecord,
    DecisionRecord,
    EvaluationRecord,
    EventRecord,
    MetricRecord,
    RunRecord,
    now_iso,
)
from data_twin.core.state import init_campaign
from data_twin.storage.jsonl_store import JsonlStore


def _load_families() -> dict[str, dict[str, float]]:
    spec = importlib.util.spec_from_file_location("eval000030_stagea_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load family definitions from {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.FAMILIES)


FAMILIES = _load_families()
GEOM_WEIGHT_KEYS = [
    "cc_weight",
    "cs_weight",
    "curvature_weight",
    "torsion_weight",
    "length_weight",
    "length_variance_weight",
    "msc_weight",
    "arclength_variation_weight",
    "linking_weight",
]
THRESHOLD_KEYS = {
    "length_threshold",
    "cc_threshold",
    "cs_threshold",
    "curvature_threshold",
    "torsion_threshold",
    "msc_threshold",
    "arclength_variation_threshold",
    "length_variance_threshold",
    "length_threshold_device",
    "cc_threshold_device",
    "cs_threshold_device",
    "curvature_threshold_device",
    "torsion_threshold_device",
    "msc_threshold_device",
    "arclength_variation_threshold_device",
    "length_variance_threshold_device",
}
INIT_GEOMETRY_KEYS = [
    "major_radius_scale",
    "minor_radius_scale",
    "radial_offset",
    "normal_offset",
    "initial_cs_gap",
    "coil_spacing_scale",
    "toroidal_phase_offset",
    "coil_phase_jitter",
    "z_offset",
]
CURRENT_INIT_KEYS = [
    "current_scale",
    "current_weights",
]


def _load_board(path: Path) -> dict[str, Any]:
    board = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    board["_board_path"] = path
    return board


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _parse_csv_values(value: str, cast: type = str) -> list[Any]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("CSV override cannot be empty")
    return [cast(item) for item in items]


def _pick_region_value(
    region: dict[str, Any],
    scan: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    if key in region:
        return region[key]
    return scan.get(key, default)


def _merge_named_params(
    region: dict[str, Any],
    scan: dict[str, Any],
    section: str,
    keys: list[str],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    scan_section = scan.get(section, {})
    region_section = region.get(section, {})
    if isinstance(scan_section, dict):
        merged.update(scan_section)
    if isinstance(region_section, dict):
        merged.update(region_section)
    for key in keys:
        if key in scan:
            merged.setdefault(key, scan[key])
        if key in region:
            merged[key] = region[key]
    return {key: value for key, value in merged.items() if value is not None}


def _slug_value(value: Any) -> str:
    return str(value).replace(".", "p").replace("/", "_").replace(" ", "_")


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _board_path(board: dict[str, Any]) -> Path:
    return Path(board.get("_board_path", DEFAULT_BOARD))


def _slug(board: dict[str, Any]) -> str:
    orders = "-".join(str(v) for v in _as_list(board["coil"]["order"]))
    res = int(board["resolution"]["surface_grid"])
    quad = int(board["resolution"]["coil_quadpoints"])
    slug = f"{board['stage']}_{board['wave']}_res{res}_q{quad}_o{orders}"
    suffix = board.get("paths", {}).get("slug_suffix") or board.get("_slug_suffix")
    if suffix:
        slug = f"{slug}_{suffix}"
    return slug


def _generated_paths(board: dict[str, Any]) -> dict[str, Path]:
    generated_dir = REPO / board["paths"]["generated_dir"]
    slug = _slug(board)
    return {
        "generated_dir": generated_dir,
        "policy": generated_dir / f"{slug}_policy.json",
        "manifest": generated_dir / f"{slug}_manifest.csv",
        "campaign_config": generated_dir / f"{slug}_campaign.yaml",
    }


def _results_dir(board: dict[str, Any]) -> str:
    if board.get("paths", {}).get("results_dir"):
        return str(board["paths"]["results_dir"])
    return f"{board['paths']['results_root']}/{_slug(board)}"


def _run_prefix(board: dict[str, Any]) -> str:
    return f"eval000030_{board['stage']}_{board['wave']}"


def _scaled_family(family: str, scale: float) -> dict[str, Any]:
    if family not in FAMILIES:
        raise KeyError(f"Unknown scan family {family!r}")
    data: dict[str, Any] = {
        key: value for key, value in FAMILIES[family].items() if key not in THRESHOLD_KEYS
    }
    for key in GEOM_WEIGHT_KEYS:
        data[key] = round(float(data.get(key, 0.0)) * scale, 10)
    return data


def _build_policy_and_rows(board: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scan = board["scan"]
    optimizer = board["optimizer"]
    thresholds = board["thresholds"]
    guards = board.get("guards", {})
    early_stop = board.get("early_stop")
    queues: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    fixed_common = {
        "wave": board["wave"],
        **thresholds,
        **guards,
    }
    if isinstance(early_stop, dict):
        fixed_common["early_stop"] = early_stop
    if scan.get("regions"):
        region_specs: list[dict[str, Any]] = []
        for region in scan["regions"]:
            flux_weights = region.get("flux_weights", region.get("flux_weight"))
            if flux_weights is None:
                flux_weights = scan["flux_weights"]
            parents = region.get("parents", scan.get("parents", [None]))
            scale = float(
                region.get(
                    "geometry_weight_scale",
                    region.get("geom_scale", scan.get("geometry_weight_scale", 1.0)),
                )
            )
            region_specs.append(
                {
                    "family": region["family"],
                    "scale": scale,
                    "init_family": _pick_region_value(
                        region, scan, "init_family", "legacy_default"
                    ),
                    "policy_family": _pick_region_value(
                        region, scan, "policy_family", region["family"]
                    ),
                    "current_family": _pick_region_value(
                        region, scan, "current_family", "uniform"
                    ),
                    "init_geometry": _merge_named_params(
                        region, scan, "init_geometry", INIT_GEOMETRY_KEYS
                    ),
                    "current_initialization": _merge_named_params(
                        region, scan, "current_initialization", CURRENT_INIT_KEYS
                    ),
                    "flux_weights": _as_list(flux_weights),
                    "label": region.get("label"),
                    "weight_multipliers": region.get("weight_multipliers", {}),
                    "weight_overrides": region.get("weight_overrides", {}),
                    "parents": _as_list(parents),
                }
            )
    else:
        geometry_weight_scales = scan.get("geometry_weight_scales", scan["geom_scales"])
        init_families = _as_list(scan.get("init_families", scan.get("init_family", "legacy_default")))
        current_families = _as_list(scan.get("current_families", scan.get("current_family", "uniform")))
        policy_families = _as_list(scan.get("policy_families", scan.get("policy_family", "")))
        region_specs = [
            {
                "family": family,
                "scale": float(scale),
                "init_family": init_family,
                "policy_family": policy_family or family,
                "current_family": current_family,
                "init_geometry": _merge_named_params({}, scan, "init_geometry", INIT_GEOMETRY_KEYS),
                "current_initialization": _merge_named_params(
                    {}, scan, "current_initialization", CURRENT_INIT_KEYS
                ),
                "flux_weights": scan["flux_weights"],
                "label": None,
                "weight_multipliers": {},
                "weight_overrides": {},
                "parents": _as_list(scan.get("parents", [None])),
            }
            for family in scan["families"]
            for scale in geometry_weight_scales
            for init_family in init_families
            for current_family in current_families
            for policy_family in policy_families
        ]

    for region in region_specs:
        family = str(region["family"])
        scale = float(region["scale"])
        init_family = str(region.get("init_family") or "legacy_default")
        policy_family = str(region.get("policy_family") or family)
        current_family = str(region.get("current_family") or "uniform")
        label = region.get("label")
        if not label:
            label = f"{board['wave']}_{family}_g{_slug_value(scale)}"
            if init_family != "legacy_default":
                label += f"_init-{_slug_value(init_family)}"
            if current_family != "uniform":
                label += f"_cur-{_slug_value(current_family)}"
            if policy_family != family:
                label += f"_pol-{_slug_value(policy_family)}"
        label = str(label).replace(".", "p")
        family_terms = _scaled_family(family, scale)
        for key, multiplier in dict(region.get("weight_multipliers", {})).items():
            if key not in family_terms:
                raise KeyError(f"Unknown weight multiplier key {key!r} in region {label}")
            family_terms[key] = round(float(family_terms[key]) * float(multiplier), 10)
        for key, value in dict(region.get("weight_overrides", {})).items():
            if key not in family_terms:
                raise KeyError(f"Unknown weight override key {key!r} in region {label}")
            family_terms[key] = float(value)
        grid = {
            "algorithm": [optimizer.get("algorithm", "L-BFGS-B")],
            "order": _as_list(board["coil"]["order"]),
            "max_iterations": [int(optimizer["max_iterations"])],
            "flux_weight": region["flux_weights"],
            "random_seed": scan["seeds"],
            "dof_perturbation": [float(optimizer.get("dof_perturbation", 0.0))],
        }
        grid.update({key: [value] for key, value in family_terms.items()})
        for parent_idx, parent in enumerate(region["parents"]):
            if parent is None:
                parent_fixed: dict[str, Any] = {}
                parent_label = label
            else:
                parent_data = dict(parent)
                initial_path = parent_data.get("initial_coils_path") or parent_data.get("coils_json")
                if not initial_path:
                    raise KeyError(f"Warm-start parent in region {label} is missing coils_json")
                initial_path = Path(str(initial_path)).expanduser()
                if not initial_path.is_absolute():
                    initial_path = REPO / initial_path
                parent_fixed = {
                    "initial_coils_path": str(initial_path),
                    "parent_run_id": parent_data.get("run_id", ""),
                    "parent_case_id": parent_data.get("case_id", parent_data.get("run_id", "")),
                }
                parent_label = f"{label}_p{parent_idx:02d}"
            fixed = {
                "family": family,
                "geom_scale": scale,
                "geometry_weight_scale": scale,
                "init_family": init_family,
                "policy_family": policy_family,
                "current_family": current_family,
                "policy_label": parent_label,
                **fixed_common,
                **dict(region.get("init_geometry", {})),
                **dict(region.get("current_initialization", {})),
                **parent_fixed,
            }
            queues[parent_label] = {"grid": grid, "fixed": fixed}

            for order, flux, seed in itertools.product(
                _as_list(board["coil"]["order"]),
                region["flux_weights"],
                scan["seeds"],
            ):
                rows.append(
                    {
                        "wave": board["wave"],
                        "queue": parent_label,
                        "family": family,
                        "geom_scale": scale,
                        "geometry_weight_scale": scale,
                        "init_family": init_family,
                        "policy_family": policy_family,
                        "current_family": current_family,
                        "order": order,
                        "max_iterations": int(optimizer["max_iterations"]),
                        "flux_weight": flux,
                        "random_seed": seed,
                        "dof_perturbation": float(optimizer.get("dof_perturbation", 0.0)),
                        **family_terms,
                        **fixed,
                    }
                )

    common = {
        "description": board.get("description", ""),
        "run_prefix": _run_prefix(board),
        "ncoils": int(board["coil"]["ncoils"]),
        "surface_range": "half period",
        "surface_resolution": int(board["resolution"]["surface_grid"]),
        "coil_quadpoints": int(board["resolution"]["coil_quadpoints"]),
        "plot_upsample_factor": int(board["resolution"].get("plot_upsample_factor", 1)),
        "thresholds": {
            "length_threshold_device": float(thresholds["length_threshold_device"]),
            "cc_threshold_device": float(thresholds["cc_threshold_device"]),
            "cs_threshold_device": float(thresholds["cs_threshold_device"]),
            "curvature_threshold_device": float(thresholds["curvature_threshold_device"]),
            "torsion_threshold_device": float(thresholds["torsion_threshold_device"]),
            "msc_threshold_device": float(thresholds["msc_threshold_device"]),
            "arclength_variation_threshold_device": float(
                thresholds["arclength_variation_threshold_device"]
            ),
            "length_variance_threshold_device": float(
                thresholds.get("length_variance_threshold_device", 0.0)
            ),
        },
    }
    if board.get("target_B") is not None:
        common["target_B"] = float(board["target_B"])
    elif (
        isinstance(board.get("surface_params"), dict)
        and board["surface_params"].get("target_B") is not None
    ):
        common["target_B"] = float(board["surface_params"]["target_B"])
    if board.get("virtual_casing") is not None:
        common["virtual_casing"] = bool(board["virtual_casing"])
    elif (
        isinstance(board.get("surface_params"), dict)
        and board["surface_params"].get("virtual_casing") is not None
    ):
        common["virtual_casing"] = bool(board["surface_params"]["virtual_casing"])

    policy = {
        "surface": board["surface"],
        "results_dir": _results_dir(board),
        "resources": {
            "max_parallel_simsopt": int(optimizer.get("max_parallel_simsopt", 1)),
            "max_parallel_focus": int(optimizer.get("max_parallel_focus", 0)),
        },
        "common": common,
        "simsopt_queues": queues,
    }
    return policy, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No manifest rows generated")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_campaign_config(board: dict[str, Any], path: Path, planned_cases: int) -> None:
    cfg = {
        "campaign_id": board["campaign"],
        "name": board.get("name", board["campaign"]),
        "description": board.get("description", ""),
        "target_type": "stellarator_coil_optimization",
        "target_metadata": {
            "plasma_surface": board["surface"],
            "stage": board["stage"],
            "wave": board["wave"],
            "backend": board["backend"],
            "n_coils": board["coil"]["ncoils"],
            "orders": _as_list(board["coil"]["order"]),
            "surface_resolution": board["resolution"]["surface_grid"],
            "coil_quadpoints": board["resolution"]["coil_quadpoints"],
            "target_B": board.get("target_B")
            or (board.get("surface_params") or {}).get("target_B"),
            "virtual_casing": board.get("virtual_casing")
            if board.get("virtual_casing") is not None
            else (board.get("surface_params") or {}).get("virtual_casing"),
            "planned_cases": planned_cases,
            "planned_parallelism": board["optimizer"].get("max_parallel_simsopt", 1),
            "parameter_vocabulary": {
                "family": "legacy objective-weight family",
                "geom_scale": "legacy geometric objective-weight multiplier",
                "geometry_weight_scale": "geometric objective-weight multiplier",
                "init_family": "fresh-coil initialization geometry family",
                "policy_family": "objective-weight strategy family",
                "current_family": "initial current distribution family",
            },
        },
        "schema_version": "0.1",
        "storage": {
            "root": board["paths"].get("data_twin_root", "experiments/data_twin"),
            "format": "jsonl",
        },
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def generate(board: dict[str, Any]) -> tuple[dict[str, Path], int]:
    paths = _generated_paths(board)
    paths["generated_dir"].mkdir(parents=True, exist_ok=True)
    policy, rows = _build_policy_and_rows(board)
    paths["policy"].write_text(json.dumps(policy, indent=2), encoding="utf-8")
    _write_csv(paths["manifest"], rows)
    _write_campaign_config(board, paths["campaign_config"], len(rows))
    return paths, len(rows)


def _grid_count(board: dict[str, Any]) -> int:
    scan = board["scan"]
    if scan.get("regions"):
        total = 0
        for region in scan["regions"]:
            flux_weights = region.get("flux_weights", region.get("flux_weight"))
            if flux_weights is None:
                flux_weights = scan["flux_weights"]
            parents = _as_list(region.get("parents", scan.get("parents", [None])))
            total += (
                len(_as_list(board["coil"]["order"]))
                * len(_as_list(flux_weights))
                * len(scan["seeds"])
                * len(parents)
            )
        return total
    return (
        len(scan["families"])
        * len(scan.get("geometry_weight_scales", scan["geom_scales"]))
        * len(_as_list(scan.get("init_families", scan.get("init_family", "legacy_default"))))
        * len(_as_list(scan.get("current_families", scan.get("current_family", "uniform"))))
        * len(_as_list(scan.get("policy_families", scan.get("policy_family", ""))))
        * len(_as_list(board["coil"]["order"]))
        * len(scan["flux_weights"])
        * len(scan["seeds"])
    )


def plan(board: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign": board["campaign"],
        "stage": board["stage"],
        "wave": board["wave"],
        "surface": board["surface"],
        "results_dir": _results_dir(board),
        "surface_grid": board["resolution"]["surface_grid"],
        "coil_quadpoints": board["resolution"]["coil_quadpoints"],
        "orders": _as_list(board["coil"]["order"]),
        "families": len(board["scan"]["families"]),
        "regions": len(board["scan"].get("regions", [])),
        "geom_scales": board["scan"].get("geom_scales", []),
        "geometry_weight_scales": board["scan"].get(
            "geometry_weight_scales", board["scan"].get("geom_scales", [])
        ),
        "init_family": board["scan"].get("init_family", ""),
        "policy_family": board["scan"].get("policy_family", ""),
        "current_family": board["scan"].get("current_family", ""),
        "flux_weights": board["scan"]["flux_weights"],
        "seeds": len(board["scan"]["seeds"]),
        "planned_cases": _grid_count(board),
        "max_parallel_simsopt": board["optimizer"].get("max_parallel_simsopt", 1),
        "cs_guard": board.get("guards", {}).get("cs_guard", False),
        "link_guard": board.get("guards", {}).get("link_guard", False),
        "early_stop": board.get("early_stop", {}).get("enabled", False)
        if isinstance(board.get("early_stop"), dict)
        else False,
    }


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_unique(store: JsonlStore, filename: str, key: str, record: dict[str, Any]) -> bool:
    existing = {row.get(key) for row in store.read(filename)}
    if record.get(key) in existing:
        return False
    store.append(filename, record)
    return True


def _upsert_unique(store: JsonlStore, filename: str, key: str, record: dict[str, Any]) -> bool:
    rows = store.read(filename)
    value = record.get(key)
    changed = False
    replaced = False
    new_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get(key) == value:
            replaced = True
            if row != record:
                changed = True
                new_rows.append(record)
            else:
                new_rows.append(row)
        else:
            new_rows.append(row)
    if not replaced:
        new_rows.append(record)
        changed = True
    if changed:
        store.write_all(filename, new_rows)
    return changed


def _upsert_many_unique(
    store: JsonlStore, filename: str, key: str, records: list[dict[str, Any]]
) -> int:
    """Upsert many records with one read and at most one write."""
    if not records:
        return 0
    rows = store.read(filename)
    by_key: dict[Any, dict[str, Any]] = {row.get(key): row for row in rows}
    order = [row.get(key) for row in rows]
    changed = 0
    for record in records:
        value = record.get(key)
        if value not in by_key:
            order.append(value)
            by_key[value] = record
            changed += 1
        elif by_key[value] != record:
            by_key[value] = record
            changed += 1
    if changed:
        store.write_all(filename, [by_key[value] for value in order])
    return changed


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _runner_run_id(board: dict[str, Any], row: dict[str, str], idx: int) -> str:
    algorithm = str(row.get("algorithm") or board["optimizer"].get("algorithm", "L-BFGS-B"))
    algorithm_slug = algorithm.lower().replace("-", "").replace("_", "")
    order = int(float(row["order"]))
    max_iter = int(float(row["max_iterations"]))
    return f"{_run_prefix(board)}_{row['queue']}_simsopt_{idx:04d}_{algorithm_slug}_o{order}_it{max_iter}"


def _command(board: dict[str, Any], policy_path: Path) -> str:
    opt = board["optimizer"]
    cmd = [
        "MPLCONFIGDIR=/tmp/stellcoilbench_mplconfig",
        "conda run -n stellcoilbench_vmec python scripts/run_simsopt_batch.py",
        "--backend simsopt",
        f"--policy {policy_path.relative_to(REPO)}",
        f"--results-dir {_results_dir(board)}",
        f"--data-twin-campaign {board['campaign']}",
        f"--surface {board['surface']}",
        f"--surface-resolution {int(board['resolution']['surface_grid'])}",
        f"--max-parallel-simsopt {int(opt.get('max_parallel_simsopt', 1))}",
        f"--submit-batch-size {int(opt.get('submit_batch_size', 2 * int(opt.get('max_parallel_simsopt', 1))))}",
    ]
    if opt.get("skip_existing") is False or opt.get("no_skip_existing", True):
        cmd.append("--no-skip-existing")
    return " ".join(cmd)


def _load_runner_jobs(policy_path: Path, surface: str) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("round1_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner from {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module._load_policy(policy_path)
    return module._simsopt_jobs(surface, None, False, policy)


def preflight(board: dict[str, Any]) -> dict[str, Any]:
    paths, planned = generate(board)
    rows = _read_manifest(paths["manifest"])
    jobs = _load_runner_jobs(paths["policy"], board["surface"])
    result_root = REPO / _results_dir(board)
    run_dir_count = len(list((result_root / "runs").glob("*"))) if (result_root / "runs").exists() else 0
    record_count = len(list(result_root.glob("runs/*/record.json"))) if result_root.exists() else 0

    errors: list[str] = []
    warnings: list[str] = []
    if planned != _grid_count(board):
        errors.append(f"planned count {planned} != grid count {_grid_count(board)}")
    if len(rows) != planned:
        errors.append(f"manifest rows {len(rows)} != planned count {planned}")
    if len(jobs) != planned:
        errors.append(f"runner jobs {len(jobs)} != planned count {planned}")
    if run_dir_count:
        warnings.append(
            f"results_dir already has {run_dir_count} run directories and {record_count} records"
        )

    expected_orders = {int(v) for v in _as_list(board["coil"]["order"])}
    expected_iter = int(board["optimizer"]["max_iterations"])
    expected_q = int(board["resolution"]["coil_quadpoints"])
    expected_surface = board["surface"]
    probe_indexes = sorted({0, min(1, len(jobs) - 1), max(0, len(jobs) // 2), len(jobs) - 1})
    for idx in probe_indexes:
        if idx < 0 or idx >= len(jobs):
            continue
        case = jobs[idx]["case"]
        if case["surface_params"]["surface"] != expected_surface:
            errors.append(
                f"job {idx} surface {case['surface_params']['surface']} != {expected_surface}"
            )
        if int(case["coils_params"]["order"]) not in expected_orders:
            errors.append(f"job {idx} order mismatch: {case['coils_params']['order']}")
        if int(case["coils_params"].get("numquadpoints", -1)) != expected_q:
            errors.append(
                f"job {idx} numquadpoints mismatch: {case['coils_params'].get('numquadpoints')}"
            )
        if int(case["optimizer_params"]["max_iterations"]) != expected_iter:
            errors.append(
                f"job {idx} max_iterations mismatch: {case['optimizer_params']['max_iterations']}"
            )

    dry_run_cmd = _command(board, paths["policy"]) + " --dry-run"
    dry_run = subprocess.run(dry_run_cmd, cwd=REPO, shell=True, capture_output=True, text=True)
    if dry_run.returncode != 0:
        errors.append(f"runner dry-run failed: {dry_run.stderr.strip() or dry_run.stdout.strip()}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "planned_cases": planned,
        "manifest_rows": len(rows),
        "runner_jobs": len(jobs),
        "results_dir": _results_dir(board),
        "existing_run_dirs": run_dir_count,
        "existing_records": record_count,
        "policy": str(paths["policy"]),
        "manifest": str(paths["manifest"]),
        "command": _command(board, paths["policy"]),
        "dry_run_stdout": dry_run.stdout.strip(),
        "dry_run_stderr": dry_run.stderr.strip(),
    }


def register(board: dict[str, Any]) -> dict[str, Any]:
    paths, planned = generate(board)
    rows = _read_manifest(paths["manifest"])
    root = init_campaign(paths["campaign_config"])
    store = JsonlStore(root)
    command = _command(board, paths["policy"])
    added_cases = 0
    added_runs = 0
    for idx, row in enumerate(rows):
        run_id = _runner_run_id(board, row, idx)
        parameters: dict[str, Any] = {
            **row,
            "surface": board["surface"],
            "backend": board["backend"],
            "policy": str(paths["policy"].relative_to(REPO)),
            "results_dir": _results_dir(board),
            "surface_resolution": board["resolution"]["surface_grid"],
            "coil_quadpoints": board["resolution"]["coil_quadpoints"],
        }
        constraints = {**board.get("thresholds", {}), **board.get("guards", {})}
        case = CaseRecord(
            case_id=run_id,
            campaign_id=board["campaign"],
            generation_index=int(board.get("generation_index", 1)),
            parent_case_ids=[row["parent_case_id"]]
            if row.get("parent_case_id")
            else [],
            proposal_source="board.yaml",
            proposal_reason=f"{board['stage']} {board['wave']} board-driven scan",
            parameter_hash=parameter_hash(parameters, constraints),
            parameters=parameters,
            constraints=constraints,
            input_refs={
                "board": _repo_relative(_board_path(board)),
                "policy": str(paths["policy"].relative_to(REPO)),
                "manifest": str(paths["manifest"].relative_to(REPO)),
            },
            tags=[
                "eval000030",
                board["stage"],
                board["wave"],
                row["family"],
                row.get("init_family", ""),
                row.get("policy_family", ""),
                row.get("current_family", ""),
                row["queue"],
            ],
            status="proposed",
        )
        if _append_unique(store, "cases.jsonl", "case_id", case.to_dict()):
            added_cases += 1
        run = RunRecord(
            run_id=run_id,
            case_id=run_id,
            campaign_id=board["campaign"],
            generation_index=int(board.get("generation_index", 1)),
            backend=str(board["backend"]),
            command=command,
            workdir=str(REPO),
            status="pending",
            config_snapshot_path=str(paths["policy"].relative_to(REPO)),
            environment_snapshot={
                "conda_env": "stellcoilbench_vmec",
                "surface_resolution": board["resolution"]["surface_grid"],
                "coil_quadpoints": board["resolution"]["coil_quadpoints"],
                "max_parallel_simsopt": board["optimizer"].get("max_parallel_simsopt", 1),
            },
            notes="registered_from_board",
        )
        if _append_unique(store, "runs.jsonl", "run_id", run.to_dict()):
            added_runs += 1

    artifacts = [
        ("board", _board_path(board), "Board control file."),
        ("policy", paths["policy"], "Generated runner policy."),
        ("manifest", paths["manifest"], "Generated case manifest."),
        ("campaign_config", paths["campaign_config"], "Generated Data Twin campaign config."),
    ]
    for artifact_type, path, description in artifacts:
        artifact = ArtifactRecord(
            artifact_id=make_id(
                "artifact",
                {"campaign": board["campaign"], "type": artifact_type, "path": str(path)},
            ),
            campaign_id=board["campaign"],
            case_id="campaign",
            run_id="campaign",
            generation_index=int(board.get("generation_index", 1)),
            artifact_type=artifact_type,
            path=str(path),
            relative_path=_repo_relative(path),
            description=description,
            metadata={"planned_cases": planned, "slug": _slug(board)},
        )
        _append_unique(store, "artifacts.jsonl", "artifact_id", artifact.to_dict())

    store.append(
        "events.jsonl",
        EventRecord(
            event_id=make_id(
                "event",
                {
                    "campaign": board["campaign"],
                    "event": "board_registered",
                    "slug": _slug(board),
                    "time": now_iso(),
                },
            ),
            timestamp=now_iso(),
            campaign_id=board["campaign"],
            object_type="campaign",
            object_id=board["campaign"],
            event_type="board_registered",
            message=f"Registered {planned} planned runs from board.",
            metadata={
                "added_cases": added_cases,
                "added_runs": added_runs,
                "policy": str(paths["policy"].relative_to(REPO)),
                "manifest": str(paths["manifest"].relative_to(REPO)),
                "results_dir": _results_dir(board),
            },
        ).to_dict(),
    )
    result = {
        "campaign_root": str(root),
        "planned_cases": planned,
        "added_cases": added_cases,
        "added_runs": added_runs,
        "policy": str(paths["policy"]),
        "manifest": str(paths["manifest"]),
        "results_dir": _results_dir(board),
        "command": command,
    }
    _write_lifecycle(board, "registered", result)
    return result


def _campaign_root(board: dict[str, Any]) -> Path:
    paths = _generated_paths(board)
    if not paths["campaign_config"].exists():
        generate(board)
    cfg = yaml.safe_load(paths["campaign_config"].read_text(encoding="utf-8")) or {}
    return REPO / cfg.get("storage", {}).get("root", "experiments/data_twin") / board["campaign"]


def _store_if_registered(board: dict[str, Any]) -> JsonlStore | None:
    root = _campaign_root(board)
    if not (root / "runs.jsonl").exists():
        return None
    return JsonlStore(root)


def _lifecycle_path(board: dict[str, Any]) -> Path:
    return _campaign_root(board) / "lifecycle.json"


def _read_lifecycle(board: dict[str, Any]) -> dict[str, Any]:
    return _load_json(_lifecycle_path(board))


def _write_lifecycle(board: dict[str, Any], state: str, metadata: dict[str, Any] | None = None) -> None:
    path = _lifecycle_path(board)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **_read_lifecycle(board),
        "campaign": board["campaign"],
        "state": state,
        "updated_at": now_iso(),
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_workflow_event(
    board: dict[str, Any], event_type: str, message: str, metadata: dict[str, Any] | None = None
) -> None:
    store = _store_if_registered(board)
    if store is None:
        return
    store.append(
        "events.jsonl",
        EventRecord(
            event_id=make_id(
                "event",
                {
                    "campaign": board["campaign"],
                    "event": event_type,
                    "time": now_iso(),
                },
            ),
            timestamp=now_iso(),
            campaign_id=board["campaign"],
            object_type="campaign",
            object_id=board["campaign"],
            event_type=event_type,
            message=message,
            metadata=metadata or {},
        ).to_dict(),
    )


def lifecycle_status(board: dict[str, Any]) -> dict[str, Any]:
    planned = _grid_count(board)
    root = _campaign_root(board)
    store = _store_if_registered(board)
    result_root = REPO / _results_dir(board)
    records = sorted(result_root.glob("runs/*/record.json")) if result_root.exists() else []
    lifecycle = _read_lifecycle(board)

    counts = {
        "planned_cases": planned,
        "registered_runs": 0,
        "registered_cases": 0,
        "raw_records": len(records),
        "evaluations": 0,
        "metrics": 0,
        "decisions": 0,
        "artifacts": 0,
        "events": 0,
    }
    if store is not None:
        counts.update(
            {
                "registered_runs": len(store.read("runs.jsonl")),
                "registered_cases": len(store.read("cases.jsonl")),
                "evaluations": len(store.read("evaluations.jsonl")),
                "metrics": len(store.read("metrics.jsonl")),
                "decisions": len(store.read("decisions.jsonl")),
                "artifacts": len(store.read("artifacts.jsonl")),
                "events": len(store.read("events.jsonl")),
            }
        )

    issues: list[str] = []
    if counts["registered_runs"] and counts["registered_runs"] != planned:
        issues.append(f"registered_runs {counts['registered_runs']} != planned_cases {planned}")
    if counts["registered_cases"] and counts["registered_cases"] != planned:
        issues.append(f"registered_cases {counts['registered_cases']} != planned_cases {planned}")
    if counts["evaluations"] and counts["evaluations"] != counts["raw_records"]:
        issues.append(f"evaluations {counts['evaluations']} != raw_records {counts['raw_records']}")
    expected_metrics = counts["evaluations"] * len(METRIC_FIELDS)
    if counts["metrics"] and counts["metrics"] != expected_metrics:
        issues.append(f"metrics {counts['metrics']} != evaluations*{len(METRIC_FIELDS)} {expected_metrics}")

    lifecycle_state = lifecycle.get("state")
    if lifecycle_state == "closed":
        state = "closed"
    elif lifecycle_state == "running" and not counts["raw_records"]:
        state = "running"
    elif counts["decisions"] > 0:
        state = "screened"
    elif counts["raw_records"] and counts["evaluations"] == counts["raw_records"]:
        state = "ingested_complete" if counts["raw_records"] >= planned else "ingested_partial"
    elif counts["raw_records"] >= planned and planned > 0:
        state = "results_complete_uningested"
    elif counts["raw_records"]:
        state = "running"
    elif counts["registered_runs"] >= planned and planned > 0:
        state = "registered"
    elif root.exists():
        state = "planned"
    else:
        state = "draft"

    return {
        "campaign": board["campaign"],
        "state": state,
        "campaign_root": str(root),
        "results_dir": str(result_root),
        "counts": counts,
        "issues": issues,
        "lifecycle": lifecycle,
    }


def _require_state(
    board: dict[str, Any],
    allowed: set[str],
    action: str,
    *,
    allow_closed: bool = False,
) -> dict[str, Any]:
    status = lifecycle_status(board)
    state = status["state"]
    if state == "closed" and not allow_closed:
        raise RuntimeError(f"Cannot {action}: campaign is closed.")
    if state == "closed" and allow_closed:
        return status
    if state not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise RuntimeError(
            f"Cannot {action}: lifecycle state is {state!r}; expected one of {allowed_text}. "
            "Run the previous workflow step first."
        )
    if status["issues"]:
        raise RuntimeError(f"Cannot {action}: lifecycle issues: {'; '.join(status['issues'])}")
    return status


def monitor(board: dict[str, Any]) -> dict[str, Any]:
    root = _campaign_root(board)
    store = JsonlStore(root)
    runs = store.read("runs.jsonl")
    result_root = REPO / _results_dir(board)
    completed_records = list(result_root.glob("runs/*/record.json"))
    record_by_run: dict[str, dict[str, Any]] = {}
    for path in completed_records:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        record_by_run[data.get("run_id", path.parent.name)] = data

    statuses: dict[str, int] = {}
    for run in runs:
        status = "completed" if run["run_id"] in record_by_run else run.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1

    best = sorted(
        record_by_run.values(),
        key=lambda r: (
            r.get("meets_targets") is not True,
            float(r.get("avg_BdotN_over_target_B") or r.get("avg_BdotN_over_B") or 999),
            -float(r.get("final_min_cs_separation") or 0),
        ),
    )[:10]
    return {
        "campaign_root": str(root),
        "results_dir": str(result_root),
        "registered_runs": len(runs),
        "records_found": len(record_by_run),
        "statuses": statuses,
        "best": [
            {
                "run_id": row.get("run_id"),
                "avg_BdotN_over_B": row.get("avg_BdotN_over_B"),
                "avg_BdotN_over_target_B": row.get("avg_BdotN_over_target_B"),
                "cc": row.get("final_min_cc_separation"),
                "cs": row.get("final_min_cs_separation"),
                "max_curvature": row.get("final_max_curvature"),
                "max_torsion": row.get("final_max_torsion"),
                "meets_targets": row.get("meets_targets"),
            }
            for row in best
        ],
    }


METRIC_FIELDS = {
    "avg_BdotN_over_B": ("physics", ""),
    "max_BdotN_over_B": ("physics", ""),
    "avg_BdotN_over_target_B": ("physics", ""),
    "max_BdotN_over_target_B": ("physics", ""),
    "final_min_cc_separation": ("geometry", "m"),
    "final_min_cs_separation": ("geometry", "m"),
    "final_total_length": ("geometry", "m"),
    "final_max_length": ("geometry", "m"),
    "final_min_length": ("geometry", "m"),
    "final_mean_coil_length": ("geometry", "m"),
    "final_length_variance": ("geometry", "m^2"),
    "final_length_std": ("geometry", "m"),
    "final_length_cv": ("geometry", ""),
    "final_length_ratio": ("geometry", ""),
    "final_max_curvature": ("geometry", "1/m"),
    "final_mean_squared_curvature": ("geometry", "1/m^2"),
    "final_arclength_variation": ("geometry", "m^2"),
    "final_max_torsion": ("geometry", "1/m"),
    "final_linking_number": ("topology", ""),
    "optimization_time": ("numerical", "s"),
    "walltime_sec": ("numerical", "s"),
}


def _evaluation_from_record(board: dict[str, Any], record: dict[str, Any]) -> EvaluationRecord:
    failures: list[str] = []
    avg = _float_or_none(record.get("avg_BdotN_over_B"))
    cc = _float_or_none(record.get("final_min_cc_separation"))
    cs = _float_or_none(record.get("final_min_cs_separation"))
    curv = _float_or_none(record.get("final_max_curvature"))
    msc = _float_or_none(record.get("final_mean_squared_curvature"))
    arc = _float_or_none(record.get("final_arclength_variation"))
    torsion = _float_or_none(record.get("final_max_torsion"))
    ratio = _float_or_none(record.get("final_length_ratio"))
    link = _float_or_none(record.get("final_linking_number"))
    checks = [
        ("avg_BdotN_over_B", avg is not None and avg <= 0.08),
        ("cc", cc is not None and cc >= 0.25),
        ("cs", cs is not None and cs >= 0.25),
        ("curvature", curv is not None and curv <= 5.0),
        ("msc", msc is not None and msc <= 5.0),
        ("arclength_variation", arc is not None and arc <= 0.5),
        ("torsion", torsion is not None and torsion <= 7.0),
        ("length_ratio", ratio is not None and ratio <= 1.25),
        ("link", link is not None and abs(link) < 0.5),
    ]
    for name, ok in checks:
        if not ok:
            failures.append(name)
    geometry_score = sum(
        [
            max(0.0, 0.25 - cc) if cc is not None else 1.0,
            max(0.0, 0.25 - cs) if cs is not None else 1.0,
            max(0.0, (curv or 999) - 5.0) * 0.02,
            max(0.0, (torsion or 999) - 7.0) * 0.01,
            max(0.0, (ratio or 999) - 1.25) * 0.03,
        ]
    )
    return EvaluationRecord(
        evaluation_id=make_id("eval", {"run": record["run_id"], "campaign": board["campaign"]}),
        campaign_id=board["campaign"],
        case_id=record["run_id"],
        run_id=record["run_id"],
        generation_index=int(board.get("generation_index", 1)),
        evaluator_name="eval000030.workflow.strict_screen",
        evaluator_version="0.1",
        physics_score=avg,
        geometry_score=geometry_score,
        numerical_score=_float_or_none(record.get("walltime_sec")),
        balanced_score=(avg or 999) + geometry_score,
        constraint_status="pass" if not failures else "fail",
        failure_labels=failures,
        summary="strict pass" if not failures else "violations: " + ",".join(failures),
    )


def ingest(board: dict[str, Any]) -> dict[str, Any]:
    _require_state(
        board,
        {"registered", "running", "results_complete_uningested", "ingested_partial", "ingested_complete"},
        "sync",
    )
    root = _campaign_root(board)
    store = JsonlStore(root)
    result_root = REPO / _results_dir(board)
    records = [_load_json(path) for path in sorted(result_root.glob("runs/*/record.json"))]
    records = [record for record in records if record.get("run_id")]
    existing_runs = {row["run_id"]: row for row in store.read("runs.jsonl")}
    run_updates: list[dict[str, Any]] = []
    metric_updates: list[dict[str, Any]] = []
    evaluation_updates: list[dict[str, Any]] = []
    artifact_updates: list[dict[str, Any]] = []
    completed = 0
    failed = 0
    for record in records:
        run_id = record["run_id"]
        run_dir = Path(record.get("run_dir") or result_root / "runs" / run_id)
        if not run_dir.is_absolute():
            run_dir = REPO / run_dir
        run = dict(existing_runs.get(run_id, {}))
        run.update(
            RunRecord(
                run_id=run_id,
                case_id=run.get("case_id", run_id),
                campaign_id=board["campaign"],
                generation_index=int(board.get("generation_index", 1)),
                backend=str(record.get("backend", board["backend"])),
                command=run.get("command", _command(board, _generated_paths(board)["policy"])),
                workdir=str(REPO),
                status=str(record.get("status", "unknown")),
                failure_reason=str(record.get("failure_reason", "")),
                runtime_seconds=_float_or_none(record.get("walltime_sec")),
                config_snapshot_path=run.get("config_snapshot_path", ""),
                environment_snapshot=run.get("environment_snapshot", {}),
                notes="ingested_from_record_json",
            ).to_dict()
        )
        run_updates.append(run)
        completed += int(record.get("status") == "completed")
        failed += int(record.get("status") == "failed")

        for name, (metric_type, unit) in METRIC_FIELDS.items():
            metric = MetricRecord(
                metric_id=make_id("metric", {"run": run_id, "name": name}),
                campaign_id=board["campaign"],
                case_id=run_id,
                run_id=run_id,
                generation_index=int(board.get("generation_index", 1)),
                metric_name=name,
                metric_value=record.get(name),
                metric_unit=unit,
                metric_type=metric_type,
                extraction_method="record.json",
                available=record.get(name) not in (None, ""),
            )
            metric_updates.append(metric.to_dict())

        evaluation = _evaluation_from_record(board, record)
        evaluation_updates.append(evaluation.to_dict())

        for artifact_type, filename in [
            ("record", "record.json"),
            ("results", "results.json"),
            ("case", "case.yaml"),
            ("coils", "coils.json"),
            ("objective_history", "objective_history.csv"),
            ("constraint_history", "constraint_history.csv"),
        ]:
            path = run_dir / filename
            if not path.exists():
                continue
            artifact = ArtifactRecord(
                artifact_id=make_id(
                    "artifact",
                    {"campaign": board["campaign"], "run": run_id, "path": str(path)},
                ),
                campaign_id=board["campaign"],
                case_id=run_id,
                run_id=run_id,
                generation_index=int(board.get("generation_index", 1)),
                artifact_type=artifact_type,
                path=str(path),
                relative_path=str(path.relative_to(REPO)),
                description=f"{artifact_type} artifact ingested from run directory.",
            )
            artifact_updates.append(artifact.to_dict())

    updated_runs = _upsert_many_unique(store, "runs.jsonl", "run_id", run_updates)
    updated_metrics = _upsert_many_unique(store, "metrics.jsonl", "metric_id", metric_updates)
    updated_evaluations = _upsert_many_unique(
        store, "evaluations.jsonl", "evaluation_id", evaluation_updates
    )
    updated_artifacts = _upsert_many_unique(
        store, "artifacts.jsonl", "artifact_id", artifact_updates
    )

    store.append(
        "events.jsonl",
        EventRecord(
            event_id=make_id(
                "event",
                {"campaign": board["campaign"], "event": "ingest", "time": now_iso()},
            ),
            timestamp=now_iso(),
            campaign_id=board["campaign"],
            object_type="campaign",
            object_id=board["campaign"],
            event_type="results_ingested",
            message=f"Ingested {len(records)} record.json files.",
            metadata={
                "results_dir": _results_dir(board),
                "completed": completed,
                "failed": failed,
                "updated_runs": updated_runs,
                "updated_metrics": updated_metrics,
                "updated_evaluations": updated_evaluations,
                "updated_artifacts": updated_artifacts,
            },
        ).to_dict(),
    )
    state = "ingested_complete" if len(records) >= _grid_count(board) else "ingested_partial"
    result = {
        "campaign_root": str(root),
        "results_dir": str(result_root),
        "records": len(records),
        "completed": completed,
        "failed": failed,
        "updated_runs": updated_runs,
        "updated_metrics": updated_metrics,
        "updated_evaluations": updated_evaluations,
        "updated_artifacts": updated_artifacts,
    }
    _write_lifecycle(board, state, result)
    return result


def _screen_output_dir(board: dict[str, Any]) -> Path:
    return EXPERIMENT_DIR / "reports" / f"{_slug(board)}_screen"


def screen(board: dict[str, Any]) -> dict[str, Any]:
    _require_state(board, {"ingested_partial", "ingested_complete", "screened"}, "screen")
    result_root = REPO / _results_dir(board)
    output_dir = _screen_output_dir(board)
    cmd = [
        sys.executable,
        str(SCREEN_SCRIPT),
        "--results-dir",
        str(result_root),
        "--output-dir",
        str(output_dir),
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "ok": False,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    root = _campaign_root(board)
    store = JsonlStore(root)
    screen_all = output_dir / "screen_all.csv"
    rows = _read_manifest(screen_all) if screen_all.exists() else []
    counts: dict[str, int] = {}
    decision_updates: list[dict[str, Any]] = []
    for row in rows:
        tier = row.get("tier", "unknown")
        counts[tier] = counts.get(tier, 0) + 1
        if tier == "tier0_strict":
            decision, next_action, parent = "accept", "verify_high_resolution", True
        elif tier == "tier1_watch":
            decision, next_action, parent = "watch", "inspect_and_refine", True
        elif tier == "tier2_repair_seed":
            decision, next_action, parent = "repair_seed", "geometry_repair", True
        elif tier == "reject_topology":
            decision, next_action, parent = "reject", "topology_reject", False
        else:
            decision, next_action, parent = "reject", "no_action", False
        record = DecisionRecord(
            decision_id=make_id(
                "decision",
                {"campaign": board["campaign"], "run": row.get("run_id"), "tier": tier},
            ),
            campaign_id=board["campaign"],
            case_id=row.get("run_id", ""),
            run_id=row.get("run_id", ""),
            generation_index=int(board.get("generation_index", 1)),
            decision=decision,
            reason=tier,
            next_action=next_action,
            parent_for_future_cases=parent,
            decided_by="eval000030.workflow.screen",
            notes=f"rank_score={row.get('rank_score', '')}",
        )
        decision_updates.append(record.to_dict())

    artifact_updates: list[dict[str, Any]] = []
    for artifact_type, path in [
        ("screen_report", output_dir / "screen_report.md"),
        ("screen_all", output_dir / "screen_all.csv"),
        ("screen_tier0", output_dir / "tier0_strict.csv"),
        ("screen_tier1", output_dir / "tier1_watch.csv"),
        ("screen_tier2", output_dir / "tier2_repair_seed.csv"),
    ]:
        if not path.exists():
            continue
        artifact = ArtifactRecord(
            artifact_id=make_id(
                "artifact",
                {"campaign": board["campaign"], "screen": artifact_type, "path": str(path)},
            ),
            campaign_id=board["campaign"],
            case_id="campaign",
            run_id="campaign",
            generation_index=int(board.get("generation_index", 1)),
            artifact_type=artifact_type,
            path=str(path),
            relative_path=str(path.relative_to(REPO)),
            description=f"{artifact_type} generated by workflow screen.",
            metadata={"results_dir": _results_dir(board)},
        )
        artifact_updates.append(artifact.to_dict())

    updated_decisions = _upsert_many_unique(
        store, "decisions.jsonl", "decision_id", decision_updates
    )
    _upsert_many_unique(store, "artifacts.jsonl", "artifact_id", artifact_updates)

    store.append(
        "events.jsonl",
        EventRecord(
            event_id=make_id(
                "event",
                {"campaign": board["campaign"], "event": "screen", "time": now_iso()},
            ),
            timestamp=now_iso(),
            campaign_id=board["campaign"],
            object_type="campaign",
            object_id=board["campaign"],
            event_type="results_screened",
            message=f"Screened {len(rows)} runs.",
            metadata={"output_dir": str(output_dir.relative_to(REPO)), "tier_counts": counts},
        ).to_dict(),
    )
    result = {
        "ok": True,
        "results_dir": str(result_root),
        "output_dir": str(output_dir),
        "screened": len(rows),
        "tier_counts": counts,
        "updated_decisions": updated_decisions,
        "stdout": proc.stdout.strip(),
    }
    _write_lifecycle(board, "screened", result)
    return result


def launch(board: dict[str, Any], yes: bool) -> dict[str, Any]:
    _require_state(board, {"registered", "running", "ingested_partial"}, "launch")
    paths, _planned = generate(board)
    command = _command(board, paths["policy"])
    if not yes:
        return {"would_run": command, "note": "pass --yes to launch"}
    _write_lifecycle(board, "running", {"command": command})
    _append_workflow_event(board, "launch_started", "Workflow launch started.", {"command": command})
    result = subprocess.run(command, cwd=REPO, shell=True)
    payload = {"command": command, "returncode": result.returncode}
    _append_workflow_event(board, "launch_finished", "Workflow launch finished.", payload)
    return payload


def prepare(board: dict[str, Any]) -> dict[str, Any]:
    preflight_result = preflight(board)
    if not preflight_result["ok"]:
        _write_lifecycle(board, "planned", preflight_result)
        return {
            "ok": False,
            "state": "planned",
            "note": "preflight failed; campaign was not registered",
            "preflight": preflight_result,
        }
    register_result = register(board)
    return {
        "ok": True,
        "state": "registered",
        "preflight": preflight_result,
        "register": register_result,
    }


def close(board: dict[str, Any]) -> dict[str, Any]:
    status = _require_state(board, {"screened"}, "close", allow_closed=True)
    if status["state"] == "closed":
        return {"ok": True, "state": "closed", "note": "campaign was already closed", **status}
    _write_lifecycle(board, "closed", status)
    _append_workflow_event(board, "campaign_closed", "Workflow campaign closed.", status["counts"])
    return {"ok": True, "state": "closed", **lifecycle_status(board)}


def apply_overrides(board: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Apply CLI overrides to the in-memory board only."""
    applied: list[str] = []

    def set_path(path: tuple[str, ...], value: Any, label: str) -> None:
        cursor = board
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = value
        applied.append(f"{label}={value}")

    if args.results_dir:
        set_path(("paths", "results_dir"), args.results_dir, "results_dir")
    if args.campaign:
        board["campaign"] = args.campaign
        applied.append(f"campaign={args.campaign}")
    if args.stage:
        board["stage"] = args.stage
        applied.append(f"stage={args.stage}")
    if args.wave:
        board["wave"] = args.wave
        applied.append(f"wave={args.wave}")
    if args.name:
        board["name"] = args.name
        applied.append(f"name={args.name}")
    if args.description:
        board["description"] = args.description
        applied.append(f"description={args.description}")
    if args.target_B is not None:
        board["target_B"] = float(args.target_B)
        applied.append(f"target_B={args.target_B}")
    if args.generation_index is not None:
        board["generation_index"] = int(args.generation_index)
        applied.append(f"generation_index={args.generation_index}")
    if args.surface_grid is not None:
        set_path(("resolution", "surface_grid"), int(args.surface_grid), "surface_grid")
    if args.coil_quadpoints is not None:
        set_path(("resolution", "coil_quadpoints"), int(args.coil_quadpoints), "coil_quadpoints")
    if args.plot_upsample_factor is not None:
        set_path(
            ("resolution", "plot_upsample_factor"),
            int(args.plot_upsample_factor),
            "plot_upsample_factor",
        )
    if args.order:
        orders = _parse_csv_values(args.order, int)
        set_path(("coil", "order"), orders, "order")
    if args.max_iterations is not None:
        set_path(("optimizer", "max_iterations"), int(args.max_iterations), "max_iterations")
    if args.max_parallel_simsopt is not None:
        set_path(
            ("optimizer", "max_parallel_simsopt"),
            int(args.max_parallel_simsopt),
            "max_parallel_simsopt",
        )
    if args.submit_batch_size is not None:
        set_path(
            ("optimizer", "submit_batch_size"),
            int(args.submit_batch_size),
            "submit_batch_size",
        )
    if args.dof_perturbation is not None:
        set_path(
            ("optimizer", "dof_perturbation"),
            float(args.dof_perturbation),
            "dof_perturbation",
        )
    if args.families:
        set_path(("scan", "families"), _parse_csv_values(args.families), "families")
    if args.seeds:
        set_path(("scan", "seeds"), _parse_csv_values(args.seeds, int), "seeds")
    if args.flux_weights:
        set_path(("scan", "flux_weights"), _parse_csv_values(args.flux_weights, float), "flux_weights")
    if args.geom_scales:
        set_path(("scan", "geom_scales"), _parse_csv_values(args.geom_scales, float), "geom_scales")
    if args.region_label_prefix:
        regions = board.get("scan", {}).get("regions") or []
        for region in regions:
            old_label = str(region.get("label", ""))
            suffix = old_label.split("_", 1)[1] if "_" in old_label else old_label
            region["label"] = f"{args.region_label_prefix}_{suffix}" if suffix else args.region_label_prefix
        applied.append(f"region_label_prefix={args.region_label_prefix}")
    for key in [
        "length_threshold_device",
        "cc_threshold_device",
        "cs_threshold_device",
        "curvature_threshold_device",
        "torsion_threshold_device",
        "msc_threshold_device",
        "arclength_variation_threshold_device",
        "length_variance_threshold_device",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            set_path(("thresholds", key), float(value), key)
    if args.cs_guard is not None:
        set_path(("guards", "cs_guard"), bool(args.cs_guard), "cs_guard")
    if args.link_guard is not None:
        set_path(("guards", "link_guard"), bool(args.link_guard), "link_guard")
    if args.early_stop is not None:
        set_path(("early_stop", "enabled"), bool(args.early_stop), "early_stop")
    if args.no_skip_existing:
        set_path(("optimizer", "no_skip_existing"), True, "no_skip_existing")
    if args.skip_existing:
        set_path(("optimizer", "no_skip_existing"), False, "skip_existing")
        set_path(("optimizer", "skip_existing"), True, "skip_existing")
    if args.slug_suffix:
        set_path(("paths", "slug_suffix"), args.slug_suffix, "slug_suffix")
    elif applied:
        digest = hashlib.sha1("|".join(applied).encode("utf-8")).hexdigest()[:8]
        board["_slug_suffix"] = f"cli_{digest}"
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "plan",
            "generate",
            "prepare",
            "register",
            "preflight",
            "status",
            "monitor",
            "sync",
            "ingest",
            "screen",
            "report",
            "close",
            "launch",
        ],
    )
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--wave", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--target-B", type=float, default=None)
    parser.add_argument("--generation-index", type=int, default=None)
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override board-derived results directory for this workflow invocation.",
    )
    parser.add_argument("--surface-grid", type=int, default=None)
    parser.add_argument("--coil-quadpoints", type=int, default=None)
    parser.add_argument("--plot-upsample-factor", type=int, default=None)
    parser.add_argument("--order", default=None, help="Comma-separated coil orders, e.g. 6 or 6,8.")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-parallel-simsopt", type=int, default=None)
    parser.add_argument("--submit-batch-size", type=int, default=None)
    parser.add_argument("--dof-perturbation", type=float, default=None)
    parser.add_argument("--families", default=None, help="Comma-separated scan families.")
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds.")
    parser.add_argument("--flux-weights", default=None, help="Comma-separated flux weights.")
    parser.add_argument("--geom-scales", default=None, help="Comma-separated geometry scales.")
    parser.add_argument(
        "--region-label-prefix",
        default=None,
        help="Replace the prefix before the first underscore in each scan region label.",
    )
    parser.add_argument("--length-threshold-device", type=float, default=None)
    parser.add_argument("--cc-threshold-device", type=float, default=None)
    parser.add_argument("--cs-threshold-device", type=float, default=None)
    parser.add_argument("--curvature-threshold-device", type=float, default=None)
    parser.add_argument("--torsion-threshold-device", type=float, default=None)
    parser.add_argument("--msc-threshold-device", type=float, default=None)
    parser.add_argument("--arclength-variation-threshold-device", type=float, default=None)
    parser.add_argument("--length-variance-threshold-device", type=float, default=None)
    parser.add_argument(
        "--slug-suffix",
        default=None,
        help="Suffix for generated policy/manifest names. Defaults to a hash for CLI overrides.",
    )
    parser.add_argument("--cs-guard", dest="cs_guard", action="store_true", default=None)
    parser.add_argument("--no-cs-guard", dest="cs_guard", action="store_false")
    parser.add_argument("--link-guard", dest="link_guard", action="store_true", default=None)
    parser.add_argument("--no-link-guard", dest="link_guard", action="store_false")
    parser.add_argument("--early-stop", dest="early_stop", action="store_true", default=None)
    parser.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--allow-legacy-direct",
        action="store_true",
        help="Bypass the repo-level workflow entrypoint guard for emergency recovery.",
    )
    parser.add_argument("--yes", action="store_true", help="Required for launch.")
    args = parser.parse_args()

    if (
        not args.allow_legacy_direct
        and os.environ.get("STELLCOILBENCH_WORKFLOW_ENTRYPOINT")
        != "scripts/optimization_workflow.py"
    ):
        raise SystemExit(
            "Direct use of experiments/wout_squid_eval_000030/workflow/experiment.py "
            "is disabled for agents. Use `conda run -n stellcoilbench_vmec python "
            "scripts/optimization_workflow.py <action> --board <board.yaml>`."
        )

    board = _load_board(args.board)
    overrides = apply_overrides(board, args)
    try:
        if args.action == "plan":
            result = plan(board)
        elif args.action == "generate":
            paths, planned = generate(board)
            result = {"planned_cases": planned, **{key: str(value) for key, value in paths.items()}}
        elif args.action == "register":
            result = register(board)
        elif args.action == "prepare":
            result = prepare(board)
        elif args.action == "preflight":
            result = preflight(board)
        elif args.action == "status":
            result = lifecycle_status(board)
        elif args.action == "monitor":
            result = monitor(board)
        elif args.action in {"sync", "ingest"}:
            result = ingest(board)
        elif args.action in {"screen", "report"}:
            result = screen(board)
        elif args.action == "close":
            result = close(board)
        elif args.action == "launch":
            result = launch(board, args.yes)
        else:
            raise ValueError(args.action)
    except RuntimeError as exc:
        print(yaml.safe_dump({"ok": False, "error": str(exc)}, sort_keys=False, allow_unicode=True))
        raise SystemExit(2) from exc
    if overrides:
        result = {"overrides": overrides, **result}
    print(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
