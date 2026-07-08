#!/usr/bin/env python3
"""Generate industrial Stage-A Round1 policy and job manifest for eval000030."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    ROOT
    / "experiments/wout_squid_eval_000030/policies/"
    / "squid_eval000030_industrial_round1_stageA_policy.yaml"
)
DEFAULT_MANIFEST = (
    ROOT
    / "experiments/wout_squid_eval_000030/policies/"
    / "squid_eval000030_industrial_round1_stageA_manifest.csv"
)


FAMILIES: dict[str, dict[str, float]] = {
    "seed_bn_recover": {
        "cc_weight": 3000.0,
        "cs_weight": 3000.0,
        "curvature_weight": 30.0,
        "torsion_weight": 15.0,
        "length_weight": 20.0,
        "msc_weight": 0.0,
        "arclength_variation_weight": 0.0,
        "linking_weight": 0.0,
    },
    "seed_balanced": {
        "cc_weight": 6000.0,
        "cs_weight": 6000.0,
        "curvature_weight": 60.0,
        "torsion_weight": 30.0,
        "length_weight": 20.0,
        "msc_weight": 30.0,
        "arclength_variation_weight": 30.0,
        "linking_weight": 0.0,
    },
    "seed_clearance_repair": {
        "cc_weight": 12000.0,
        "cs_weight": 12000.0,
        "curvature_weight": 60.0,
        "torsion_weight": 30.0,
        "length_weight": 20.0,
        "msc_weight": 30.0,
        "arclength_variation_weight": 30.0,
        "linking_weight": 0.0,
    },
    "seed_smooth_repair": {
        "cc_weight": 6000.0,
        "cs_weight": 6000.0,
        "curvature_weight": 120.0,
        "torsion_weight": 60.0,
        "length_weight": 60.0,
        "msc_weight": 100.0,
        "arclength_variation_weight": 100.0,
        "linking_weight": 0.0,
    },
    "clearance_anchor": {
        "cc_weight": 4500.0,
        "cs_weight": 4500.0,
        "curvature_weight": 280.0,
        "torsion_weight": 180.0,
        "length_weight": 60.0,
        "msc_weight": 120.0,
        "arclength_variation_weight": 70.0,
        "linking_weight": 250.0,
        "cc_threshold": 0.30,
        "cs_threshold": 0.30,
        "curvature_threshold": 4.5,
        "torsion_threshold": 6.5,
        "msc_threshold": 4.5,
        "arclength_variation_threshold": 0.35,
        "length_threshold": 11.5,
    },
    "balanced_geometry": {
        "cc_weight": 3500.0,
        "cs_weight": 3500.0,
        "curvature_weight": 220.0,
        "torsion_weight": 160.0,
        "length_weight": 70.0,
        "msc_weight": 100.0,
        "arclength_variation_weight": 80.0,
        "linking_weight": 220.0,
        "cc_threshold": 0.28,
        "cs_threshold": 0.28,
        "curvature_threshold": 4.8,
        "torsion_threshold": 7.0,
        "msc_threshold": 5.0,
        "arclength_variation_threshold": 0.40,
        "length_threshold": 12.0,
    },
    "length_uniform": {
        "cc_weight": 3000.0,
        "cs_weight": 3200.0,
        "curvature_weight": 180.0,
        "torsion_weight": 140.0,
        "length_weight": 120.0,
        "msc_weight": 90.0,
        "arclength_variation_weight": 160.0,
        "linking_weight": 220.0,
        "cc_threshold": 0.27,
        "cs_threshold": 0.27,
        "curvature_threshold": 5.0,
        "torsion_threshold": 7.0,
        "msc_threshold": 5.0,
        "arclength_variation_threshold": 0.30,
        "length_threshold": 11.0,
    },
    "smooth_torsion": {
        "cc_weight": 3000.0,
        "cs_weight": 3000.0,
        "curvature_weight": 260.0,
        "torsion_weight": 260.0,
        "length_weight": 70.0,
        "msc_weight": 160.0,
        "arclength_variation_weight": 90.0,
        "linking_weight": 240.0,
        "cc_threshold": 0.27,
        "cs_threshold": 0.27,
        "curvature_threshold": 4.5,
        "torsion_threshold": 6.0,
        "msc_threshold": 4.5,
        "arclength_variation_threshold": 0.35,
        "length_threshold": 12.0,
    },
    "bn_balanced": {
        "cc_weight": 2600.0,
        "cs_weight": 2800.0,
        "curvature_weight": 170.0,
        "torsion_weight": 130.0,
        "length_weight": 55.0,
        "msc_weight": 80.0,
        "arclength_variation_weight": 60.0,
        "linking_weight": 200.0,
        "cc_threshold": 0.26,
        "cs_threshold": 0.26,
        "curvature_threshold": 5.0,
        "torsion_threshold": 7.0,
        "msc_threshold": 5.5,
        "arclength_variation_threshold": 0.45,
        "length_threshold": 12.5,
    },
    "high_clearance_soft_bn": {
        "cc_weight": 5200.0,
        "cs_weight": 5200.0,
        "curvature_weight": 200.0,
        "torsion_weight": 140.0,
        "length_weight": 60.0,
        "msc_weight": 90.0,
        "arclength_variation_weight": 70.0,
        "linking_weight": 300.0,
        "cc_threshold": 0.32,
        "cs_threshold": 0.32,
        "curvature_threshold": 5.0,
        "torsion_threshold": 7.0,
        "msc_threshold": 5.5,
        "arclength_variation_threshold": 0.40,
        "length_threshold": 12.5,
    },
    "msc_first": {
        "cc_weight": 3000.0,
        "cs_weight": 3000.0,
        "curvature_weight": 220.0,
        "torsion_weight": 160.0,
        "length_weight": 70.0,
        "msc_weight": 220.0,
        "arclength_variation_weight": 90.0,
        "linking_weight": 240.0,
        "cc_threshold": 0.27,
        "cs_threshold": 0.27,
        "curvature_threshold": 4.8,
        "torsion_threshold": 7.0,
        "msc_threshold": 4.0,
        "arclength_variation_threshold": 0.35,
        "length_threshold": 12.0,
    },
    "strict_no_knot": {
        "cc_weight": 4200.0,
        "cs_weight": 4200.0,
        "curvature_weight": 320.0,
        "torsion_weight": 320.0,
        "length_weight": 120.0,
        "msc_weight": 240.0,
        "arclength_variation_weight": 180.0,
        "linking_weight": 350.0,
        "cc_threshold": 0.30,
        "cs_threshold": 0.30,
        "curvature_threshold": 4.3,
        "torsion_threshold": 5.8,
        "msc_threshold": 4.0,
        "arclength_variation_threshold": 0.25,
        "length_threshold": 11.0,
    },
    "late_bn_push": {
        "cc_weight": 2500.0,
        "cs_weight": 2500.0,
        "curvature_weight": 140.0,
        "torsion_weight": 120.0,
        "length_weight": 50.0,
        "msc_weight": 70.0,
        "arclength_variation_weight": 50.0,
        "linking_weight": 220.0,
        "cc_threshold": 0.26,
        "cs_threshold": 0.26,
        "curvature_threshold": 5.2,
        "torsion_threshold": 7.5,
        "msc_threshold": 5.5,
        "arclength_variation_threshold": 0.45,
        "length_threshold": 13.0,
    },
    "strict_clearance": {
        "cc_weight": 6500.0,
        "cs_weight": 6500.0,
        "curvature_weight": 240.0,
        "torsion_weight": 180.0,
        "length_weight": 80.0,
        "msc_weight": 120.0,
        "arclength_variation_weight": 90.0,
        "linking_weight": 420.0,
        "cc_threshold": 0.35,
        "cs_threshold": 0.35,
        "curvature_threshold": 5.0,
        "torsion_threshold": 7.0,
        "msc_threshold": 5.0,
        "arclength_variation_threshold": 0.35,
        "length_threshold": 12.5,
    },
    "a1r_cs_wall": {
        "cc_weight": 70000.0,
        "cs_weight": 160000.0,
        "curvature_weight": 4500.0,
        "torsion_weight": 7000.0,
        "length_weight": 900.0,
        "msc_weight": 5000.0,
        "arclength_variation_weight": 2500.0,
        "linking_weight": 1800.0,
        "cc_threshold": 0.34,
        "cs_threshold": 0.38,
        "curvature_threshold": 4.2,
        "torsion_threshold": 5.5,
        "msc_threshold": 4.0,
        "arclength_variation_threshold": 0.22,
        "length_threshold": 10.5,
    },
    "a1r_cs_curvature": {
        "cc_weight": 55000.0,
        "cs_weight": 120000.0,
        "curvature_weight": 9000.0,
        "torsion_weight": 5500.0,
        "length_weight": 800.0,
        "msc_weight": 9000.0,
        "arclength_variation_weight": 2200.0,
        "linking_weight": 1600.0,
        "cc_threshold": 0.32,
        "cs_threshold": 0.36,
        "curvature_threshold": 4.0,
        "torsion_threshold": 6.0,
        "msc_threshold": 3.8,
        "arclength_variation_threshold": 0.25,
        "length_threshold": 10.8,
    },
    "a1r_torsion_wall": {
        "cc_weight": 50000.0,
        "cs_weight": 100000.0,
        "curvature_weight": 6000.0,
        "torsion_weight": 12000.0,
        "length_weight": 1000.0,
        "msc_weight": 7000.0,
        "arclength_variation_weight": 3000.0,
        "linking_weight": 1800.0,
        "cc_threshold": 0.32,
        "cs_threshold": 0.35,
        "curvature_threshold": 4.2,
        "torsion_threshold": 4.8,
        "msc_threshold": 4.0,
        "arclength_variation_threshold": 0.20,
        "length_threshold": 10.5,
    },
    "a1r_clearance_balanced": {
        "cc_weight": 90000.0,
        "cs_weight": 90000.0,
        "curvature_weight": 5000.0,
        "torsion_weight": 6000.0,
        "length_weight": 800.0,
        "msc_weight": 5500.0,
        "arclength_variation_weight": 2200.0,
        "linking_weight": 2200.0,
        "cc_threshold": 0.38,
        "cs_threshold": 0.34,
        "curvature_threshold": 4.3,
        "torsion_threshold": 5.5,
        "msc_threshold": 4.2,
        "arclength_variation_threshold": 0.25,
        "length_threshold": 10.8,
    },
    "a1r_length_smooth": {
        "cc_weight": 50000.0,
        "cs_weight": 90000.0,
        "curvature_weight": 6500.0,
        "torsion_weight": 6500.0,
        "length_weight": 2500.0,
        "msc_weight": 6500.0,
        "arclength_variation_weight": 5000.0,
        "linking_weight": 1800.0,
        "cc_threshold": 0.30,
        "cs_threshold": 0.34,
        "curvature_threshold": 4.2,
        "torsion_threshold": 5.5,
        "msc_threshold": 4.0,
        "arclength_variation_threshold": 0.15,
        "length_threshold": 10.0,
    },
    "a1r_low_flux_probe": {
        "cc_weight": 45000.0,
        "cs_weight": 80000.0,
        "curvature_weight": 4000.0,
        "torsion_weight": 4500.0,
        "length_weight": 700.0,
        "msc_weight": 4500.0,
        "arclength_variation_weight": 1800.0,
        "linking_weight": 1500.0,
        "cc_threshold": 0.30,
        "cs_threshold": 0.32,
        "curvature_threshold": 4.5,
        "torsion_threshold": 6.0,
        "msc_threshold": 4.5,
        "arclength_variation_threshold": 0.28,
        "length_threshold": 11.0,
    },
}

WAVES = {
    "smoke": {
        "families": [
            "clearance_anchor",
            "balanced_geometry",
            "smooth_torsion",
            "strict_no_knot",
        ],
        "orders": [6],
        "seeds": [101, 102],
        "flux_weights": [60.0],
        "geom_scales": [1.0],
        "max_iterations": 80,
        "dof_perturbation": 0.006,
    },
    "a0": {
        "families": [
            "clearance_anchor",
            "balanced_geometry",
            "length_uniform",
            "smooth_torsion",
            "bn_balanced",
            "high_clearance_soft_bn",
            "msc_first",
            "strict_no_knot",
        ],
        "orders": [6],
        "seeds": [1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108],
        "flux_weights": [45.0, 90.0],
        "geom_scales": [0.85, 1.15],
        "max_iterations": 650,
        "dof_perturbation": 0.008,
    },
    "a1": {
        "families": list(FAMILIES.keys()),
        "orders": [6],
        "seeds": [2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112],
        "flux_weights": [35.0, 70.0, 120.0],
        "geom_scales": [0.75, 1.25],
        "max_iterations": 900,
        "dof_perturbation": 0.010,
    },
    "a1_revised": {
        "families": [
            "a1r_cs_wall",
            "a1r_cs_curvature",
            "a1r_torsion_wall",
            "a1r_clearance_balanced",
            "a1r_length_smooth",
            "a1r_low_flux_probe",
        ],
        "orders": [6],
        "seeds": [3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108],
        "flux_weights": [8.0, 18.0, 35.0],
        "geom_scales": [1.0, 2.5],
        "max_iterations": 900,
        "dof_perturbation": 0.006,
    },
}

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
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", choices=sorted(WAVES), default="a0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-parallel", type=int, default=32)
    parser.add_argument("--surface-resolution", type=int, default=128)
    parser.add_argument("--coil-quadpoints", type=int, default=256)
    return parser.parse_args()


def _scaled_family(family: str, scale: float) -> dict[str, Any]:
    data: dict[str, Any] = {
        key: value for key, value in FAMILIES[family].items() if key not in THRESHOLD_KEYS
    }
    for key in GEOM_WEIGHT_KEYS:
        if key in data:
            data[key] = round(float(data[key]) * scale, 10)
    return data


def _policy(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    wave = WAVES[args.wave]
    queues: dict[str, Any] = {}
    manifest_rows: list[dict[str, Any]] = []
    for family in wave["families"]:
        for scale in wave["geom_scales"]:
            label = f"{args.wave}_{family}_g{str(scale).replace('.', 'p')}"
            family_terms = _scaled_family(family, float(scale))
            grid = {
                "algorithm": ["L-BFGS-B"],
                "order": wave["orders"],
                "max_iterations": [wave["max_iterations"]],
                "flux_weight": wave["flux_weights"],
                "random_seed": wave["seeds"],
                "dof_perturbation": [wave["dof_perturbation"]],
            }
            grid.update({key: [value] for key, value in family_terms.items()})
            fixed = {
                "family": family,
                "policy_label": label,
                "wave": args.wave,
                "length_threshold_device": 24.0,
                "cc_threshold_device": 0.25,
                "cs_threshold_device": 0.25,
                "curvature_threshold_device": 5.0,
                "torsion_threshold_device": 7.0,
                "msc_threshold_device": 5.0,
                "arclength_variation_threshold_device": 0.5,
                "cs_guard": False,
                "cs_guard_interval": 5,
                "cs_guard_hard_min": 0.25,
                "cs_guard_soft_min": 0.32,
                "cs_guard_penalty": 1.0e8,
                "cs_guard_rollback": True,
                "link_guard": False,
                "link_guard_interval": 10,
                "link_guard_penalty": 1.0e7,
                "link_guard_tolerance": 0.2,
                "link_guard_rollback": True,
                "link_guard_sample_stride": 2,
                "link_guard_record_interval": 50,
                "early_stop": {
                    "enabled": False,
                    "min_eval": 150,
                    "check_interval": 25,
                    "hard_min_cc": 0.15,
                    "hard_min_cs": 0.15,
                    "sustained_bad_checks": 3,
                    "max_curvature_abort": 20.0,
                    "max_torsion_abort": 40.0,
                    "max_msc_abort": 40.0,
                    "max_link_guard_violations": 1,
                    "objective_stall_window": 100,
                    "objective_min_relative_improvement": 0.03,
                },
            }
            queues[label] = {"grid": grid, "fixed": fixed}
            for order, flux, seed in itertools.product(
                wave["orders"], wave["flux_weights"], wave["seeds"]
            ):
                manifest_rows.append(
                    {
                        "wave": args.wave,
                        "queue": label,
                        "family": family,
                        "geom_scale": scale,
                        "order": order,
                        "max_iterations": wave["max_iterations"],
                        "flux_weight": flux,
                        "random_seed": seed,
                        "dof_perturbation": wave["dof_perturbation"],
                        **family_terms,
                        **fixed,
                    }
                )

    policy = {
        "surface": "plasma_surfaces/wout_squid_eval_000030.nc",
        "results_dir": (
            "experiments/wout_squid_eval_000030/raw/results/"
            f"industrial_round1_stageA_{args.wave}_res{args.surface_resolution}_q{args.coil_quadpoints}"
        ),
        "resources": {
            "max_parallel_simsopt": args.max_parallel,
            "max_parallel_focus": 0,
        },
        "common": {
            "description": (
                "Round1 Stage-A cold scan for wout_squid_eval_000030 "
                "with physically fixed thresholds"
            ),
            "run_prefix": f"eval000030_industrial_stageA_{args.wave}",
            "ncoils": 4,
            "surface_range": "half period",
            "surface_resolution": args.surface_resolution,
            "coil_quadpoints": args.coil_quadpoints,
            "plot_upsample_factor": 1,
            "thresholds": {
                "length_threshold": 24.0,
                "cc_threshold": 0.25,
                "cs_threshold": 0.25,
                "curvature_threshold": 5.0,
                "torsion_threshold": 7.0,
                "msc_threshold": 5.0,
                "arclength_variation_threshold": 0.5,
            },
        },
        "simsopt_queues": queues,
    }
    return policy, manifest_rows


def main() -> None:
    args = _parse_args()
    policy, rows = _policy(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.manifest}")
    print(f"Generated {len(rows)} simsopt cases for wave {args.wave}")


if __name__ == "__main__":
    main()
