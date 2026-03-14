#!/usr/bin/env python3
"""Structural stress vs coil metrics correlation study.

Runs structural (Von Mises) optimizations with varying stress thresholds,
multiple seeds per threshold, and correlates max_von_mises_stress_Pa with
all scalar final metrics: d_cc, d_cs, SquaredFlux, pointwise force, pointwise
torque, curvature, MSC, torsion, length, arclength variation, linking number, etc.

Uses order=4 coils. Run with 8 MPI processes for parallel structural evaluation:
  mpirun -n 8 python tools/force_vm_correlation.py
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING
import json
import os
import sys
from pathlib import Path

import numpy as np

if TYPE_CHECKING:
    from stellcoilbench.config_scheme import CaseConfig

# All scalar final metrics to correlate with Von Mises (excludes per-coil arrays)
TARGET_METRICS = (
    "final_squared_flux",
    "final_min_cc_separation",
    "final_min_cs_separation",
    "final_max_max_coil_force",
    "final_avg_max_coil_force",
    "final_max_max_coil_torque",
    "final_avg_max_coil_torque",
    "final_max_curvature",
    "final_average_curvature",
    "final_mean_squared_curvature",
    "final_max_torsion",
    "final_total_length",
    "final_arclength_variation",
    "final_linking_number",
)
VON_MISES_KEY = "max_von_mises_stress_Pa"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structural stress vs coil metrics correlation: sweep stress threshold, run optimizations.",
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=Path("cases/LandremanPaulQA_structural_optimization.yaml"),
        help="Base case YAML with structural_stress (order 4).",
    )
    parser.add_argument(
        "--stress-thresholds",
        type=str,
        default="5e7,1e8,2e8,5e8, 1e9, 2e9, 5e9",
        help="Comma-separated structural_stress_threshold [Pa]. E.g. 1e8=100MPa. Default 5e7,1e8,2e8,5e8.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of random seeds per threshold (default 3).",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base seed for RNG (default 42).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Max optimizer iterations per run (default 200).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("structural_correlation_output"),
        help="Output directory for runs and summary JSON.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate scatter plots of each metric vs Von Mises.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip runs; load existing data and regenerate plot + correlations.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing data and start fresh (default: merge with existing).",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=8,
        help="Expected MPI processes; run via mpirun -n N (default 8). Informational.",
    )
    return parser.parse_args()


def _load_and_modify_case(
    case_path: Path,
    structural_stress_threshold: float,
    max_iterations: int,
    order: int = 4,
) -> CaseConfig:
    """Load case and set structural_stress + threshold, order."""
    from stellcoilbench.case_loader import load_case
    from stellcoilbench.config_scheme import CaseConfig

    case_cfg = load_case(case_path)
    coil_terms = dict(case_cfg.coil_objective_terms or {})
    coil_terms["structural_stress"] = "l2_threshold"
    coil_terms["structural_stress_threshold"] = structural_stress_threshold
    coil_terms.setdefault("structural_eval_interval", 2)
    coil_terms.setdefault("structural_mesh_resolution_coarse", 0.20)

    opt_params = dict(case_cfg.optimizer_params)
    opt_params["max_iterations"] = max_iterations

    coils_params = dict(case_cfg.coils_params)
    coils_params["order"] = order

    return CaseConfig(
        description=case_cfg.description,
        surface_params=case_cfg.surface_params,
        coils_params=coils_params,
        optimizer_params=opt_params,
        scoring=case_cfg.scoring,
        coil_objective_terms=coil_terms,
        fourier_continuation=None,
        post_processing_params=case_cfg.post_processing_params,
    )


def _run_single(
    case_path: Path,
    case_cfg: CaseConfig,
    output_dir: Path,
    seed: int,
) -> dict:
    """Run one structural optimization and collect metrics."""
    from stellcoilbench.constants import COILS_FILENAME
    from stellcoilbench.coil_optimization import optimize_coils

    np.random.seed(seed)
    thr = case_cfg.coil_objective_terms["structural_stress_threshold"]
    thr_str = f"{thr:.0e}".replace(".", "p").replace("+", "")
    run_dir = output_dir / f"run_thr{thr_str}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    coils_path = run_dir / COILS_FILENAME

    results = optimize_coils(
        case_path=case_path,
        coils_out_path=coils_path,
        case_cfg=case_cfg,
        output_dir=run_dir,
        surface_resolution=32,
        run_structural=True,
        run_vmec=False,
        plot_poincare=False,
        finite_build_width=0.1,
        finite_build_height=0.1,
    )

    sm = results.get("structural_metrics") or {}
    von_mises = sm.get(VON_MISES_KEY) if isinstance(sm, dict) else None

    record = {
        "structural_stress_threshold": thr,
        "seed": seed,
        VON_MISES_KEY: von_mises,
        "run_dir": str(run_dir),
    }
    for m in TARGET_METRICS:
        record[m] = results.get(m)

    record["success"] = von_mises is not None
    return record


def _compute_correlations(records: list[dict]) -> dict[str, dict]:
    """Compute Pearson (and Spearman) correlation of each target metric with Von Mises."""
    vm_vals = []
    for r in records:
        v = r.get(VON_MISES_KEY)
        if v is not None and np.isfinite(v):
            vm_vals.append((float(v), r))
    if len(vm_vals) < 2:
        return {m: {"n_valid": len(vm_vals), "pearson_r": None} for m in TARGET_METRICS}

    out: dict[str, dict] = {}
    for metric in TARGET_METRICS:
        xs, ys = [], []
        for vm, rec in vm_vals:
            val = rec.get(metric)
            if val is not None and np.isfinite(val):
                xs.append(float(vm))
                ys.append(float(val))
        if len(xs) < 2:
            out[metric] = {"n_valid": len(xs), "pearson_r": None}
            continue
        ax = np.array(xs)
        ay = np.array(ys)
        r = float(np.corrcoef(ax, ay)[0, 1])
        entry = {"n_valid": len(xs), "pearson_r": r}
        try:
            from scipy import stats

            sr, sp = stats.spearmanr(ax, ay)
            entry["spearman_r"] = float(sr)
            entry["spearman_p"] = float(sp)
        except ImportError:
            pass
        out[metric] = entry
    return out


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    case_path = (repo_root / args.case) if not args.case.is_absolute() else args.case
    if not case_path.exists():
        print(f"Case not found: {case_path}", file=sys.stderr)
        return 1

    nprocs = os.environ.get("OMPI_COMM_WORLD_SIZE") or os.environ.get("PMI_SIZE")
    if nprocs:
        print(f"Running with {nprocs} MPI processes")
    else:
        print("Hint: run with mpirun -n 8 for parallel structural evaluation")

    thresholds = [float(x.strip()) for x in args.stress_thresholds.split(",")]
    seeds = [args.seed_base + i for i in range(args.seeds)]

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "structural_correlation.json"

    # Load existing data so new runs can be merged (unless --overwrite or --plot-only)
    existing_records: list[dict] = []
    if not args.overwrite and summary_path.exists():
        try:
            with open(summary_path) as f:
                data = json.load(f)
            existing_records = data.get("records", [])
            if existing_records:
                print(f"Loaded {len(existing_records)} existing records from {summary_path}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"Could not load existing data: {e}", file=sys.stderr)

    new_records: list[dict] = []
    if not args.plot_only:
        for thr in thresholds:
            case_cfg = _load_and_modify_case(case_path, thr, args.max_iterations, order=4)
            for seed in seeds:
                print(f"Running structural_stress_threshold={thr:.0e} Pa, seed={seed}...")
                rec = _run_single(case_path, case_cfg, output_dir, seed)
                new_records.append(rec)
                status = "OK" if rec["success"] else "FAIL"
                vals = ", ".join(f"{m}={rec.get(m)}" for m in TARGET_METRICS[:4])  # abbreviate
                print(f"  {status} {VON_MISES_KEY}={rec.get(VON_MISES_KEY)} | {vals}")

        # Merge: existing + new; for same (threshold, seed), new wins
        combined = existing_records + new_records
        by_key: dict[tuple[float, int], dict] = {}
        for r in combined:
            k = (float(r["structural_stress_threshold"]), int(r["seed"]))
            by_key[k] = r
        records = list(by_key.values())
    else:
        records = existing_records
        if not records:
            print("No existing data and --plot-only; nothing to do.", file=sys.stderr)
            return 1
    print(f"Total records: {len(records)}")

    corr = _compute_correlations(records)
    summary = {
        "structural_stress_thresholds": sorted(set(r["structural_stress_threshold"] for r in records)),
        "order": 4,
        "finite_build_width_m": 0.1,
        "finite_build_height_m": 0.1,
        "records": records,
        "correlations_with_von_mises": corr,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    for metric in TARGET_METRICS:
        c = corr.get(metric, {})
        r = c.get("pearson_r")
        if r is not None:
            print(f"  {metric} vs Von Mises: Pearson r={r:.4f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            vm_vals = [
                r[VON_MISES_KEY]
                for r in records
                if r.get(VON_MISES_KEY) is not None and np.isfinite(r[VON_MISES_KEY])
            ]
            if len(vm_vals) < 2:
                print("  Skipping plot (need >= 2 valid points)")
            else:
                n_metrics = len(TARGET_METRICS)
                ncol = min(4, n_metrics)
                nrow = (n_metrics + ncol - 1) // ncol
                fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
                axes = np.atleast_2d(axes)
                for idx, metric in enumerate(TARGET_METRICS):
                    ax = axes[idx // ncol, idx % ncol]
                    xs = [r[VON_MISES_KEY] for r in records if r.get(metric) is not None]
                    ys = [r[metric] for r in records if r.get(metric) is not None]
                    if len(xs) >= 2:
                        ax.scatter(xs, ys, alpha=0.7)
                        r_val = corr.get(metric, {}).get("pearson_r")
                        if r_val is not None:
                            ax.set_title(f"{metric}\nPearson r={r_val:.3f}")
                    ax.set_xlabel(f"{VON_MISES_KEY} [Pa]")
                    ax.set_ylabel(metric)
                for idx in range(n_metrics, nrow * ncol):
                    axes[idx // ncol, idx % ncol].set_visible(False)
                plt.tight_layout()
                plot_path = output_dir / "structural_correlation_scatter.pdf"
                plt.savefig(plot_path)
                plt.close()
                print(f"  Plot saved to {plot_path}")
        except ImportError:
            print("  matplotlib not available, skipping plot")

    return 0


if __name__ == "__main__":
    sys.exit(main())
