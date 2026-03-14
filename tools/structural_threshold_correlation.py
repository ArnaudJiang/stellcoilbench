#!/usr/bin/env python3
"""Structural weight sweep and von Mises correlation analysis.

Runs structural coil optimization at varying structural_stress_weight with
structural_stress="l2" and threshold fixed (from case file, default 0).
Collects final metrics (von Mises, torsion, curvature, flux, etc.) and computes
Pearson and Spearman correlations against von Mises stress.

Usage:
    conda activate stellcoilbench_vmec
    python tools/structural_threshold_correlation.py --case cases/LandremanPaulQA_structural_optimization.yaml
    python tools/structural_threshold_correlation.py --weights 1e-9,1e-8,1e-7,1e-6 --max-iterations 100
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

# Prepend project root for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default weight sweep (structural_stress="l2", threshold from case file)
DEFAULT_WEIGHTS = np.logspace(-5, 6, 20, endpoint=True)

VM_COL = "max_von_mises_stress_Pa"
WEIGHT_COL = "structural_stress_weight"

METRIC_KEYS = [
    "final_squared_flux",
    "final_total_length",
    "final_min_cc_separation",
    "final_min_cs_separation",
    "final_max_curvature",
    "final_mean_squared_curvature",
    "final_arclength_variation",
    "final_max_max_coil_force",
    "final_max_max_coil_torque",
    "final_linking_number",
    "final_max_torsion",
    "final_mean_squared_torsion",
    VM_COL,
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Structural threshold sweep and von Mises correlation analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=REPO_ROOT / "cases" / "LandremanPaulQA_structural_optimization.yaml",
        help="Case YAML path",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Comma-separated structural_stress_weight values (weight sweep), e.g. 1e-9,1e-8,1e-7",
    )
    parser.add_argument(
        "--n-weights",
        type=int,
        default=None,
        help="Number of log-spaced weights between --min-weight and --max-weight",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=1e-10,
        help="Minimum weight for log-spaced weight sweep",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=1e-5,
        help="Maximum weight for log-spaced weight sweep",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "sweep_output",
        help="Root output directory for sweep runs",
    )
    parser.add_argument(
        "--run-structural",
        action="store_true",
        default=False,
        help="Run structural post-processing (default: no, von Mises from optimization)",
    )
    parser.add_argument(
        "--no-run-structural",
        action="store_false",
        dest="run_structural",
        help="Skip structural post-processing (default)",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Only recompute metrics from existing coils (re-analysis mode)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Cap iterations per optimization run",
    )
    parser.add_argument(
        "--correlation-output",
        type=Path,
        default=None,
        help="Path for correlation table CSV (default: <output-dir>/correlations.csv)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Generate and display scatter and heatmap plots (default: True)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_false",
        dest="plots",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--no-save-csv",
        action="store_true",
        help="Skip saving sweep_data.csv and correlations.csv to disk",
    )
    return parser.parse_args()


def parse_weights(args: argparse.Namespace) -> list[float]:
    """Parse structural_stress_weight values from args."""
    if args.weights:
        return [float(x.strip()) for x in args.weights.split(",")]
    if args.n_weights is not None:
        return list(
            np.logspace(
                np.log10(args.min_weight),
                np.log10(args.max_weight),
                args.n_weights,
            )
        )
    return DEFAULT_WEIGHTS


def parse_sweep_values(
    args: argparse.Namespace,
) -> tuple[list[float], str, str]:
    """Return (weights, param_col_name, dir_suffix) for weight sweep.

    Returns
    -------
    tuple
        (sweep_values, param_column, dir_suffix)
        e.g. ([1e-8, 1e-7], "structural_stress_weight", "weight")
    """
    values = parse_weights(args)
    return values, WEIGHT_COL, "weight"


def load_and_modify_case(
    case_path: Path,
    *,
    structural_stress_weight: float | None = None,
    max_iterations: int | None = None,
):
    """Load case config and set structural stress for weight sweep.

    Always uses structural_stress="l2" and threshold from case file (default 0).
    """
    from stellcoilbench.case_loader import load_case

    cfg = load_case(case_path)
    terms = dict(cfg.coil_objective_terms or {})
    terms["structural_stress"] = "l2"
    terms["structural_stress_threshold"] = float(terms.get("structural_stress_threshold", 0.0))
    if structural_stress_weight is not None:
        terms["structural_stress_weight"] = structural_stress_weight
    modified_cfg = dataclasses.replace(cfg, coil_objective_terms=terms)

    if max_iterations is not None:
        opt_params = dict(cfg.optimizer_params or {})
        opt_params["max_iterations"] = max_iterations
        modified_cfg = dataclasses.replace(modified_cfg, optimizer_params=opt_params)

    return modified_cfg


def run_optimization(
    case_path: Path,
    case_cfg,
    out_dir: Path,
    run_structural: bool,
) -> dict:
    """Run coil optimization for given case config. Returns results dict."""
    from stellcoilbench.constants import COILS_FILENAME
    from stellcoilbench.coil_optimization import optimize_coils

    coils_out_path = out_dir / COILS_FILENAME
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dict = optimize_coils(
        case_path=case_path,
        coils_out_path=coils_out_path,
        case_cfg=case_cfg,
        output_dir=out_dir,
        run_structural=run_structural,
        skip_post_processing=False,
    )
    return results_dict


def collect_metrics_from_results(results_dict: dict) -> dict:
    """Extract optimization metrics from results dict."""
    metrics = {}
    for key in METRIC_KEYS:
        if key in results_dict:
            metrics[key] = results_dict[key]
    # Von Mises may be top-level or under structural_metrics (from optimization)
    if VM_COL not in metrics and "structural_metrics" in results_dict:
        vm = results_dict["structural_metrics"].get("max_von_mises_stress_Pa")
        if vm is not None:
            metrics[VM_COL] = vm
    # Normalize some keys if present under different names
    if (
        "final_max_max_coil_force" not in metrics
        and "final_max_force_per_coil" in results_dict
    ):
        metrics["final_max_max_coil_force"] = float(
            np.max(results_dict["final_max_force_per_coil"])
        )
    if (
        "final_max_max_coil_torque" not in metrics
        and "final_max_torque_per_coil" in results_dict
    ):
        metrics["final_max_max_coil_torque"] = float(
            np.max(results_dict["final_max_torque_per_coil"])
        )
    return metrics


def collect_von_mises_from_post_processing(out_dir: Path) -> float | None:
    """Read max von Mises stress from post_processing_results.json if present."""
    pp_path = out_dir / "post_processing_results.json"
    if not pp_path.exists():
        return None
    try:
        data = json.loads(pp_path.read_text())
        structural = data.get("structural") or {}
        return structural.get("max_von_mises_stress_Pa")
    except (json.JSONDecodeError, OSError):
        return None


def collect_geometry_metrics_from_coils(
    coils_json_path: Path,
    case_path: Path,
    plasma_surfaces_dir: Path | None = None,
) -> dict:
    """Compute geometry metrics (including torsion) from coils.json via evaluate_external_coils."""
    from stellcoilbench.coil_optimization import evaluate_external_coils
    from stellcoilbench.path_utils import get_surface_filename

    from stellcoilbench.case_loader import load_case

    cfg = load_case(case_path)
    surface_file = get_surface_filename(cfg)
    if not surface_file:
        return {}

    plasma_surfaces_dir = plasma_surfaces_dir or REPO_ROOT / "plasma_surfaces"
    try:
        ext_metrics = evaluate_external_coils(
            coils_json_path,
            surface_file,
            plasma_surfaces_dir=plasma_surfaces_dir,
        )
    except Exception:
        return {}

    out = {}
    for key in METRIC_KEYS:
        if key in ext_metrics:
            out[key] = ext_metrics[key]
    return out


def run_structural_analysis_on_coils(
    coils_json_path: Path,
    out_dir: Path,
    case_path: Path,
) -> float | None:
    """Run structural analysis on coils and return max von Mises stress [Pa]."""
    from stellcoilbench.post_processing import load_bfield_from_coils_json
    from stellcoilbench.post_processing._coil_io import _get_coils_from_bfield
    from stellcoilbench.post_processing._coil_io import get_unique_coils
    from stellcoilbench.path_utils import load_surface_with_range
    from stellcoilbench.path_utils import resolve_surface_path, get_surface_filename
    from stellcoilbench.path_utils import get_reference_radii
    from stellcoilbench.config_scheme import ARIES_CS_MINOR_RADIUS
    from stellcoilbench.coil_optimization._thresholds import (
        _MAX_FINITE_BUILD_WIDTH,
        _MIN_FINITE_BUILD_WIDTH,
    )

    try:
        bfield = load_bfield_from_coils_json(coils_json_path)
        coils = _get_coils_from_bfield(bfield)
        if not coils:
            return None

        from stellcoilbench.case_loader import load_case

        cfg = load_case(case_path)
        surface_file = get_surface_filename(cfg)
        if not surface_file:
            return None

        base_dirs = [REPO_ROOT / "plasma_surfaces", REPO_ROOT]
        resolved = resolve_surface_path(surface_file, base_dirs)
        if resolved is None:
            return None

        s = load_surface_with_range(str(resolved), nphi=32, ntheta=32)
        nfp = int(s.nfp) if hasattr(s, "nfp") else 1
        stellsym = bool(getattr(s, "stellsym", False))
        coils_for_sa = get_unique_coils(coils, nfp=nfp, stellsym=stellsym)
        if not coils_for_sa:
            return None

        _, minor_radius = get_reference_radii(s)
        a0 = ARIES_CS_MINOR_RADIUS / float(minor_radius)
        fb_w = max(
            _MAX_FINITE_BUILD_WIDTH / a0,
            _MIN_FINITE_BUILD_WIDTH,
        )
        fb_h = fb_w

        from stellcoilbench.structural_analysis import run_structural_analysis

        from simsopt.field import BiotSavart

        bs = bfield if isinstance(bfield, BiotSavart) else BiotSavart(coils)
        sa_results = run_structural_analysis(
            coils=coils_for_sa,
            bs=bs,
            output_dir=out_dir,
            width=fb_w,
            height=fb_h,
        )
        return sa_results.get("max_von_mises_stress_Pa")
    except Exception:
        return None


def collect_metrics(
    out_dir: Path,
    case_path: Path,
    results_dict: dict | None = None,
) -> dict:
    """Collect all metrics from a run directory (and optional results dict)."""
    metrics: dict = {}
    coils_path = out_dir / "coils.json"

    # From optimization results
    if results_dict:
        metrics.update(collect_metrics_from_results(results_dict))

    # Von Mises: prefer results_dict (from optimization), then post_processing_results.json
    if VM_COL not in metrics:
        vm = collect_von_mises_from_post_processing(out_dir)
        if vm is not None:
            metrics[VM_COL] = vm

    # Geometry (torsion, etc.) from coils via evaluate_external_coils
    if coils_path.exists():
        geom = collect_geometry_metrics_from_coils(coils_path, case_path)
        for k, v in geom.items():
            if k not in metrics or metrics[k] is None:
                metrics[k] = v

    # If von Mises still missing, run structural analysis
    if VM_COL not in metrics and coils_path.exists():
        vm_sa = run_structural_analysis_on_coils(coils_path, out_dir, case_path)
        if vm_sa is not None:
            metrics[VM_COL] = vm_sa

    return metrics


# Metrics to exclude from correlation table (constant for valid coils, correlation undefined)
_CORRELATION_SKIP = frozenset({"final_linking_number"})

# Metric key -> LaTeX label for scatter plot axes
_METRIC_LATEX_LABELS: dict[str, str] = {
    "final_squared_flux": r"$f_B$",
    "final_total_length": r"$L$",
    "final_min_cc_separation": r"$d_{cc}$",
    "final_min_cs_separation": r"$d_{cs}$",
    "final_max_curvature": r"$\kappa_{\max}$",
    "final_mean_squared_curvature": r"MSC",
    "final_arclength_variation": r"Var($\ell_i$)",
    "final_max_max_coil_force": r"$F_{\max}$",
    "final_max_max_coil_torque": r"$\tau_{\max}$",
    "final_max_torsion": r"$\zeta_{\max}$",
    "final_mean_squared_torsion": r"$\langle\zeta^2\rangle$",
    "max_von_mises_stress_Pa": r"$\sigma_{\mathrm{vm}}$",
    "structural_stress_weight": r"$w_{\sigma_{\mathrm{vm}}}$",
}


def compute_correlations(df, ref_col: str, min_n: int = 3):
    """Compute Pearson and Spearman correlations of each numeric column vs ref_col."""
    import warnings

    import pandas as pd
    import scipy.stats

    sweep_param_cols = {WEIGHT_COL}
    other_cols = [
        c
        for c in df.columns
        if c != ref_col
        and c not in sweep_param_cols
        and c not in _CORRELATION_SKIP
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    rows = []
    for col in other_cols:
        mask = df[[ref_col, col]].notna().all(axis=1)
        n = int(mask.sum())
        if n < min_n:
            continue
        x = df.loc[mask, ref_col].values
        y = df.loc[mask, col].values
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", category=scipy.stats.ConstantInputWarning
                )
                warnings.simplefilter(
                    "ignore", category=scipy.stats.NearConstantInputWarning
                )
                pearson = scipy.stats.pearsonr(x, y)
                spearman = scipy.stats.spearmanr(x, y)
        except Exception:
            continue
        rows.append(
            {
                "metric": col,
                "pearson_r": float(pearson[0]),
                "pearson_p": float(pearson[1]),
                "spearman_r": float(spearman[0]),
                "spearman_p": float(spearman[1])
                if spearman.pvalue is not None
                else np.nan,
                "n_valid": n,
            }
        )
    return pd.DataFrame(rows)


def plot_correlations(
    df, corr_table, output_dir: Path, save_plots: bool = True
) -> None:
    """Generate scatter and heatmap plots."""
    import pandas as pd

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    vm_col = VM_COL
    if vm_col not in df.columns:
        return
    if corr_table.empty or "spearman_r" not in corr_table.columns:
        return

    x_label = _METRIC_LATEX_LABELS.get(vm_col, vm_col) + r" [Pa]"

    # All metrics sorted by |Spearman|
    sorted_corr = corr_table.sort_values(
        by="spearman_r", key=abs, ascending=False
    )
    all_metrics = sorted_corr["metric"].tolist()
    n_plots = len(all_metrics)
    if n_plots == 0:
        return

    ncol = 2
    nrow = (n_plots + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 4 * nrow))
    axes = np.atleast_2d(axes)
    for idx, metric in enumerate(all_metrics):
        if metric not in df.columns:
            continue
        ax = axes.flat[idx]
        mask = df[[vm_col, metric]].notna().all(axis=1)
        x = df.loc[mask, vm_col].values
        y = df.loc[mask, metric].values
        ax.scatter(x, y, alpha=0.7)
        ax.set_xlabel(x_label)
        ax.set_ylabel(_METRIC_LATEX_LABELS.get(metric, metric))
        r = corr_table.loc[corr_table["metric"] == metric, "spearman_r"].iloc[0]
        ax.set_title(f"Spearman r = {r:.3f}")
        if metric == "final_squared_flux":
            ax.set_yscale("log")
        ax.grid(True)
    for j in range(idx + 1, axes.size):
        axes.flat[j].set_visible(False)
    fig.tight_layout()
    if save_plots:
        scatter_path = output_dir / "correlation_scatter.png"
        fig.savefig(scatter_path, dpi=120)

    # Heatmap of correlation matrix (metrics only, exclude constant/skip columns)
    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in _CORRELATION_SKIP
    ]
    if len(numeric_cols) >= 2:
        corr_mat = df[numeric_cols].corr(method="spearman")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        im = ax2.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax2.set_xticks(range(len(numeric_cols)))
        ax2.set_yticks(range(len(numeric_cols)))
        labels = [_METRIC_LATEX_LABELS.get(c, c) for c in numeric_cols]
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        ax2.set_yticklabels(labels)
        ax2.grid(True)
        plt.colorbar(im, ax=ax2, label="Spearman correlation")
        fig2.tight_layout()
        if save_plots:
            heatmap_path = output_dir / "correlation_heatmap.png"
            fig2.savefig(heatmap_path, dpi=120)

    plt.show()


def main() -> None:
    """Main entry point."""
    from stellcoilbench.mpi_utils import comm_world, is_mpi_enabled

    args = parse_args()
    is_rank0 = not is_mpi_enabled() or comm_world.rank == 0

    # All ranks participate in setup and sweep loop (needed for MPI structural dJ workers).
    # Only rank 0 collects metrics, computes correlations, prints, and saves.
    case_path = Path(args.case).resolve()
    if not case_path.is_absolute():
        case_path = (REPO_ROOT / case_path).resolve()
    if not case_path.exists():
        if is_rank0:
            print(f"Error: case file not found: {case_path}", file=sys.stderr)
        sys.exit(1)

    sweep_values, param_col, dir_suffix = parse_sweep_values(args)
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    from stellcoilbench.submission_packaging import _extract_surface_name
    from stellcoilbench.case_loader import load_case

    cfg = load_case(case_path)
    surface_name = _extract_surface_name(cfg)
    case_stem = case_path.stem
    sweep_base = output_dir / surface_name / case_stem
    sweep_base.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, param_val in enumerate(sweep_values):
        param_str = f"{param_val:.0e}".replace("-", "m").replace("+", "p")
        out_dir = sweep_base / f"{dir_suffix}_{param_str}"
        if is_rank0:
            print(
                f"[{i + 1}/{len(sweep_values)}] {param_col} = {param_val:.2e} ...",
                flush=True,
            )

        if not args.skip_optimization:
            modified_cfg = load_and_modify_case(
                case_path,
                structural_stress_weight=param_val,
                max_iterations=args.max_iterations,
            )
            try:
                results_dict = run_optimization(
                    case_path,
                    modified_cfg,
                    out_dir,
                    run_structural=args.run_structural,
                )
                if is_rank0:
                    opt_time = results_dict.get(
                        "optimization_time"
                    ) or results_dict.get("walltime_sec")
                    if opt_time is not None:
                        mins, secs = divmod(float(opt_time), 60)
                        print(
                            f"  Optimization time: {opt_time:.1f}s ({int(mins)}m {int(secs)}s)",
                            flush=True,
                        )
            except Exception as e:
                if is_rank0:
                    print(f"  Optimization failed: {e}", flush=True)
                results_dict = None
        else:
            results_dict = None

        if is_rank0:
            metrics = collect_metrics(out_dir, case_path, results_dict)
            metrics[param_col] = param_val
            rows.append(metrics)

            # Checkpoint: append to CSV after each run (unless --no-save-csv)
            if not args.no_save_csv:
                import pandas as pd

                df_partial = pd.DataFrame(rows)
                df_partial.to_csv(sweep_base / "sweep_data.csv", index=False)

    # Correlation analysis (rank 0 only)
    if is_rank0:
        import pandas as pd

        df = pd.DataFrame(rows)

        if VM_COL not in df.columns:
            print(
                f"Warning: {VM_COL} not found; cannot compute correlations.",
                file=sys.stderr,
            )
            corr_table = pd.DataFrame()
        else:
            corr_table = compute_correlations(df, VM_COL)
            print("\n--- Correlation analysis (vs von Mises stress) ---")
            n_valid = int(df[VM_COL].notna().sum())
            if corr_table.empty:
                print(
                    f"(no correlations computed: need at least 3 runs with valid {VM_COL}, "
                    f"got {n_valid}. Use --weights or --n-weights for more values.)"
                )
            else:
                pd.set_option("display.max_columns", None)
                pd.set_option("display.width", None)
                print(corr_table.to_string(index=False))

            if not args.no_save_csv:
                corr_output = args.correlation_output
                if corr_output is None:
                    corr_output = sweep_base / "correlations.csv"
                else:
                    corr_output = Path(corr_output).resolve()
                    if not corr_output.is_absolute():
                        corr_output = sweep_base / corr_output.name
                corr_output.parent.mkdir(parents=True, exist_ok=True)
                corr_table.to_csv(corr_output, index=False)
                print(f"\nCorrelations saved to {corr_output}")

        if not args.no_save_csv:
            df.to_csv(sweep_base / "sweep_data.csv", index=False)
            print(f"Sweep data saved to {sweep_base / 'sweep_data.csv'}")

        if args.plots and not corr_table.empty:
            plot_correlations(
                df, corr_table, sweep_base, save_plots=not args.no_save_csv
            )
            if not args.no_save_csv:
                print(f"Plots saved to {sweep_base}")

    if is_mpi_enabled():
        comm_world.barrier()


if __name__ == "__main__":
    main()
