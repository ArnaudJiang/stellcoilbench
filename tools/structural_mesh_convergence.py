#!/usr/bin/env python3
"""Structural mesh convergence study for QA stellarator coils.

Runs FEM structural analysis at multiple mesh resolutions on a real QA stellarator
coil, collects avg/max displacements and avg/max stresses, and plots convergence
versus element size. Writes VTK files (displacement and Von Mises stress) at each
resolution to output_dir/res_<h>/structural_results.vtk. For each resolution, writes VTK files (displacement and Von
Mises stress) to ``<output_dir>/res_<h>/structural_results.vtk``.

Usage:
    conda activate stellcoilbench_vmec
    python tools/structural_mesh_convergence.py --coils structural_correlation_output/run_thr5e08_seed42/coils.json
    python tools/structural_mesh_convergence.py --coils path/to/coils.json --resolutions 0.20,0.12,0.08
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simsopt.field import BiotSavart

import csv
import sys
import time
from pathlib import Path

# Prepend project root for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_COILS = REPO_ROOT / "structural_correlation_output" / "run_thr5e08_seed42" / "coils.json"
DEFAULT_RESOLUTIONS = [0.20, 0.16, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02]


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    args : list[str] | None
        Argument list (e.g. from sys.argv[1:]). If None, uses sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="Structural mesh convergence: avg/max displacement and stress vs resolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--coils",
        type=Path,
        default=None,
        help="Path to coils.json. Default: structural_correlation_output/run_thr5e08_seed42/coils.json if exists",
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=REPO_ROOT / "cases" / "LandremanPaulQA_structural_optimization.yaml",
        help="Case YAML for surface (nfp, stellsym) and finite_build_width",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "structural_convergence_output",
        help="Output directory for table, plots, per-resolution subdirs",
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        default=",".join(f"{r:.2f}" for r in DEFAULT_RESOLUTIONS),
        help="Comma-separated element sizes [m], e.g. 0.20,0.16,0.12,0.10,0.08",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating convergence plots",
    )
    parser.add_argument(
        "--quadrature-degrees",
        "-q",
        type=str,
        dest="quadrature_degrees",
        default="1, 8",
        help="Comma-separated quadrature degrees q for body-force integration (e.g. 1,4,5; q>4 supported)",
    )
    parser.add_argument(
        "--polynomial-degrees",
        "-p",
        type=str,
        dest="polynomial_degrees",
        default="1, 2",
        help="Comma-separated polynomial degrees p for displacement (e.g. 1,2,3)",
    )
    parser.add_argument(
        "--export-full-coil-set",
        action="store_true",
        help="Export structural_results_full.vtk with full coil set (unique coils + symmetry copies).",
    )
    return parser.parse_args(args)


def _load_coils_and_config(
    coils_path: Path,
    case_path: Path,
) -> tuple[list, "BiotSavart", float, float, int, bool]:  # noqa: F821
    """Load coils, BiotSavart, and width/height for structural analysis.

    Returns
    -------
    coils_for_sa : list
        Unique coils for structural analysis.
    bs : BiotSavart
        Magnetic field evaluator.
    width : float
        Winding-pack cross-section width [m].
    height : float
        Winding-pack cross-section height [m].
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether stellarator symmetry is used.
    """
    from simsopt.field import BiotSavart

    from stellcoilbench.case_loader import load_case
    from stellcoilbench.config_scheme import ARIES_CS_MINOR_RADIUS
    from stellcoilbench.coil_optimization._thresholds import (
        _MAX_FINITE_BUILD_WIDTH,
        _MIN_FINITE_BUILD_WIDTH,
    )
    from stellcoilbench.path_utils import (
        get_reference_radii,
        get_surface_filename,
    )
    from stellcoilbench.path_utils import load_surface_with_range, resolve_surface_path
    from stellcoilbench.post_processing import load_bfield_from_coils_json
    from stellcoilbench.post_processing._coil_io import _get_coils_from_bfield
    from stellcoilbench.post_processing._coil_io import get_unique_coils

    bfield = load_bfield_from_coils_json(coils_path)
    coils = _get_coils_from_bfield(bfield)
    if not coils:
        raise ValueError("No coils found in coils.json")

    cfg = load_case(case_path)
    surface_file = get_surface_filename(cfg)
    if not surface_file:
        raise ValueError("Case has no surface_params.surface")

    base_dirs = [REPO_ROOT / "plasma_surfaces", REPO_ROOT]
    resolved = resolve_surface_path(surface_file, base_dirs)
    if resolved is None:
        raise ValueError(f"Surface file not found: {surface_file}")

    s = load_surface_with_range(str(resolved), nphi=32, ntheta=32)
    nfp = int(s.nfp) if hasattr(s, "nfp") else 1
    stellsym = bool(getattr(s, "stellsym", False))
    coils_for_sa = get_unique_coils(coils, nfp=nfp, stellsym=stellsym)
    if not coils_for_sa:
        raise ValueError("No unique coils after symmetry extraction")

    # Width/height: prefer case post_processing, else a0-scaled
    pp_params = getattr(cfg, "post_processing_params", None) or {}
    fb_w = pp_params.get("finite_build_width")
    if fb_w is not None:
        fb_w = float(fb_w)
        fb_h = pp_params.get("finite_build_height", fb_w)
        fb_h = float(fb_h) if fb_h is not None else fb_w
    else:
        _, minor_radius = get_reference_radii(s)
        a0 = ARIES_CS_MINOR_RADIUS / float(minor_radius)
        fb_w = max(_MAX_FINITE_BUILD_WIDTH / a0, _MIN_FINITE_BUILD_WIDTH)
        fb_h = fb_w

    bs = bfield if isinstance(bfield, BiotSavart) else BiotSavart(coils)
    return coils_for_sa, bs, fb_w, fb_h, nfp, stellsym


def _get_plot_configs(
    q_degrees: list[int] | None = None,
    p_degrees: list[int] | None = None,
) -> list[tuple[str, str | None, int | None, int | None]]:
    """Return (label, backend, quadrature_degree, polynomial_degree) configs.

    When DOLFINx is available, returns configs for each (q, p) in q_degrees × p_degrees
    (colors by q, linestyle by p). When only scikit-fem, returns default config.

    Parameters
    ----------
    q_degrees : list[int] | None
        Quadrature degrees for body-force integration. Default [1,2,3,4].
    p_degrees : list[int] | None
        Polynomial degrees for displacement space. Default [1, 2, 3].
    """
    from stellcoilbench.structural_analysis import _DOLFINX_AVAILABLE

    if q_degrees is None:
        q_degrees = [1, 2, 3, 4]
    if p_degrees is None:
        p_degrees = [1, 2, 3]

    configs: list[tuple[str, str | None, int | None, int | None]] = []
    if _DOLFINX_AVAILABLE:
        for q in q_degrees:
            for p in p_degrees:
                configs.append((f"q={q} p={p}", "dolfinx", q, p))
    if not configs:
        configs = [("default", None, None, None)]
    return configs


def run_convergence_study(
    coils_path: Path,
    case_path: Path,
    output_dir: Path,
    resolutions: list[float],
    make_plots: bool,
    *,
    q_degrees: list[int] | None = None,
    p_degrees: list[int] | None = None,
    export_full_coil_set: bool = False,
) -> list[dict]:
    """Run structural analysis at each mesh resolution and collect metrics.

    Runs each (resolution, config) with Winkler spring-foundation BC.
    VTKs are saved in subdirs per config.
    Results include ``label`` and ``use_spring_bc`` keys for plotting.

    Returns
    -------
    list[dict]
        One dict per (resolution, config) with keys: mesh_size_m,
        label, use_spring_bc, n_nodes, avg_displacement_m, max_displacement_m,
        avg_stress_Pa, max_stress_Pa, p95_stress_Pa, skipped, reason.
    """
    from stellcoilbench.finite_build import finite_build_coils_to_msh
    from stellcoilbench.structural_analysis import run_structural_analysis

    coils_for_sa, bs, width, height, nfp, stellsym = _load_coils_and_config(
        coils_path, case_path
    )
    configs = _get_plot_configs(q_degrees=q_degrees, p_degrees=p_degrees)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for mesh_size in resolutions:
        subdir_base = output_dir / f"res_{mesh_size:.3f}".replace(".", "_")
        subdir_base.mkdir(parents=True, exist_ok=True)
        msh_path = subdir_base / "coils.msh"

        # Generate mesh once per resolution
        mesh_result = finite_build_coils_to_msh(
            coils_for_sa,
            msh_path,
            width=width,
            height=height,
            mesh_size=mesh_size,
        )
        if mesh_result is None:
            for label, *_ in configs:
                results.append({
                    "mesh_size_m": mesh_size,
                    "label": label,
                    "use_spring_bc": True,
                    "n_nodes": None,
                    "avg_displacement_m": None,
                    "max_displacement_m": None,
                    "avg_stress_Pa": None,
                    "max_stress_Pa": None,
                    "p95_stress_Pa": None,
                    "skipped": True,
                    "reason": "Mesh generation failed",
                })
            print(f"[res {mesh_size:.3f} m] SKIP: mesh generation failed")
            continue

        n_nodes: int | None = None
        try:
            import meshio
            mesh_data = meshio.read(str(msh_path))
            n_nodes = len(mesh_data.points)
        except Exception:
            pass

        for label, backend_override, quad_degree, poly_degree in configs:
            if label == "default":
                label_slug = "default"
            else:
                # q=1 p=1 -> q1_p1, etc.
                label_slug = label.replace(" ", "_").replace("=", "")
            subdir = subdir_base / label_slug
            subdir.mkdir(parents=True, exist_ok=True)

            result: dict = {
                "mesh_size_m": mesh_size,
                "label": label,
                "use_spring_bc": True,
                "quadrature_degree": quad_degree,
                "polynomial_degree": poly_degree,
                "n_nodes": n_nodes,
                "avg_displacement_m": None,
                "max_displacement_m": None,
                "avg_stress_Pa": None,
                "max_stress_Pa": None,
                "p95_stress_Pa": None,
                "time_s": None,
                "skipped": False,
                "reason": None,
            }

            t0 = time.perf_counter()
            sa_result = run_structural_analysis(
                coils=coils_for_sa,
                bs=bs,
                output_dir=subdir,
                msh_path=msh_path,
                width=width,
                height=height,
                backend=backend_override,
                quadrature_degree=quad_degree,
                polynomial_degree=poly_degree,
                use_spring_bc=True,
                export_full_coil_set=export_full_coil_set,
                nfp=nfp,
                stellsym=stellsym,
            )
            elapsed = time.perf_counter() - t0
            result["time_s"] = elapsed

            if sa_result.get("skipped"):
                result["skipped"] = True
                result["reason"] = sa_result.get("reason", "unknown")
                results.append(result)
                print(f"[res {mesh_size:.3f} m] [{label}] SKIP: {result['reason']}")
                continue

            result["avg_displacement_m"] = sa_result.get("mean_displacement_m")
            result["max_displacement_m"] = sa_result.get("max_displacement_m")
            result["avg_stress_Pa"] = sa_result.get("mean_von_mises_stress_Pa")
            result["max_stress_Pa"] = sa_result.get("max_von_mises_stress_Pa")
            result["p95_stress_Pa"] = sa_result.get("p95_von_mises_stress_Pa")
            result["structural_vtk"] = sa_result.get("structural_vtk")
            result["quadrature_degree"] = quad_degree
            result["polynomial_degree"] = poly_degree
            results.append(result)
            vtk_path = result.get("structural_vtk") or ""
            p95 = result.get("p95_stress_Pa")
            p95_str = f", p95={p95:.3e}" if p95 is not None else ""
            time_str = f", time={elapsed:.2f}s"
            print(
                f"[res {mesh_size:.3f} m] [{label}] max_disp={result['max_displacement_m']:.3e} m, "
                f"max_stress={result['max_stress_Pa']:.3e} Pa{p95_str}{time_str}"
                + (f"  VTK: {vtk_path}" if vtk_path else "")
            )

    return results


def write_csv(results: list[dict], csv_path: Path) -> None:
    """Write convergence table to CSV."""
    cols = [
        "mesh_size_m",
        "label",
        "use_spring_bc",
        "n_nodes",
        "avg_displacement_m",
        "max_displacement_m",
        "avg_stress_Pa",
        "max_stress_Pa",
        "p95_stress_Pa",
        "structural_vtk",
        "time_s",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["skipped", "reason"], extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in cols}
            row["skipped"] = r.get("skipped", False)
            row["reason"] = r.get("reason") or ""
            w.writerow(row)


def plot_convergence(
    results: list[dict],
    plot_path: Path,
    *,
    use_spring_bc: bool | None = None,
) -> None:
    """Generate 1x4 convergence plots (avg/max displacement, avg/max stress vs h).

    Results are grouped by ``label``. Color by quadrature degree q, solid/dashed
    by polynomial degree p.

    Parameters
    ----------
    results : list[dict]
        Per-run results with label, use_spring_bc, etc.
    plot_path : Path
        Output PNG path.
    use_spring_bc : bool | None, optional
        If set, filter results to this BC type only. If None, use all results.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    subset = results
    if use_spring_bc is not None:
        subset = [r for r in results if r.get("use_spring_bc") == use_spring_bc]
    valid = [r for r in subset if not r.get("skipped") and r.get("max_displacement_m") is not None]
    if len(valid) < 2:
        print("[plots] Skipping: need at least 2 successful runs")
        return

    labels = []
    seen = set()
    for r in valid:
        lbl = r.get("label", "default")
        if lbl not in seen:
            seen.add(lbl)
            labels.append(lbl)

    # Colors by q (quadrature degree), solid/dashed by p (polynomial degree)
    Q_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # q=1,2,3,4

    P_LINESTYLES = {1: "-", 2: "--", 3: ":"}

    def style_for_result(r: dict) -> tuple[str, str]:
        """Return (color, linestyle) from result: color by q, linestyle by p."""
        q = r.get("quadrature_degree")
        p = r.get("polynomial_degree")
        if q is not None and q >= 1 and p in P_LINESTYLES:
            color = Q_COLORS[(q - 1) % len(Q_COLORS)]
            ls = P_LINESTYLES.get(p, "-")
            return color, ls
        # Legacy: parse label or fallback
        lbl = r.get("label", "default")
        if "p=2" in lbl or lbl == "DOLFINx (p=2)":
            return "gray", "--"
        if "p=3" in lbl:
            return "gray", ":"
        return "gray", "-"

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    def plot_series(ax, metric_key: str, marker: str = "o") -> None:
        for label in labels:
            subset = [r for r in valid if r.get("label", "default") == label]
            if not subset:
                continue
            h = np.array([r["mesh_size_m"] for r in subset], dtype=float)
            vals = np.array([r[metric_key] for r in subset], dtype=float)
            color, ls = style_for_result(subset[0])
            ax.semilogy(h, vals, marker=marker, color=color, linestyle=ls, label=label)

    plot_series(axes[0], "avg_displacement_m")
    axes[0].set_xlabel("Element size h [m]")
    axes[0].set_ylabel("Mean displacement [m]")
    axes[0].set_title("Average displacement vs mesh resolution")
    axes[0].legend(loc="best")
    axes[0].grid(True, which="both", alpha=0.3)

    plot_series(axes[1], "max_displacement_m")
    axes[1].set_xlabel("Element size h [m]")
    axes[1].set_ylabel("Max displacement [m]")
    axes[1].set_title("Max displacement vs mesh resolution")
    axes[1].legend(loc="best")
    axes[1].grid(True, which="both", alpha=0.3)

    plot_series(axes[2], "avg_stress_Pa")
    axes[2].set_xlabel("Element size h [m]")
    axes[2].set_ylabel("Mean Von Mises stress [Pa]")
    axes[2].set_title("Average stress vs mesh resolution")
    axes[2].legend(loc="best")
    axes[2].grid(True, which="both", alpha=0.3)

    plot_series(axes[3], "max_stress_Pa")
    axes[3].set_xlabel("Element size h [m]")
    axes[3].set_ylabel("Max Von Mises stress [Pa]")
    axes[3].set_title("Max stress vs mesh resolution")
    axes[3].legend(loc="best")
    axes[3].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[plots] Saved {plot_path}")


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (without program name). If None, uses sys.argv[1:].
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)

    coils_path = args.coils
    if coils_path is None:
        if DEFAULT_COILS.exists():
            coils_path = DEFAULT_COILS
            print(f"Using default coils: {coils_path}")
        else:
            print(
                "Error: --coils required. Default path does not exist:\n  "
                f"{DEFAULT_COILS}\nProvide coils.json from a Landreman-Paul run."
            )
            return 1

    if not coils_path.exists():
        print(f"Error: coils file not found: {coils_path}")
        return 1

    try:
        resolutions = [float(x.strip()) for x in args.resolutions.split(",")]
    except ValueError as e:
        print(f"Error: invalid --resolutions: {e}")
        return 1

    if not resolutions:
        print("Error: at least one resolution required")
        return 1

    try:
        q_degrees = [int(x.strip()) for x in args.quadrature_degrees.split(",") if x.strip()]
    except ValueError as e:
        print(f"Error: invalid --quadrature-degrees: {e}")
        return 1

    if not q_degrees:
        print("Error: at least one quadrature degree required")
        return 1

    try:
        p_degrees = [int(x.strip()) for x in args.polynomial_degrees.split(",") if x.strip()]
    except ValueError as e:
        print(f"Error: invalid --polynomial-degrees: {e}")
        return 1

    if not p_degrees:
        print("Error: at least one polynomial degree required")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = run_convergence_study(
            coils_path=coils_path,
            case_path=Path(args.case),
            output_dir=output_dir,
            resolutions=resolutions,
            make_plots=not args.no_plots,
            q_degrees=q_degrees,
            p_degrees=p_degrees,
            export_full_coil_set=getattr(args, "export_full_coil_set", False),
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1

    write_csv(results, output_dir / "convergence_table.csv")
    print("Wrote convergence_table.csv")

    # Timing summary
    successful = [r for r in results if not r.get("skipped") and r.get("time_s") is not None]
    if successful:
        total_s = sum(r["time_s"] for r in successful)
        print("\n[timing] Per-run (mesh_res, label):")
        for r in successful:
            print(f"  res={r['mesh_size_m']:.3f} m  [{r.get('label', '?')}]: {r['time_s']:.2f}s")
        print(f"[timing] Total (all runs): {total_s:.2f}s")

    if not args.no_plots:
        plot_path = output_dir / "convergence_plots.png"
        plot_convergence(results, plot_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
