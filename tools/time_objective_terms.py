#!/usr/bin/env python3
"""Time J() and dJ() for each coil objective term individually.

Loads a case config (default: basic_LandremanPaulQA_structural.yaml), builds
the full coil optimization objective setup (with force, torque, torsion, and
other objectives merged in for timing when not in the case), and reports
wall-clock times for flux, CC/CS distance, length, curvature, MSC, arclength
variation, linking number, force, torque, torsion, and structural stress.

Run: conda activate stellcoilbench_vmec; python tools/time_objective_terms.py
     python tools/time_objective_terms.py --case cases/basic_tokamak.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

import numpy as np

# Prepend project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _setup(case_path: Path) -> dict | None:
    """Load case, initialize coils, build objectives. Returns setup dict or None on failure."""
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = devnull, devnull

        from stellcoilbench.case_loader import load_case
        from stellcoilbench.coil_optimization._config_parsing import (
            _prepare_optimization_config,
        )
        from stellcoilbench.coil_optimization._optimization_loop import (
            _get_regularization_circ,
            initialize_coils_loop,
        )
        from stellcoilbench.coil_optimization._optimization_setup import (
            _build_objectives_and_constraints,
            _setup_biotSavart_and_initial_save,
        )
        from stellcoilbench.coil_optimization._scipy_optimizer import (
            _parse_optimizer_config,
        )
    finally:
        sys.stdout, sys.stderr = old_out, old_err

    root = Path(__file__).resolve().parents[1]
    case_path = (root / case_path) if not case_path.is_absolute() else case_path
    if not case_path.exists():
        print(f"Error: case file not found: {case_path}", file=sys.stderr)
        devnull.close()
        return None

    case_cfg = load_case(case_path, validate=False)

    # Merge in force, torque, torsion, structural, and other objectives for timing
    timing_terms = {
        "coil_coil_force": "lp_threshold",
        "coil_coil_torque": "lp_threshold",
        "coil_torsion": "lp_threshold",
        "force_threshold": 200.0,
        "torque_threshold": 200.0,
        "structural_stress": "l2_threshold",
        "structural_stress_metric": "mean_von_mises",
        "structural_stress_threshold": 0.0,
        "structural_mesh_resolution_coarse": 0.20,
        "structural_quadrature_degree": 1,
        "structural_eval_interval": 1,
    }
    terms = dict(case_cfg.coil_objective_terms or {})
    for k, v in timing_terms.items():
        if k not in terms:
            terms[k] = v
    case_cfg = dataclasses.replace(case_cfg, coil_objective_terms=terms)

    out_dir = root / "submissions" / "per_term_timing" / case_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    coils_out = out_dir / "coils.json"

    config = _prepare_optimization_config(
        case_cfg,
        case_path,
        case_path.resolve(),
        coils_out,
        out_dir,
        64,
    )

    s = config["surface"]
    coil_params = config["coil_params"]
    ncoils = coil_params["ncoils"]
    order = coil_params["order"]
    target_B = coil_params.get("target_B", 5.7)
    regularization = _get_regularization_circ()
    coil_objective_terms = config.get("coil_objective_terms") or {}
    threshold_kwargs = config.get("threshold_kwargs") or {}

    # Parse optimizer config to get full thresholds.
    # Use finite_build_width=0.2 so structural mesh 0.16 m is not clamped
    # (clamp triggers when mesh_res > width; table specifies Δx=0.16 m).
    kwargs = dict(threshold_kwargs)
    kwargs.setdefault("finite_build_width", 0.1)
    parse_config = _parse_optimizer_config(
        s,
        kwargs,
        max_iterations=10,
        coil_objective_terms=coil_objective_terms,
    )
    th = parse_config["thresholds"]
    coil_width = th.get("coil_width", 0.4 / th["a0"])

    try:
        sys.stdout, sys.stderr = devnull, devnull

        coils = initialize_coils_loop(
            s,
            out_dir=out_dir,
            target_B=target_B,
            ncoils=ncoils,
            order=order,
            coil_width=coil_width,
            regularization=regularization,
        )
        bs, curves, _ = _setup_biotSavart_and_initial_save(
            coils, s, s, 64, 64, out_dir, save_coils_surface_vtk=False
        )
        base_curves = [coil.curve for coil in coils[:ncoils]]
        total_current = sum(c.current.get_value() for c in coils[:ncoils])
        major_radius = th["major_radius"]

        obj_result = _build_objectives_and_constraints(
            s,
            bs,
            coils,
            coils,
            base_curves,
            curves,
            ncoils,
            total_current,
            major_radius,
            coil_objective_terms,
            th,
            {},
            out_dir=None,
        )
    finally:
        sys.stdout, sys.stderr = old_out, old_err

    c_list = obj_result["c_list"]
    constraint_names_and_thresholds = obj_result["constraint_names_and_thresholds"]
    structural_obj_raw = obj_result.get("structural_obj_raw")

    devnull.close()
    return {
        "c_list": c_list,
        "constraint_names_and_thresholds": constraint_names_and_thresholds,
        "structural_obj_raw": structural_obj_raw,
        "case_path": case_path,
    }


def _time_j(
    c, x0: np.ndarray, parent, n_warmup: int, n_timed: int
) -> tuple[float, float]:
    """Time J() for objective c; return (median_s, j_value)."""
    parent.full_x = x0
    for _ in range(n_warmup):
        c.J()
    times = []
    for _ in range(n_timed):
        parent.full_x = x0
        t0 = time.perf_counter()
        j_val = c.J()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(j_val)


def _time_dj(c, x0: np.ndarray, parent, n_warmup: int, n_timed: int) -> float:
    """Time dJ() for objective c; return median_s."""
    for _ in range(n_warmup):
        parent.full_x = x0
        c.dJ()
    times = []
    for _ in range(n_timed):
        parent.full_x = x0
        t0 = time.perf_counter()
        c.dJ()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time J() and dJ() per coil objective term"
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=Path("cases/basic_LandremanPaulQA.yaml"),
        help="Case YAML path (default: cases/basic_LandremanPaulQA.yaml)",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=1,
        help="Warmup calls before timing (default: 1)",
    )
    parser.add_argument(
        "--n-timed",
        type=int,
        default=10,
        help="Number of timed J() runs for median (default: 3)",
    )
    parser.add_argument(
        "--n-dj-timed",
        type=int,
        default=10,
        help="Number of timed dJ() runs for median (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print tracebacks on errors",
    )
    args = parser.parse_args()

    print("Loading case and building objectives...")
    setup = _setup(args.case)
    if setup is None:
        sys.exit(1)

    c_list = setup["c_list"]
    constraint_names_and_thresholds = setup["constraint_names_and_thresholds"]
    structural_obj_raw = setup["structural_obj_raw"]
    case_path = setup["case_path"]

    from simsopt.objectives import Weight

    JF = sum(Weight(1.0) * c for c in c_list)
    x0 = np.array(JF.full_x, dtype=float).copy()
    n_dofs = len(x0)

    print(f"\nPer-term J() / dJ() timing ({case_path.name})")
    print(f"n_dofs: {n_dofs}")

    # Clarify structural FEM mesh resolution when present
    if structural_obj_raw is not None:
        raw = structural_obj_raw
        while hasattr(raw, "objective"):
            raw = raw.objective
        mesh_res = getattr(raw, "_mesh_resolution", None)
        mesh_coarse = getattr(raw, "_mesh_resolution_coarse", None)
        mesh_fine = getattr(raw, "_mesh_resolution_fine", None)
        if mesh_coarse is not None and mesh_fine is not None:
            print(
                f"Structural FEM: adaptive mesh coarse={mesh_coarse} m, fine={mesh_fine} m"
            )
        elif mesh_res is not None:
            print(f"Structural FEM: mesh resolution={mesh_res} m")
    print()

    name_width = 20
    time_width = 10
    val_width = 12
    header = (
        f"{'Term':<{name_width}} "
        f"{'J() [s]':<{time_width}} "
        f"{'dJ() [s]':<{time_width}} "
        f"{'J value':<{val_width}}"
    )
    print(header)
    print("-" * (name_width + time_width * 2 + val_width + 3))

    total_j_time = 0.0
    total_dj_time = 0.0

    for i in range(len(c_list)):
        if i == 0:
            name = "Flux"
        elif i - 1 < len(constraint_names_and_thresholds):
            name = constraint_names_and_thresholds[i - 1][0]
        else:
            name = f"c[{i}]"

        # Use raw FEM for σ_vm timing: the wrapped constraint short-circuits
        # (Guard when d_cc small, or QuadraticPenalty when below threshold)
        # and would report misleadingly fast dJ().
        obj_to_time = c_list[i]
        if name == "σ_vm" and structural_obj_raw is not None:
            obj_to_time = structural_obj_raw

        try:
            t_j, j_val = _time_j(obj_to_time, x0, JF, args.n_warmup, args.n_timed)
            t_dj = _time_dj(obj_to_time, x0, JF, args.n_warmup, args.n_dj_timed)
        except Exception as e:
            import traceback

            print(f"{name:<{name_width}} (error: {e})")
            if args.verbose:
                traceback.print_exc()
            continue

        total_j_time += t_j
        total_dj_time += t_dj

        j_str = (
            f"{j_val:.2e}" if abs(j_val) < 1e-2 or abs(j_val) >= 1e4 else f"{j_val:.4f}"
        )
        print(
            f"{name:<{name_width}} {t_j:<{time_width}.4f} {t_dj:<{time_width}.4f} {j_str:<{val_width}}"
        )

    # Total JF
    try:
        JF.full_x = x0
        for _ in range(args.n_warmup):
            JF.J()
        t0 = time.perf_counter()
        for _ in range(args.n_timed):
            JF.full_x = x0
            j_total = JF.J()
        t_j_total = (time.perf_counter() - t0) / args.n_timed

        for _ in range(args.n_warmup):
            JF.full_x = x0
            JF.dJ()
        t0 = time.perf_counter()
        JF.full_x = x0
        JF.dJ()
        t_dj_total = time.perf_counter() - t0

        j_str = (
            f"{j_total:.2e}"
            if abs(j_total) < 1e-2 or abs(j_total) >= 1e4
            else f"{j_total:.4f}"
        )
        print("-" * (name_width + time_width * 2 + val_width + 3))
        print(
            f"{'Total JF':<{name_width}} {t_j_total:<{time_width}.4f} {t_dj_total:<{time_width}.4f} {j_str:<{val_width}}"
        )
    except Exception as e:
        print(f"Total JF (error: {e})")


if __name__ == "__main__":
    main()
