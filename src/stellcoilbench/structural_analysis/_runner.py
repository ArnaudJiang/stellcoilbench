"""
Structural analysis runner — top-level orchestrator for FEM structural analysis.

Moved from __init__.py to isolate the coordination logic (mesh loading, elasticity
solve, VTK/XDMF export) from the package's public API.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..constants import WP_POISSON_RATIO, WP_YOUNGS_MODULUS_PA
from ..mpi_utils import proc0_print
from ..utils import timed_section

if TYPE_CHECKING:
    from simsopt.field import BiotSavart, Coil


def run_structural_analysis(
    coils: list[Coil],
    bs: BiotSavart,
    output_dir: Path,
    msh_path: Optional[Path] = None,
    vtk_path: Optional[Path] = None,
    width: float = 0.05,
    height: float = 0.05,
    E: Optional[float] = None,
    nu: Optional[float] = None,
    structural_mesh_resolution_coarse: Optional[float] = None,
    backend: Optional[str] = None,
    quadrature_degree: Optional[int] = None,
    polynomial_degree: Optional[int] = None,
    use_spring_bc: bool = True,
    export_full_coil_set: bool = False,
    nfp: int = 1,
    stellsym: bool = False,
) -> dict[str, Any]:
    """FEM structural analysis of coil geometry (J×B body force, linear elasticity, VTK/XDMF export).

    Loads mesh, computes Lorentz body force, solves elasticity, exports displacement and
    Von Mises stress. Uses Landreman et al. regularized internal field for self-field.
    BC: fixed_supports pins bottom 15% z-range. See _dolfinx/_skfem for implementation details.

    Parameters
    ----------
    coils : list[Coil]
        simsopt Coil objects.
    bs : BiotSavart
        Magnetic field evaluator.
    output_dir : Path
        Output directory.
    msh_path : Path, optional
        Path to .msh file. If None, uses finite_build_coils_parastell.msh if present,
        else generates via sweep fallback (structural_coils.msh).
    vtk_path : Path, optional
        Unused (API compatibility).
    width, height : float
        Winding-pack cross-section [m] (default 0.05 each).
    E, nu : float, optional
        Young's modulus [Pa], Poisson ratio. Default WP_YOUNGS_MODULUS_PA, WP_POISSON_RATIO.
    structural_mesh_resolution_coarse : float, optional
        Element size [m] for tetrahedral mesh. If None, uses default (0.08 m).
    backend : str, optional
        Override backend: ``"dolfinx"`` or ``"skfem"``.
    quadrature_degree : int, optional
        Quadrature degree q for body-force RHS integration. Default 4.
    polynomial_degree : int, optional
        FEM Lagrange order p (DOLFINx only). Default 1. Ignored by scikit-fem.
    use_spring_bc : bool, optional
        Use Winkler spring-foundation BC when True (default).
    export_full_coil_set : bool, optional
        When True and symmetry_factor > 1, write structural_results_full.vtk
        with the full coil set (unique coils + symmetry copies).
    nfp : int, optional
        Number of field periods for toroidal symmetry (default 1).
    stellsym : bool, optional
        Whether stellarator symmetry is used (default False).

    Returns
    -------
    dict[str, Any]
        max_von_mises_stress_Pa, mean_von_mises_stress_Pa, max_displacement_m, backend,
        structural_vtk, displacement_xdmf, von_mises_xdmf.
    """
    from ._pipeline import (
        DEFAULT_STRUCTURAL_MESH_RESOLUTION_M,
        _extract_structural_mesh_arrays,
        _resolve_backend,
        _symmetrize_structural_mesh_to_full_coil_set,
        compute_lorentz_body_force,
        compute_stress_field,
        export_results,
        load_coil_mesh,
        solve_linear_elasticity,
        write_structural_vtk,
        write_structural_vtk_arrays,
    )

    if structural_mesh_resolution_coarse is None:
        structural_mesh_resolution_coarse = DEFAULT_STRUCTURAL_MESH_RESOLUTION_M

    backend = _resolve_backend(backend)
    output_dir = Path(output_dir)

    if E is None:
        E = WP_YOUNGS_MODULUS_PA
    if nu is None:
        nu = WP_POISSON_RATIO

    # Mesh source: (a) msh_path if provided and exists; (b) finite_build_coils_parastell.msh;
    # (c) else generate via sweep fallback.
    meshed_coils = coils
    if msh_path is not None:
        msh_path = Path(msh_path)
        if not msh_path.exists():
            proc0_print(
                f"[structural] Mesh file not found: {msh_path}. Skipping structural analysis."
            )
            return {
                "skipped": True,
                "reason": f"Mesh file not found: {msh_path}",
            }
        proc0_print(f"[structural] Using caller-provided mesh: {msh_path}")
    else:
        parastell_msh = output_dir / "finite_build_coils_parastell.msh"
        if parastell_msh.exists():
            msh_path = parastell_msh
            proc0_print(f"[structural] Using ParaStell mesh: {msh_path}")
        else:
            structural_msh = output_dir / "structural_coils.msh"
            try:
                from stellcoilbench.finite_build import finite_build_coils_to_msh

                result = finite_build_coils_to_msh(
                    coils,
                    structural_msh,
                    width,
                    height,
                    structural_mesh_resolution_coarse,
                )
                if result is not None:
                    msh_path, meshed_indices = result
                    meshed_coils = [coils[i] for i in meshed_indices]
                    proc0_print(
                        f"[structural] Generated mesh via sweep fallback (resolution={structural_mesh_resolution_coarse} m)"
                    )
                else:
                    proc0_print(
                        "[structural] Mesh generation failed for all coils. "
                        "Skipping structural analysis."
                    )
                    return {
                        "skipped": True,
                        "reason": "No mesh file available. Sweep fallback failed for all coils.",
                    }
            except (ImportError, ValueError, Exception) as e:
                proc0_print(
                    f"[structural] Mesh generation failed ({type(e).__name__}: {e}). "
                    "Skipping structural analysis."
                )
                return {
                    "skipped": True,
                    "reason": f"Mesh generation failed: {type(e).__name__}: {e}",
                }

    assert msh_path is not None  # guaranteed by logic above
    cross_section_area = width * height

    proc0_print(f"[structural] Loading mesh from {msh_path}  (backend={backend})")
    with timed_section("structural_load_mesh"):
        mesh = load_coil_mesh(msh_path, backend=backend)

    n_nodes: int = 0
    if hasattr(mesh, "p"):
        n_nodes = int(mesh.p.shape[1])
    elif hasattr(mesh, "geometry") and hasattr(mesh.geometry, "x"):
        n_nodes = int(mesh.geometry.x.shape[0])
    if n_nodes > 100_000:
        proc0_print(
            f"[structural] WARNING: Large mesh ({n_nodes:,} nodes); J×B and solve may "
            "take many minutes. Use structural_mesh_resolution_coarse=0.08 to generate a coarser mesh."
        )

    # Full coil set (including symmetry copies) for physically correct mutual B
    all_coils_list = getattr(bs, "coils", None) or coils

    proc0_print(
        "[structural] Computing Lorentz body force (J × B) [regularized internal field] ..."
    )
    with timed_section("structural_body_force"):
        body_force = compute_lorentz_body_force(
            coils,
            bs,
            mesh,
            cross_section_area,
            width=width,
            height=height,
            use_regularized=True,
            mesh_coils=meshed_coils,
            all_coils=all_coils_list,
            backend=backend,
            quadrature_degree=quadrature_degree,
        )

    proc0_print("[structural] Solving linear elasticity ...")
    with timed_section("structural_solve"):
        displacement = solve_linear_elasticity(
            mesh,
            body_force,
            E=E,
            nu=nu,
            coils=coils,
            bs=bs,
            cross_section_area=cross_section_area,
            width=width,
            height=height,
            mesh_coils=meshed_coils,
            all_coils=all_coils_list,
            backend=backend,
            polynomial_degree=polynomial_degree,
            use_spring_bc=use_spring_bc,
        )

    proc0_print("[structural] Computing stress fields ...")
    with timed_section("structural_stress"):
        sigma, von_mises = compute_stress_field(
            mesh, displacement, E=E, nu=nu, backend=backend
        )

    proc0_print("[structural] Exporting results ...")
    with timed_section("structural_export"):
        structural_dir = output_dir / "structural"
        structural_dir.mkdir(parents=True, exist_ok=True)
        file_paths = export_results(
            mesh, displacement, von_mises, structural_dir, backend=backend
        )

        # Standalone tetrahedral VTK with displacement + Von Mises
        structural_vtk = output_dir / "structural_results.vtk"
        write_structural_vtk(mesh, displacement, von_mises, structural_vtk)
        file_paths["structural_vtk"] = str(structural_vtk)
        proc0_print(f"[structural] Wrote {structural_vtk}")

        symmetry_factor = nfp * (2 if stellsym else 1)
        if export_full_coil_set and symmetry_factor > 1:
            points, cells, u_array, vm_array = _extract_structural_mesh_arrays(
                mesh, displacement, von_mises
            )
            pts_full, cells_full, disp_full, vm_full = (
                _symmetrize_structural_mesh_to_full_coil_set(
                    points, cells, u_array, vm_array, nfp, stellsym
                )
            )
            structural_vtk_full = output_dir / "structural_results_full.vtk"
            write_structural_vtk_arrays(
                pts_full, cells_full, disp_full, vm_full, structural_vtk_full
            )
            file_paths["structural_vtk_full"] = str(structural_vtk_full)
            proc0_print(f"[structural] Wrote full coil set: {structural_vtk_full}")

    # Summary metrics
    if backend == "dolfinx":
        u_array = displacement.x.array.reshape(-1, 3)
        vm_array = von_mises.x.array
    else:
        u_array = displacement
        vm_array = von_mises

    disp_mags = np.linalg.norm(u_array, axis=1)
    max_disp = float(np.max(disp_mags))
    std_disp = float(np.std(disp_mags))
    max_vm = float(np.max(vm_array))
    # 95th-percentile Von Mises stress: robust alternative to max that is unaffected
    # by the clamped-free BC stress singularity (which inflates max_vm on fine meshes).
    p95_vm = float(np.percentile(vm_array, 95)) if vm_array.size > 0 else max_vm

    # Volume-weighted mean displacement and Von Mises stress.
    # Simple nodal/cell averages (np.mean) are not volume-weighted: on non-uniform
    # meshes (or when BC-adjacent cells are small), they over-weight fine-mesh regions.
    # The correct metric is ∫|u| dV / ∫dV  and  ∫σ_vm dV / ∫dV.
    mean_disp: float
    mean_vm: float
    if backend == "dolfinx":
        try:
            import dolfinx.fem as _dfem
            import ufl as _ufl

            _one = _dfem.Constant(mesh, np.float64(1.0))
            _vol_form = _dfem.form(_one * _ufl.dx)
            _vol_total = float(_dfem.assemble_scalar(_vol_form))
            if _vol_total > 0.0:
                _u_mag_form = _dfem.form(
                    _ufl.sqrt(_ufl.inner(displacement, displacement)) * _ufl.dx
                )
                mean_disp = float(_dfem.assemble_scalar(_u_mag_form)) / _vol_total
                _vm_form = _dfem.form(von_mises * _ufl.dx)
                mean_vm = float(_dfem.assemble_scalar(_vm_form)) / _vol_total
            else:
                mean_disp = float(np.mean(disp_mags))
                mean_vm = float(np.mean(vm_array))
        except Exception:
            mean_disp = float(np.mean(disp_mags))
            mean_vm = float(np.mean(vm_array))
    else:
        # scikit-fem: volume-weighted mean (vm_array is per-element)
        pts = mesh.p[:, mesh.t].transpose(2, 1, 0)  # (n_cells, 4, 3)
        X = pts[:, 1:, :] - pts[:, 0:1, :]  # (n_cells, 3, 3)
        vols = np.abs(np.linalg.det(X)) / 6.0
        vol_total = float(np.sum(vols))
        if vol_total > 0.0:
            mean_vm = float(np.sum(vm_array * vols)) / vol_total
        else:
            mean_vm = float(np.mean(vm_array))
        mean_disp = float(np.mean(disp_mags))

    summary: dict[str, Any] = {
        "max_von_mises_stress_Pa": max_vm,
        "p95_von_mises_stress_Pa": p95_vm,
        "mean_von_mises_stress_Pa": mean_vm,
        "max_displacement_m": max_disp,
        "mean_displacement_m": mean_disp,
        "std_displacement_m": std_disp,
        "youngs_modulus_Pa": E,
        "poisson_ratio": nu,
        "bc_type": "spring_foundation" if use_spring_bc else "fixed_supports",
        "backend": backend,
    }
    summary.update(file_paths)

    proc0_print(
        f"[structural] Done.  max σ_vm = {max_vm:.3e} Pa, max |u| = {max_disp:.3e} m"
    )
    return summary
