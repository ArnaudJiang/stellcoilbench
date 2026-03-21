"""FEM-based structural stress objective for coil optimization.

This module provides ``StructuralStressObjective``, a simsopt-compatible objective
that computes a scalar stress metric from a low-resolution scikit-fem linear
elasticity solve on a coarse tetrahedral coil mesh. Designed for use inside
the coil optimization loop with finite-difference gradients.

J() and dJ() return values in GPa (1 GPa = 10^9 Pa) to reduce penalty magnitude
and improve optimizer numerics when stress violates the threshold.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree

from simsopt._core.derivative import Derivative, derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt.field import BiotSavart

from stellcoilbench.coil_optimization._structural_mesh import _surface_sweep_to_msh
from stellcoilbench.mpi_utils import comm_world, is_mpi_enabled, is_proc0, proc0_print
from stellcoilbench.structural_analysis import write_structural_vtk
from stellcoilbench.structural_analysis._common import (
    _build_coil_centerline_data,
    _compute_Breg_for_coil,
    _compute_coil_frame,
)
from stellcoilbench.structural_analysis._skfem import (
    _compute_von_mises_skfem,
    _load_mesh_skfem,
    _solve_elasticity_skfem,
)

PA_TO_GPA: float = 1e-9
"""Conversion factor from Pa to GPa. Multiply Pa by this to get GPa."""


class StructuralStressObjective(Optimizable):
    """FEM-based structural stress objective for coil optimization.

    Computes a scalar stress metric (max Von Mises, mean Von Mises, or
    Lp-norm of Von Mises) from a low-resolution scikit-fem linear elasticity
    solve on a coarse tetrahedral coil mesh. Designed for use in optimization
    loops with finite-difference gradients.

    The mesh topology is generated once (via Gmsh torus approximation) and
    cached. On subsequent evaluations, mesh node positions are deformed to
    follow the updated coil centerline geometry using nearest-neighbor
    assignment and coil frame rotation.
    """

    def __init__(
        self,
        unique_coils: list,
        bs: Any,
        *,
        all_coils: list | None = None,
        width: float = 0.05,
        height: float = 0.05,
        E: float = 100e9,
        nu: float = 0.3,
        mesh_resolution: float = 0.16,
        msh_path: Path | None = None,
        mesh_resolution_coarse: float | None = None,
        mesh_resolution_fine: float | None = None,
        refine_stress_ratio: float = 0.5,
        stress_metric: str = "mean_von_mises",
        lp_exponent: int = 4,
        fd_step: float = 1e-5,
        eval_interval: int = 1,
        use_cached_K: bool = False,
        structural_backend: str = "dolfinx",
        quadrature_degree: int = 1,
        polynomial_degree: int = 2,
        animation_frames_dir: Path | None = None,
        animation_frame_counter: list[int] | None = None,
        animation_surface_snap: Callable[[int], None] | None = None,
    ) -> None:
        """Initialize the structural stress objective.

        Parameters
        ----------
        unique_coils : list
            Base coil objects for FEM meshing and solving (the ones we compute
            stress on). Symmetry copies are omitted; stress is computed only on
            these unique coils.
        bs : BiotSavart
            simsopt ``BiotSavart`` field evaluator for J×B computation.
        all_coils : list, optional
            Full coil set including symmetry copies, for J×B body-force
            computation. The magnetic field from all coils affects each coil.
            If None, defaults to unique_coils.
        width : float, optional
            Winding-pack full width [m]. Default 0.05.
        height : float, optional
            Winding-pack full height [m]. Default 0.05.
        E : float, optional
            Young's modulus [Pa]. Default 100e9.
        nu : float, optional
            Poisson ratio. Default 0.3.
        mesh_resolution : float, optional
            Gmsh characteristic length [m]. Default 0.16.
            Used only when msh_path is None and adaptive mesh is disabled.
        mesh_resolution_coarse : float, optional
            Coarse mesh resolution [m] for adaptive strategy. Default 0.16.
            When set (with mesh_resolution_fine and refine_stress_ratio),
            optimization starts with this resolution and refines to fine when
            stress reaches refine_stress_ratio times threshold.
        mesh_resolution_fine : float, optional
            Fine mesh resolution [m] after refinement. Defaults to coarse / 2.
        refine_stress_ratio : float, optional
            Refine mesh when stress reaches this fraction of threshold. Default 0.5.
        msh_path : Path, optional
            Path to a pre-built Gmsh ``.msh`` coil mesh file.
            When provided, this mesh is loaded and its topology is cached; node
            positions are deformed each evaluation. When None, a torus mesh is
            generated from coil centerlines.
        stress_metric : str, optional
            One of ``"max_von_mises"``, ``"mean_von_mises"``, ``"lp_von_mises"``.
            Default ``"mean_von_mises"`` (volume-weighted average).
        lp_exponent : int, optional
            Exponent for ``lp_von_mises`` metric. Default 4.
        fd_step : float, optional
            Finite-difference step size for gradient. Default 1e-5.
        eval_interval : int, optional
            Evaluate J() every N optimization iterations; return cached value
            otherwise. Default 1 (every iteration).
        use_cached_K : bool, optional
            When True, cache stiffness K at baseline and reuse for all FD
            perturbations in dJ() (faster but introduces large gradient error).
            When False, assemble fresh K per perturbation (accurate). Default False.
        structural_backend : str, optional
            FEM backend: ``"dolfinx"`` (default when available) or ``"skfem"``.
        quadrature_degree : int, optional
            Quadrature degree for weak-form integration (1=centroid, default).
            Used by DOLFINx backend. Default 1.
        polynomial_degree : int, optional
            Lagrange polynomial order for displacement (DOLFINx backend only).
            scikit-fem uses P1. Default 2.
        animation_frames_dir : Path, optional
            Directory for ``snapshot_structural_%06d.vtk`` when animation export is on.
        animation_frame_counter : list[int], optional
            Single-element list shared across Fourier stages; incremented per export.
        animation_surface_snap : callable, optional
            ``snap(idx)`` writes ``snapshot_surface_{idx:06d}.vts`` (B·n fields).
        """
        self._unique_coils = list(unique_coils)
        self._all_coils = (
            list(all_coils) if all_coils is not None else list(unique_coils)
        )
        self._bs = bs
        self._width = width
        self._height = height
        self._E = E
        self._nu = nu
        self._msh_path = Path(msh_path) if msh_path is not None else None

        # Adaptive mesh: coarse -> fine when stress reaches ratio * threshold
        self._adaptive_mesh = mesh_resolution_coarse is not None
        if self._adaptive_mesh:
            self._mesh_resolution_coarse = float(mesh_resolution_coarse)
            self._mesh_resolution_fine = (
                float(mesh_resolution_fine)
                if mesh_resolution_fine is not None
                else self._mesh_resolution_coarse / 2
            )
            self._refine_stress_ratio = float(refine_stress_ratio)
            self._mesh_resolution = self._mesh_resolution_coarse
            self._refinement_done = False
        else:
            self._mesh_resolution = mesh_resolution
            self._refinement_done = True  # No refinement for non-adaptive
        self._stress_metric = stress_metric
        self._lp_exponent = lp_exponent
        self._fd_step = fd_step
        self._eval_interval = eval_interval
        self._use_cached_K = use_cached_K
        self._structural_backend = str(structural_backend).lower()
        self._quadrature_degree = int(quadrature_degree)
        self._polynomial_degree = int(polynomial_degree)

        self._anim_frames_dir = (
            Path(animation_frames_dir).resolve()
            if animation_frames_dir is not None
            else None
        )
        self._anim_frame_counter = animation_frame_counter
        self._anim_surface_snap = animation_surface_snap
        self._anim_last_export_x: np.ndarray | None = None

        self._cross_section_area = width * height

        if stress_metric not in ("max_von_mises", "mean_von_mises", "lp_von_mises"):
            raise ValueError(
                f"stress_metric must be one of max_von_mises, mean_von_mises, "
                f"lp_von_mises, got {stress_metric!r}"
            )

        self._mesh: Any = None
        self._ref_points: np.ndarray | None = None
        self._nearest_coil_idx: np.ndarray | None = None
        self._nearest_centerline_idx: np.ndarray | None = None
        self._local_offsets: np.ndarray | None = None
        self._node_offset_by_coil: list[int] | None = None

        self._eval_count = 0
        self._cached_J: float | None = None
        self._cached_x: np.ndarray | None = None

        self._build_mesh()

        # Pre-create BiotSavart objects for JxB; they track coil DOFs, so only
        # set_points() is needed per evaluation. One BiotSavart per unique coil:
        # bs_mutual[k] = field from all coils EXCEPT copies of unique k (indices
        # j where j % n_unique == k). This correctly excludes all symmetry copies.
        n_unique = len(self._unique_coils)
        self._bs_mutual: list[BiotSavart] = [
            BiotSavart([c for i, c in enumerate(self._all_coils) if i % n_unique != k])
            for k in range(n_unique)
        ]

        Optimizable.__init__(self, x0=np.array([]), depends_on=list(self._unique_coils))

    def _build_mesh(self) -> None:
        """Generate or load coil mesh and cache topology.

        Default: ParaStell (handles QA/stellarator coils reliably). Fallback:
        per-coil sweep for simple circular/toroidal coils.
        """
        import skfem as _skfem

        if self._msh_path is not None:
            self._build_mesh_from_file(self._msh_path)
            return

        half_w = max(self._width, self._height) / 2.0
        if half_w < 1e-9:
            half_w = 0.15
        width, height = self._width, self._height

        # Default: ParaStell (handles Landreman-Paul QA etc. reliably)
        try:
            from stellcoilbench.finite_build import finite_build_coils_to_msh

            fd = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
            tmp_path = Path(fd.name)
            fd.close()
            try:
                result = finite_build_coils_to_msh(
                    self._unique_coils,
                    tmp_path,
                    width,
                    height,
                    self._mesh_resolution,
                )
                if result is not None:
                    msh_path, _ = result
                    try:
                        self._build_mesh_from_file(msh_path)
                        return
                    finally:
                        msh_path.unlink(missing_ok=True)
            finally:
                tmp_path.unlink(missing_ok=True)
        except ImportError:
            pass

        # Fallback: per-coil sweep (simple coils only)
        all_points: list[np.ndarray] = []
        all_cells: list[np.ndarray] = []
        node_offset = 0
        node_offset_by_coil: list[int] = []
        all_ref_points: list[np.ndarray] = []
        all_nearest_coil: list[np.ndarray] = []
        all_nearest_cl: list[np.ndarray] = []
        all_offsets: list[np.ndarray] = []

        initial_frames: list[dict[str, Any]] = []
        for coil in self._unique_coils:
            initial_frames.append(_compute_coil_frame(coil))

        for k, (coil, frame) in enumerate(zip(self._unique_coils, initial_frames)):
            gamma = frame["gamma"]
            curve = coil.curve
            gammadash = np.asarray(curve.gammadash(), dtype=float).reshape(-1, 3)
            if len(gamma) != len(gammadash) or len(gamma) < 2:
                raise ValueError(
                    f"Coil {k}: invalid gamma/gammadash (len gamma={len(gamma)}, "
                    f"len gammadash={len(gammadash)})"
                )
            msh_path = _surface_sweep_to_msh(
                gamma,
                gammadash,
                width=width,
                height=height,
                mesh_size=self._mesh_resolution,
            )
            if msh_path is None:
                raise ValueError(
                    "ParaStell is required for structural stress on complex coils "
                    "(e.g. Landreman-Paul QA). Sweep mesh failed for coil "
                    f"{k}. Run: bash tools/install_parastell_in_vmec.sh "
                    "with stellcoilbench_vmec active."
                )
            try:
                sk_mesh = _load_mesh_skfem(msh_path)
            finally:
                msh_path.unlink(missing_ok=True)

            pts = sk_mesh.p.T
            cells = sk_mesh.t.T + node_offset

            all_points.append(pts)
            all_cells.append(cells)
            node_offset_by_coil.append(node_offset)
            node_offset += pts.shape[0]

            tree = cKDTree(gamma)
            p_arr = frame["p"]
            q_arr = frame["q"]
            t_arr = frame["t"]

            _, nearest_cl = tree.query(pts, k=1)
            nearest_cl = np.asarray(nearest_cl, dtype=np.intp).flatten()
            delta = pts - gamma[nearest_cl]
            offsets_pq = np.zeros((pts.shape[0], 3))
            offsets_pq[:, 0] = np.sum(delta * p_arr[nearest_cl], axis=1)
            offsets_pq[:, 1] = np.sum(delta * q_arr[nearest_cl], axis=1)
            offsets_pq[:, 2] = np.sum(delta * t_arr[nearest_cl], axis=1)

            all_ref_points.append(pts)
            all_nearest_coil.append(np.full(pts.shape[0], k, dtype=np.intp))
            all_nearest_cl.append(nearest_cl)
            all_offsets.append(offsets_pq)

        combined_points = np.vstack(all_points)
        combined_cells = np.vstack(all_cells)

        self._mesh = _skfem.MeshTet1(combined_points.T, combined_cells.T)
        self._ref_points = combined_points
        self._nearest_coil_idx = np.concatenate(all_nearest_coil)
        self._nearest_centerline_idx = np.concatenate(all_nearest_cl)
        self._local_offsets = np.vstack(all_offsets)
        self._node_offset_by_coil = node_offset_by_coil

    def _build_mesh_from_file(self, msh_path: Path) -> None:
        """Load a pre-built .msh mesh and compute nearest-neighbor assignment.

        Uses a single combined cKDTree over unique coil centerlines to assign each
        mesh node to the nearest coil and compute local (p,q,t) offsets.
        """
        all_gamma, _, _, coil_boundaries = _build_coil_centerline_data(
            self._unique_coils
        )
        all_p_list: list[np.ndarray] = []
        all_q_list: list[np.ndarray] = []
        all_t_list: list[np.ndarray] = []
        for coil in self._unique_coils:
            frame = _compute_coil_frame(coil)
            all_p_list.append(frame["p"])
            all_q_list.append(frame["q"])
            all_t_list.append(frame["t"])
        all_p = np.vstack(all_p_list)
        all_q = np.vstack(all_q_list)
        all_t = np.vstack(all_t_list)

        tree = cKDTree(all_gamma)
        sk_mesh = _load_mesh_skfem(msh_path)
        pts = sk_mesh.p.T  # shape (n_nodes, 3)

        _, nearest_global = tree.query(pts, k=1)
        nearest_global = np.asarray(nearest_global, dtype=np.intp).flatten()

        nearest_coil_idx = (
            np.searchsorted(coil_boundaries, nearest_global, side="right") - 1
        )
        nearest_coil_idx = np.clip(nearest_coil_idx, 0, len(self._unique_coils) - 1)
        nearest_centerline_idx = nearest_global - coil_boundaries[nearest_coil_idx]

        # Validate nearest-neighbor assignments
        n_coils = len(self._unique_coils)
        assert np.all(np.isin(nearest_coil_idx, np.arange(n_coils))), (
            "nearest_coil_idx out of range"
        )
        for k in range(n_coils):
            mask = nearest_coil_idx == k
            if not np.any(mask):
                continue
            n_pts_k = coil_boundaries[k + 1] - coil_boundaries[k]
            assert np.all(
                (nearest_centerline_idx[mask] >= 0)
                & (nearest_centerline_idx[mask] < n_pts_k)
            ), f"Coil {k}: nearest_centerline_idx out of range [0, {n_pts_k})"

        delta = pts - all_gamma[nearest_global]
        offsets_pq = np.zeros((pts.shape[0], 3))
        offsets_pq[:, 0] = np.sum(delta * all_p[nearest_global], axis=1)
        offsets_pq[:, 1] = np.sum(delta * all_q[nearest_global], axis=1)
        offsets_pq[:, 2] = np.sum(delta * all_t[nearest_global], axis=1)

        self._mesh = sk_mesh
        self._ref_points = pts
        self._nearest_coil_idx = nearest_coil_idx
        self._nearest_centerline_idx = nearest_centerline_idx
        self._local_offsets = offsets_pq
        self._node_offset_by_coil = [0]

    def _deform_mesh(self) -> None:
        """Deform cached mesh node positions to follow current coil geometry.

        For each mesh node, find the nearest coil centerline point, compute the
        local offset in the coil frame (p, q, t), then reconstruct the 3D
        position using the updated frame. Replaces the mesh with a new instance
        since scikit-fem mesh.p may not be reliably writable in-place.
        """
        import skfem as _skfem

        new_p = np.zeros_like(self._mesh.p)

        for k, coil in enumerate(self._unique_coils):
            mask = self._nearest_coil_idx == k
            if not np.any(mask):
                continue

            frame = _compute_coil_frame(coil)
            gamma = frame["gamma"]
            p_arr = frame["p"]
            q_arr = frame["q"]
            t_arr = frame["t"]

            local_idx = self._nearest_centerline_idx[mask]
            op = self._local_offsets[mask, 0:1]
            oq = self._local_offsets[mask, 1:2]
            ot = self._local_offsets[mask, 2:3]

            new_pos = (
                gamma[local_idx]
                + op * p_arr[local_idx]
                + oq * q_arr[local_idx]
                + ot * t_arr[local_idx]
            )
            node_indices = np.where(mask)[0]
            new_p[0, node_indices] = new_pos[:, 0]
            new_p[1, node_indices] = new_pos[:, 1]
            new_p[2, node_indices] = new_pos[:, 2]

        # Ensure all nodes were assigned (diagnostic)
        n_assigned = sum(
            np.sum(self._nearest_coil_idx == k) for k in range(len(self._unique_coils))
        )
        assert n_assigned == self._mesh.p.shape[1], (
            f"Not all mesh nodes assigned: {n_assigned} / {self._mesh.p.shape[1]}"
        )

        # Replace mesh with deformed copy (mesh.p may not persist in-place updates)
        old_mesh = self._mesh
        self._mesh = _skfem.MeshTet1(new_p.copy(), old_mesh.t.copy())
        if hasattr(old_mesh, "_structural_cell_tags"):
            self._mesh._structural_cell_tags = old_mesh._structural_cell_tags  # type: ignore[attr-defined]

    def refine_mesh(self, new_resolution: float) -> None:
        """Rebuild mesh at finer resolution. Call from optimization callback when stress
        approaches threshold. Clears cached J and regenerates mesh topology.

        Parameters
        ----------
        new_resolution : float
            Gmsh characteristic length [m] for the refined mesh.
        """
        self._mesh_resolution = float(new_resolution)
        self._cached_J = None
        self._cached_x = None
        self._refinement_done = True
        self._build_mesh()

    def _maybe_emit_animation_vtk(
        self,
        mesh_for_vtk: Any,
        displacement: Any,
        von_mises: Any,
        *,
        animation_export_ok: bool,
    ) -> None:
        """Write paired structural + surface VTK when animation export is configured.

        Called only from scalar stress paths (not gradient assembly). Skips duplicate
        exports for the same DOF vector (e.g. repeated ``J()`` on the same ``x``).
        """
        if not animation_export_ok:
            return
        if self._anim_frames_dir is None or self._anim_frame_counter is None:
            return
        if not is_proc0():
            return
        x = np.asarray(self.full_x, dtype=float)
        if self._anim_last_export_x is not None and x.shape == self._anim_last_export_x.shape:
            if np.allclose(x, self._anim_last_export_x):
                return
        try:
            idx = int(self._anim_frame_counter[0])
            self._anim_frame_counter[0] = idx + 1
            path = self._anim_frames_dir / f"snapshot_structural_{idx:06d}.vtk"
            write_structural_vtk(mesh_for_vtk, displacement, von_mises, path)
            if self._anim_surface_snap is not None:
                self._anim_surface_snap(idx)
            self._anim_last_export_x = x.copy()
        except Exception as exc:
            proc0_print(f"[structural_animation] VTK export failed: {exc}")

    def _evaluate_stress(self) -> float:
        """Run the full FEM pipeline: deform mesh -> J×B -> solve -> Von Mises -> scalar."""
        return self._evaluate_stress_impl(animation_export_ok=True)

    def _evaluate_stress_impl(
        self,
        cached_K=None,
        cached_ib=None,
        cached_fixed_dofs=None,
        cached_K_ff=None,
        cached_free_dofs=None,
        return_assembly: bool = False,
        cached_coil_frames=None,
        cached_Breg_list=None,
        animation_export_ok: bool = False,
    ) -> float | tuple[float, Any, Any, np.ndarray]:
        """Internal FEM pipeline. When return_assembly=True, returns (scalar, K, ib, fixed_dofs)."""
        self._deform_mesh()

        if self._structural_backend == "dolfinx":
            return self._evaluate_stress_dolfinx(
                return_assembly=return_assembly,
                cached_K=cached_K,
                cached_ib=cached_ib,
                cached_fixed_dofs=cached_fixed_dofs,
                cached_coil_frames=cached_coil_frames,
                cached_Breg_list=cached_Breg_list,
                animation_export_ok=animation_export_ok,
            )
        return self._evaluate_stress_skfem(
            cached_K=cached_K,
            cached_ib=cached_ib,
            cached_fixed_dofs=cached_fixed_dofs,
            cached_K_ff=cached_K_ff,
            cached_free_dofs=cached_free_dofs,
            return_assembly=return_assembly,
            cached_coil_frames=cached_coil_frames,
            cached_Breg_list=cached_Breg_list,
            animation_export_ok=animation_export_ok,
        )

    def _evaluate_stress_dolfinx(
        self,
        return_assembly: bool = False,
        cached_K=None,
        cached_ib=None,
        cached_fixed_dofs=None,
        cached_coil_frames=None,
        cached_Breg_list=None,
        animation_export_ok: bool = False,
    ) -> float | tuple[float, Any, Any, np.ndarray]:
        """DOLFINx backend: create mesh from deformed topology, solve, return scalar."""
        from stellcoilbench.structural_analysis import (
            _compute_body_force_dolfinx,
            _compute_stress_dolfinx,
            _create_mesh_from_points_cells,
            _solve_elasticity_dolfinx,
        )

        x = np.asarray(self._mesh.p.T, dtype=np.float64)
        cells = np.asarray(self._mesh.t.T, dtype=np.int64)
        block_ids = getattr(self._mesh, "_structural_cell_tags", None)
        if block_ids is not None:
            block_ids = np.asarray(block_ids, dtype=np.int32)
        mesh_dx = _create_mesh_from_points_cells(x, cells, block_ids=block_ids)
        n_unique = len(self._unique_coils)
        jcross_kwargs: dict[str, Any] = {
            "coils": self._all_coils,
            "bs": self._bs,
            "mesh": mesh_dx,
            "cross_section_area": self._cross_section_area,
            "width": self._width,
            "height": self._height,
            "mesh_coils": self._unique_coils,
            "all_coils": self._all_coils,
            "bs_mutual_list": self._bs_mutual[:n_unique],
            "q_degree": self._quadrature_degree,
        }
        if cached_coil_frames is not None and cached_Breg_list is not None:
            jcross_kwargs["cached_coil_frames"] = cached_coil_frames[:n_unique]
            jcross_kwargs["cached_Breg_list"] = cached_Breg_list[:n_unique]
        body_force = _compute_body_force_dolfinx(**jcross_kwargs)

        u = _solve_elasticity_dolfinx(
            mesh_dx,
            body_force,
            self._E,
            self._nu,
            polynomial_degree=self._polynomial_degree,
        )
        _sigma, vm = _compute_stress_dolfinx(mesh_dx, u, self._E, self._nu)
        if self._stress_metric == "mean_von_mises":
            import dolfinx.fem as _dfem
            import ufl as _ufl

            _one = _dfem.Constant(mesh_dx, np.float64(1.0))
            _vol_form = _dfem.form(_one * _ufl.dx)
            _vol_total = float(_dfem.assemble_scalar(_vol_form))
            if _vol_total > 0.0:
                _vm_form = _dfem.form(vm * _ufl.dx)
                scalar = float(_dfem.assemble_scalar(_vm_form)) / _vol_total
            else:
                vm_arr = np.asarray(vm.x.array)
                scalar = self._stress_scalar_from_vm(vm_arr)
        else:
            vm_arr = np.asarray(vm.x.array)
            scalar = self._stress_scalar_from_vm(vm_arr)
        if return_assembly:
            return scalar, None, None, np.array([]), None, None
        self._maybe_emit_animation_vtk(
            mesh_dx, u, vm, animation_export_ok=animation_export_ok
        )
        return scalar

    def _evaluate_stress_skfem(
        self,
        cached_K=None,
        cached_ib=None,
        cached_fixed_dofs=None,
        cached_K_ff=None,
        cached_free_dofs=None,
        return_assembly: bool = False,
        cached_coil_frames=None,
        cached_Breg_list=None,
        animation_export_ok: bool = False,
    ) -> float | tuple[float, Any, Any, np.ndarray]:
        """scikit-fem backend: solve elasticity, return scalar (and optionally K, ib, fixed_dofs)."""
        body_force_array = np.zeros((self._mesh.p.shape[1], 3))
        n_unique = len(self._unique_coils)
        kwargs = {
            "coils": self._all_coils,
            "bs": self._bs,
            "cross_section_area": self._cross_section_area,
            "width": self._width,
            "height": self._height,
            "bs_mutual_list": self._bs_mutual[:n_unique],
            "mesh_coils": self._unique_coils,
            "all_coils": self._all_coils,
        }
        if cached_coil_frames is not None and cached_Breg_list is not None:
            kwargs["cached_coil_frames"] = cached_coil_frames[:n_unique]
            kwargs["cached_Breg_list"] = cached_Breg_list[:n_unique]
        if (
            cached_K is not None
            and cached_ib is not None
            and cached_fixed_dofs is not None
        ):
            kwargs["cached_K"] = cached_K
            kwargs["cached_ib"] = cached_ib
            kwargs["cached_fixed_dofs"] = cached_fixed_dofs
        if cached_K_ff is not None and cached_free_dofs is not None:
            kwargs["cached_K_ff"] = cached_K_ff
            kwargs["cached_free_dofs"] = cached_free_dofs
        if return_assembly:
            kwargs["return_assembly"] = True

        result = _solve_elasticity_skfem(
            self._mesh,
            body_force_array,
            self._E,
            self._nu,
            **kwargs,
        )
        if return_assembly:
            if len(result) >= 6:
                u_array, K, ib, fixed_dofs, K_ff, free_dofs = result
            else:
                u_array, K, ib, fixed_dofs = result
                K_ff, free_dofs = None, None
        else:
            u_array = result

        vm = _compute_von_mises_skfem(self._mesh, u_array, self._E, self._nu)
        if self._stress_metric == "mean_von_mises":
            scalar = self._volume_weighted_mean_vm_skfem(vm)
        else:
            scalar = self._stress_scalar_from_vm(vm)
        if return_assembly:
            if len(result) >= 6:
                return scalar, K, ib, fixed_dofs, K_ff, free_dofs
            return scalar, K, ib, fixed_dofs
        self._maybe_emit_animation_vtk(
            self._mesh, u_array, vm, animation_export_ok=animation_export_ok
        )
        return scalar

    def _volume_weighted_mean_vm_skfem(self, vm: np.ndarray) -> float:
        """Volume-weighted mean Von Mises stress for scikit-fem per-element array."""
        pts = self._mesh.p[:, self._mesh.t].transpose(2, 1, 0)  # (n_elem, 4, 3)
        X = pts[:, 1:, :] - pts[:, 0:1, :]  # (n_elem, 3, 3)
        vols = np.abs(np.linalg.det(X)) / 6.0
        vol_total = float(np.sum(vols))
        if vol_total > 0.0:
            return float(np.sum(vm * vols) / vol_total)
        return float(np.mean(vm))

    def _stress_scalar_from_vm(self, vm: np.ndarray) -> float:
        """Extract scalar stress metric from per-cell Von Mises array."""
        if self._stress_metric == "max_von_mises":
            return float(np.max(vm))
        if self._stress_metric == "mean_von_mises":
            return float(np.mean(vm))
        if self._stress_metric == "lp_von_mises":
            p = self._lp_exponent
            return float((np.mean(vm**p)) ** (1.0 / p))
        return 0.0

    def J(self) -> float:
        """Return the scalar stress metric value in GPa.

        If eval_interval > 1, returns the cached value between evaluations,
        but only when x has not changed (avoids stale J during L-BFGS-B line
        search). Values are converted from Pa (internal FEM units) to GPa.
        """
        if self._eval_interval <= 1:
            self._cached_J = self._evaluate_stress()
            self._cached_x = np.array(self.full_x, dtype=float).copy()
            return self._cached_J * PA_TO_GPA

        current_x = np.array(self.full_x, dtype=float)
        x_changed = (
            self._cached_x is None
            or len(current_x) != len(self._cached_x)
            or not np.allclose(current_x, self._cached_x)
        )
        if x_changed or self._eval_count % self._eval_interval == 0:
            self._cached_J = self._evaluate_stress()
            self._cached_x = current_x.copy()

        self._eval_count += 1
        assert self._cached_J is not None
        return self._cached_J * PA_TO_GPA

    def _compute_gradient_impl(self, use_cached_K: bool) -> np.ndarray:
        """Compute FD gradient; optionally reuse stiffness K for speed.

        Parameters
        ----------
        use_cached_K : bool
            When True, cache K at baseline and reuse for all perturbations (fast).
            When False, assemble fresh K for each perturbation (accurate reference).

        Returns
        -------
        np.ndarray
            Gradient in GPa/dof, same length as full_x.
        """
        x0 = np.array(self.full_x, dtype=float)
        n = len(x0)
        grad = np.zeros(n)

        self.full_x = x0
        result = self._evaluate_stress_impl(return_assembly=True)
        J0 = result[0]
        K = result[1]
        ib = result[2]
        fixed_dofs = result[3]
        K_ff = result[4] if len(result) > 4 else None
        free_dofs = result[5] if len(result) > 5 else None

        n_total = len(self._all_coils)
        n_unique = len(self._unique_coils)
        baseline_frames = [_compute_coil_frame(c) for c in self._all_coils]
        baseline_Breg = [
            _compute_Breg_for_coil(c, self._width, self._height)
            for c in self._all_coils
        ]

        dof_to_base = np.full(n, -1, dtype=np.intp)
        for opt, (start, end) in self._full_dof_indices.items():
            if opt is self:
                continue
            for k in range(n_unique):
                if (
                    opt is self._unique_coils[k].curve
                    or opt is self._unique_coils[k].current
                ):
                    dof_to_base[start:end] = k
                    break

        use_cache = use_cached_K
        if use_cache:
            cache_K = (K, ib, fixed_dofs, K_ff, free_dofs)
        else:
            cache_K = (None, None, None, None, None)
        K_cached, ib_cached, fixed_dofs_cached, K_ff_cached, free_dofs_cached = cache_K

        try:
            for i in range(n):
                base_b = dof_to_base[i]
                if base_b < 0:
                    grad[i] = 0.0
                    continue

                xp = x0.copy()
                xp[i] = x0[i] + self._fd_step
                self.full_x = xp

                frames = list(baseline_frames)
                Breg = list(baseline_Breg)
                step = n_total // n_unique
                for j in range(step):
                    coil_idx = base_b + j * n_unique
                    frames[coil_idx] = _compute_coil_frame(self._all_coils[coil_idx])
                    Breg[coil_idx] = _compute_Breg_for_coil(
                        self._all_coils[coil_idx], self._width, self._height
                    )

                grad[i] = (
                    self._evaluate_stress_impl(
                        cached_K=K_cached,
                        cached_ib=ib_cached,
                        cached_fixed_dofs=fixed_dofs_cached,
                        cached_K_ff=K_ff_cached,
                        cached_free_dofs=free_dofs_cached,
                        cached_coil_frames=frames,
                        cached_Breg_list=Breg,
                    )
                    - J0
                ) / self._fd_step
        finally:
            self.full_x = x0
        grad *= PA_TO_GPA
        return grad

    def _compute_gradient_chunk(
        self,
        x0: np.ndarray,
        J0: float,
        indices: np.ndarray,
        use_cached_K: bool,
        *,
        skip_baseline_eval: bool = False,
    ) -> np.ndarray:
        """Compute FD gradient for a subset of indices. Used by MPI path.

        Parameters
        ----------
        x0 : np.ndarray
            Baseline DOF vector.
        J0 : float
            Baseline stress value (Pa).
        indices : np.ndarray
            DOF indices to compute (e.g. rank + arange(0, n, size)).
        use_cached_K : bool
            When True, cache K at baseline and reuse.
        skip_baseline_eval : bool, optional
            When True, skip the baseline FEM solve; use J0 as provided and set
            cache_K to None. Used on MPI workers to avoid redundant baseline evals.

        Returns
        -------
        np.ndarray
            Gradient contribution for these indices only; length n, zeros elsewhere.
        """
        n = len(x0)
        grad = np.zeros(n)

        self.full_x = x0
        if skip_baseline_eval:
            K, ib, fixed_dofs, K_ff, free_dofs = None, None, None, None, None
        else:
            result = self._evaluate_stress_impl(return_assembly=True)
            K = result[1]
            ib = result[2]
            fixed_dofs = result[3]
            K_ff = result[4] if len(result) > 4 else None
            free_dofs = result[5] if len(result) > 5 else None

        n_total = len(self._all_coils)
        n_unique = len(self._unique_coils)
        baseline_frames = [_compute_coil_frame(c) for c in self._all_coils]
        baseline_Breg = [
            _compute_Breg_for_coil(c, self._width, self._height)
            for c in self._all_coils
        ]

        dof_to_base = np.full(n, -1, dtype=np.intp)
        for opt, (start, end) in self._full_dof_indices.items():
            if opt is self:
                continue
            for k in range(n_unique):
                if (
                    opt is self._unique_coils[k].curve
                    or opt is self._unique_coils[k].current
                ):
                    dof_to_base[start:end] = k
                    break

        use_cache = use_cached_K
        if use_cache:
            cache_K = (K, ib, fixed_dofs, K_ff, free_dofs)
        else:
            cache_K = (None, None, None, None, None)
        K_cached, ib_cached, fixed_dofs_cached, K_ff_cached, free_dofs_cached = cache_K

        try:
            for i in indices:
                if i >= n:
                    continue
                base_b = dof_to_base[i]
                if base_b < 0:
                    grad[i] = 0.0
                    continue

                xp = x0.copy()
                xp[i] = x0[i] + self._fd_step
                self.full_x = xp

                frames = list(baseline_frames)
                Breg = list(baseline_Breg)
                step = n_total // n_unique
                for j in range(step):
                    coil_idx = base_b + j * n_unique
                    frames[coil_idx] = _compute_coil_frame(self._all_coils[coil_idx])
                    Breg[coil_idx] = _compute_Breg_for_coil(
                        self._all_coils[coil_idx], self._width, self._height
                    )

                grad[i] = (
                    self._evaluate_stress_impl(
                        cached_K=K_cached,
                        cached_ib=ib_cached,
                        cached_fixed_dofs=fixed_dofs_cached,
                        cached_K_ff=K_ff_cached,
                        cached_free_dofs=free_dofs_cached,
                        cached_coil_frames=frames,
                        cached_Breg_list=Breg,
                    )
                    - J0
                ) / self._fd_step
        finally:
            self.full_x = x0
        grad *= PA_TO_GPA
        return grad

    def _collective_dj_body(self, use_cached_K: bool, n: int) -> np.ndarray:
        """Collective FD gradient: all ranks participate. Called by rank 0 from dJ
        and by workers from the MPI worker loop after receiving control=1.

        Parameters
        ----------
        use_cached_K : bool
            Whether to cache stiffness K at baseline.
        n : int
            Number of DOFs (must match on all ranks).

        Returns
        -------
        np.ndarray
            Full gradient (meaningful on rank 0 only; others get it from Allreduce).
        """
        from mpi4py import MPI

        comm = comm_world
        rank = comm.rank
        size = comm.size

        J0_buf = np.empty(1, dtype=np.float64)
        x0_buf = np.empty(n, dtype=np.float64)
        if rank == 0:
            x0 = np.array(self.full_x, dtype=float)
            self.full_x = x0
            result = self._evaluate_stress_impl(return_assembly=True)
            J0_buf[0] = float(result[0])
            x0_buf[:] = x0

        comm.Bcast(J0_buf, root=0)
        comm.Bcast(x0_buf, root=0)
        J0 = float(J0_buf[0])
        x0 = x0_buf.copy()

        indices = np.arange(rank, n, size, dtype=np.intp)
        skip_baseline = rank != 0  # Only rank 0 ran baseline; workers use Bcast J0
        my_grad = self._compute_gradient_chunk(
            x0,
            J0,
            indices,
            use_cached_K=use_cached_K,
            skip_baseline_eval=skip_baseline,
        )

        full_grad = np.zeros(n, dtype=np.float64)
        comm.Allreduce(my_grad, full_grad, op=MPI.SUM)
        return full_grad

    def _compute_gradient_impl_mpi(self, use_cached_K: bool) -> np.ndarray:
        """Compute FD gradient using MPI: signal workers, then run collective body.

        Rank 0 broadcasts control=1, then all ranks run _collective_dj_body.
        Workers are blocked in the worker loop waiting for this signal.
        """
        comm = comm_world
        x0 = np.array(self.full_x, dtype=float)
        n = len(x0)

        control = np.array([1, n], dtype=np.int64)
        comm.Bcast(control, root=0)
        return self._collective_dj_body(use_cached_K=use_cached_K, n=int(control[1]))

    @derivative_dec
    def dJ(self):
        """Compute gradient via forward finite differences (n+1 evaluations).

        Uses full_x (all DOFs including fixed) so that Derivative dict values
        match local_full_dof_size per opt, as required by simsopt's chain rule.
        Fixed DOFs receive zero gradient since they do not affect J.
        When use_cached_K is True, caches stiffness K at baseline for speed;
        when False, assembles fresh K per perturbation for gradient accuracy.
        When MPI is enabled with size > 1, splits perturbations across ranks.
        """
        if is_mpi_enabled() and comm_world.size > 1:
            grad = self._compute_gradient_impl_mpi(use_cached_K=self._use_cached_K)
        else:
            grad = self._compute_gradient_impl(use_cached_K=self._use_cached_K)
        deriv_dict: dict = {}
        for opt, (start, end) in self._full_dof_indices.items():
            if opt is self:
                continue
            if end > start:
                deriv_dict[opt] = grad[start:end].copy()
        return Derivative(deriv_dict)

    return_fn_map = {"J": J, "dJ": dJ}

    def evaluate_and_export_vtk(self, output_path: Path) -> float:
        """Evaluate stress, export mesh/displacement/Von Mises to VTK, return scalar metric.

        Runs the full FEM pipeline: deform mesh, solve elasticity (with J×B body force),
        compute Von Mises stress, write VTK via write_structural_vtk, and return the
        scalar stress metric (max/mean/lp Von Mises as per stress_metric) in GPa.

        Parameters
        ----------
        output_path : Path
            Path to the output VTK file.

        Returns
        -------
        float
            The scalar stress metric value in GPa (same units as J()).
        """
        self._deform_mesh()

        body_force_array = np.zeros((self._mesh.p.shape[1], 3))
        u_array = _solve_elasticity_skfem(
            self._mesh,
            body_force_array,
            self._E,
            self._nu,
            coils=self._all_coils,
            bs=self._bs,
            cross_section_area=self._cross_section_area,
            width=self._width,
            height=self._height,
            bs_mutual_list=self._bs_mutual[: len(self._unique_coils)],
            mesh_coils=self._unique_coils,
            all_coils=self._all_coils,
        )

        vm = _compute_von_mises_skfem(self._mesh, u_array, self._E, self._nu)

        write_structural_vtk(
            self._mesh,
            u_array,
            vm,
            Path(output_path),
        )

        if self._stress_metric == "max_von_mises":
            return float(np.max(vm)) * PA_TO_GPA
        if self._stress_metric == "mean_von_mises":
            return float(np.mean(vm)) * PA_TO_GPA
        if self._stress_metric == "lp_von_mises":
            p = self._lp_exponent
            return float((np.mean(vm**p)) ** (1.0 / p)) * PA_TO_GPA
        return 0.0
