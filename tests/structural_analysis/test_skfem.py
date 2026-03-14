"""Tests for scikit-fem Von Mises, elasticity solver, export, and mesh loading."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


class TestSkfemVonMises:
    """Test the scikit-fem Von Mises element-wise computation."""

    def test_single_tet_known_displacement(self):
        """Compute Von Mises for a single tetrahedron with known displacement."""
        from stellcoilbench.structural_analysis import _compute_von_mises_skfem

        skfem = pytest.importorskip("skfem")

        # Unit tetrahedron
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        cells = np.array([[0, 1, 2, 3]])
        mesh = skfem.MeshTet1(points.T, cells.T)

        E = 200e9
        nu = 0.3

        # Uniaxial extension in x
        u = np.zeros((4, 3))
        u[1, 0] = 0.001  # node 1 at (1,0,0) displaced by 0.001 in x

        vm = _compute_von_mises_skfem(mesh, u, E, nu)
        assert vm.shape == (1,)
        assert vm[0] > 0  # should produce non-zero stress

    def test_zero_displacement_gives_zero_stress(self):
        """Zero displacement → zero Von Mises stress."""
        from stellcoilbench.structural_analysis import _compute_von_mises_skfem

        skfem = pytest.importorskip("skfem")

        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        cells = np.array([[0, 1, 2, 3]])
        mesh = skfem.MeshTet1(points.T, cells.T)

        u = np.zeros((4, 3))
        vm = _compute_von_mises_skfem(mesh, u, 100e9, 0.3)
        assert np.isclose(vm[0], 0.0, atol=1e-3)


class TestSkfemSolveElasticity:
    """Test the scikit-fem linear-elasticity solver."""

    def test_zero_force_zero_displacement(self):
        """Zero body force should give zero displacement (up to numerics)."""
        from stellcoilbench.structural_analysis import _solve_elasticity_skfem

        skfem = pytest.importorskip("skfem")

        # Minimal tet mesh: two tets sharing a face
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 1],
            ],
            dtype=float,
        )
        cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        mesh = skfem.MeshTet1(points.T, cells.T)

        body_force = np.zeros((5, 3))
        u = _solve_elasticity_skfem(mesh, body_force, E=100e9, nu=0.3)
        assert u.shape == (5, 3)
        assert np.allclose(u, 0.0, atol=1e-10)


class TestExportSkfem:
    """Test the scikit-fem VTK export."""

    def test_export_creates_file(self, tmp_path):
        """export_results should write a VTK file when using skfem backend."""
        _skfem = pytest.importorskip("skfem")
        pytest.importorskip("meshio")

        from stellcoilbench.structural_analysis import _export_skfem

        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        cells = np.array([[0, 1, 2, 3]])
        mesh = _skfem.MeshTet1(points.T, cells.T)

        u = np.zeros((4, 3))
        vm = np.array([42.0])

        paths = _export_skfem(mesh, u, vm, tmp_path)
        assert "structural_vtk" in paths
        vtk_path = Path(paths["structural_vtk"])
        assert vtk_path.exists()
        assert vtk_path.suffix == ".vtk"


class TestSkfemEndToEnd:
    """End-to-end test: mesh → solve → stress → export using scikit-fem."""

    def test_gravity_like_load_produces_nonzero_displacement_and_stress(self, tmp_path):
        """Apply a uniform body force and verify displacement + Von Mises > 0."""
        _skfem = pytest.importorskip("skfem")

        from stellcoilbench.structural_analysis import (
            _solve_elasticity_skfem,
            _compute_von_mises_skfem,
            _export_skfem,
        )

        pts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
            dtype=float,
        )
        cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        mesh = _skfem.MeshTet1(pts.T, cells.T)

        body_force = np.zeros((5, 3))
        body_force[:, 2] = -1e6  # uniform downward load

        u = _solve_elasticity_skfem(mesh, body_force, E=100e9, nu=0.3)
        assert u.shape == (5, 3)
        assert np.max(np.abs(u)) > 0, "Non-zero force should give non-zero displacement"

        vm = _compute_von_mises_skfem(mesh, u, E=100e9, nu=0.3)
        assert vm.shape == (2,)
        assert np.any(vm > 0), "Non-zero displacement should give non-zero Von Mises"

        paths = _export_skfem(mesh, u, vm, tmp_path)
        vtk_path = Path(paths["structural_vtk"])
        assert vtk_path.exists()


class TestMeshLoading:
    """Test mesh loading functions."""

    def test_load_mesh_skfem_with_tet(self, tmp_path):
        """Load a simple .msh file via the skfem path."""
        pytest.importorskip("skfem")
        _meshio = pytest.importorskip("meshio")

        from stellcoilbench.structural_analysis import _load_mesh_skfem

        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        cells = [("tetra", np.array([[0, 1, 2, 3]]))]
        m = _meshio.Mesh(points, cells)
        msh_path = tmp_path / "test.msh"
        _meshio.gmsh.write(str(msh_path), m)

        loaded = _load_mesh_skfem(msh_path)
        assert loaded.p.shape[1] == 4  # 4 nodes
        assert loaded.t.shape[1] == 1  # 1 element

    def test_load_mesh_skfem_no_tet_raises(self, tmp_path):
        """Should raise ValueError if no tet cells found."""
        _meshio = pytest.importorskip("meshio")

        from stellcoilbench.structural_analysis import _load_mesh_skfem

        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        cells = [("triangle", np.array([[0, 1, 2]]))]
        m = _meshio.Mesh(points, cells)
        msh_path = tmp_path / "tri.msh"
        _meshio.gmsh.write(str(msh_path), m)

        with pytest.raises(ValueError, match="No tetrahedral cells"):
            _load_mesh_skfem(msh_path)

    def test_extract_tet_blocks_from_meshio(self, tmp_path):
        """_extract_tet_blocks_from_meshio returns cells, block_ids, points."""
        _meshio = pytest.importorskip("meshio")

        from stellcoilbench.structural_analysis._common import (
            _extract_tet_blocks_from_meshio,
        )

        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        cells = [("tetra", np.array([[0, 1, 2, 3]]))]
        m = _meshio.Mesh(points, cells)
        msh_path = tmp_path / "tet.msh"
        _meshio.gmsh.write(str(msh_path), m)

        cells_out, block_ids, points_out = _extract_tet_blocks_from_meshio(msh_path)
        assert cells_out.shape == (1, 4)
        assert cells_out.dtype in (np.int32, np.int64)
        assert block_ids.shape == (1,)
        assert block_ids.dtype == np.int32
        assert block_ids[0] == 1
        assert points_out.shape == (4, 3)
        assert points_out.dtype == np.float64
        np.testing.assert_array_equal(cells_out[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(points_out, points)

    def test_extract_tet_blocks_no_tet_raises(self, tmp_path):
        """_extract_tet_blocks_from_meshio raises if no tetrahedral cells."""
        _meshio = pytest.importorskip("meshio")

        from stellcoilbench.structural_analysis._common import (
            _extract_tet_blocks_from_meshio,
        )

        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        cells = [("triangle", np.array([[0, 1, 2]]))]
        m = _meshio.Mesh(points, cells)
        msh_path = tmp_path / "tri_only.msh"
        _meshio.gmsh.write(str(msh_path), m)

        with pytest.raises(ValueError, match="No tetrahedral cells"):
            _extract_tet_blocks_from_meshio(msh_path)

    def test_per_coil_bc_prevents_displacement_blowup(self, tmp_path):
        """Structural analysis with fixed-support BC yields bounded displacement."""
        _meshio = pytest.importorskip("meshio")
        from stellcoilbench.structural_analysis import (
            _DOLFINX_AVAILABLE,
            _SKFEM_AVAILABLE,
            run_structural_analysis,
        )
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.geo import create_equally_spaced_curves

        if not (_DOLFINX_AVAILABLE or _SKFEM_AVAILABLE):
            pytest.skip("No FEM backend available")

        points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]],
            dtype=float,
        )
        cells = [("tetra", np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))]
        m = _meshio.Mesh(points, cells)
        msh_path = tmp_path / "coil_mesh.msh"
        _meshio.gmsh.write(str(msh_path), m)

        coils_raw = create_equally_spaced_curves(
            1, 1, stellsym=False, R0=1.0, R1=0.1, order=4
        )
        coils = coils_via_symmetries(coils_raw, [Current(1e5)], 1, False)
        bs = BiotSavart(coils)

        summary = run_structural_analysis(
            coils=coils,
            bs=bs,
            output_dir=tmp_path,
            msh_path=msh_path,
            width=0.05,
            height=0.05,
        )
        max_disp = summary["max_displacement_m"]
        assert max_disp < 1.0, (
            f"BC should prevent blow-up; got max_displacement_m={max_disp:.2e}"
        )

    def test_per_coil_bc_pins_every_coil(self):
        """Each mesh block must have at least one pinned node (DOLFINx)."""
        dolfinx = pytest.importorskip("dolfinx")
        basix = pytest.importorskip("basix")
        ufl = pytest.importorskip("ufl")
        from mpi4py import MPI

        from stellcoilbench.structural_analysis._dolfinx import (
            _build_per_coil_support_bcs,
        )

        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 5],
                [1, 0, 5],
                [0, 1, 5],
                [0, 0, 6],
            ],
            dtype=np.float64,
        )
        cells = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)

        ufl_tet = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
        mesh = dolfinx.mesh.create_mesh(
            MPI.COMM_WORLD,
            cells,
            ufl_tet,
            points,
            partitioner=dolfinx.mesh.create_cell_partitioner(
                dolfinx.mesh.GhostMode.none
            ),
        )
        tdim = mesh.topology.dim
        n_cells = mesh.topology.index_map(tdim).size_local
        cell_tags = dolfinx.mesh.meshtags(
            mesh,
            tdim,
            np.arange(n_cells, dtype=np.int32),
            np.array([1, 2], dtype=np.int32)[:n_cells],
        )
        mesh._structural_cell_tags = cell_tags  # type: ignore[attr-defined]

        el = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
        V = dolfinx.fem.functionspace(mesh, el)
        fdim = tdim - 1

        bcs = _build_per_coil_support_bcs(mesh, V, tdim, fdim)
        assert len(bcs) == 1
        bc = bcs[0]
        di = bc.dof_indices()
        pinned_dofs = np.concatenate([np.atleast_1d(x) for x in di]).ravel()
        pinned_nodes = np.unique(pinned_dofs // 3)

        mesh.topology.create_connectivity(tdim, 0)
        c2v = mesh.topology.connectivity(tdim, 0)
        tag_indices = np.asarray(cell_tags.indices)
        tag_values = np.asarray(cell_tags.values)
        cell_to_tag = {
            int(tag_indices[i]): int(tag_values[i]) for i in range(len(tag_indices))
        }
        node_to_blocks: dict[int, set[int]] = {}
        for c in range(n_cells):
            tag = cell_to_tag.get(c, 1)
            for vi in c2v.links(c):
                node_to_blocks.setdefault(int(vi), set()).add(tag)

        blocks_with_pins = set()
        for n in pinned_nodes:
            for b in node_to_blocks.get(n, set()):
                blocks_with_pins.add(b)

        assert 1 in blocks_with_pins
        assert 2 in blocks_with_pins

    def test_per_coil_bc_skfem_pins_every_coil(self):
        """skfem backend: each mesh block must have at least one pinned node."""
        skfem = pytest.importorskip("skfem")
        from stellcoilbench.structural_analysis import _solve_elasticity_skfem

        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 5],
                [1, 0, 5],
                [0, 1, 5],
                [0, 0, 6],
            ],
            dtype=float,
        ).T
        cells = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.intp).T
        mesh = skfem.MeshTet1(points, cells)
        mesh._structural_cell_tags = np.array([1, 2], dtype=np.int32)  # type: ignore[attr-defined]

        body_force = np.zeros((8, 3))
        body_force[:, 2] = -1e4
        result = _solve_elasticity_skfem(
            mesh, body_force, E=100e9, nu=0.3, return_assembly=True
        )
        u, K, ib, fixed_dofs = result[:4]
        pinned_nodes = np.unique(np.asarray(fixed_dofs) // 3)

        cells_mat = mesh.t.T
        cell_tags = mesh._structural_cell_tags
        node_to_blocks: dict[int, set[int]] = {}
        for c in range(cells_mat.shape[0]):
            tag = int(cell_tags[c])
            for n in cells_mat[c]:
                node_to_blocks.setdefault(int(n), set()).add(tag)

        blocks_with_pins = set()
        for n in pinned_nodes:
            for b in node_to_blocks.get(n, set()):
                blocks_with_pins.add(b)

        assert 1 in blocks_with_pins
