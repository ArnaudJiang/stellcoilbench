"""Unit tests for structural analysis pipeline helpers."""

from __future__ import annotations

import numpy as np

from stellcoilbench.structural_analysis._pipeline import (
    _symmetrize_structural_mesh_to_full_coil_set,
    write_structural_vtk_arrays,
)


class TestSymmetrizeStructuralMeshToFullCoilSet:
    """Tests for _symmetrize_structural_mesh_to_full_coil_set."""

    def test_symmetry_factor_one_returns_unchanged(
        self,
    ) -> None:
        """When nfp=1 and stellsym=False, returns input unchanged."""
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cells = np.array([[0, 1, 0, 1]], dtype=np.int64)
        displacement = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
        von_mises = np.array([1e6])

        out_pts, out_cells, out_disp, out_vm = _symmetrize_structural_mesh_to_full_coil_set(
            points, cells, displacement, von_mises, nfp=1, stellsym=False
        )
        np.testing.assert_array_equal(out_pts, points)
        np.testing.assert_array_equal(out_cells, cells)
        np.testing.assert_array_equal(out_disp, displacement)
        np.testing.assert_array_equal(out_vm, von_mises)

    def test_toroidal_nfp2_produces_two_copies(self) -> None:
        """With nfp=2, stellsym=False, produces 2 toroidal copies."""
        # Single point at (1, 0, 0) - rotate by pi gives (-1, 0, 0)
        points = np.array([[1.0, 0.0, 0.0]])
        cells = np.array([[0, 0, 0, 0]], dtype=np.int64)
        displacement = np.array([[0.01, 0.0, 0.0]])
        von_mises = np.array([1e6])

        out_pts, out_cells, out_disp, out_vm = _symmetrize_structural_mesh_to_full_coil_set(
            points, cells, displacement, von_mises, nfp=2, stellsym=False
        )
        assert out_pts.shape[0] == 2
        assert out_cells.shape[0] == 2
        assert out_disp.shape[0] == 2
        assert out_vm.shape[0] == 2

        np.testing.assert_allclose(out_pts[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(out_pts[1], [-1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(out_disp[0], [0.01, 0.0, 0.0])
        np.testing.assert_allclose(out_disp[1], [-0.01, 0.0, 0.0], atol=1e-12)
        np.testing.assert_array_equal(out_vm, [1e6, 1e6])

    def test_stellarator_symmetry_flips_yz(self) -> None:
        """Stellarator symmetry: (x,y,z) -> (x,-y,-z)."""
        points = np.array([[1.0, 1.0, 1.0]])
        cells = np.array([[0, 0, 0, 0]], dtype=np.int64)
        displacement = np.array([[0.1, 0.2, 0.3]])
        von_mises = np.array([1e6])

        out_pts, out_cells, out_disp, out_vm = _symmetrize_structural_mesh_to_full_coil_set(
            points, cells, displacement, von_mises, nfp=1, stellsym=True
        )
        # symmetry_factor = 1*2 = 2, so we get identity + stellarator copy
        assert out_pts.shape[0] == 2
        np.testing.assert_allclose(out_pts[0], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(out_pts[1], [1.0, -1.0, -1.0])
        np.testing.assert_allclose(out_disp[0], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(out_disp[1], [0.1, -0.2, -0.3])


class TestWriteStructuralVtkArrays:
    """Tests for write_structural_vtk_arrays."""

    def test_writes_valid_vtk_file(self, tmp_path) -> None:
        """write_structural_vtk_arrays produces a readable VTK file."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]])
        cells = np.array([[0, 1, 2, 3]], dtype=np.int64)
        displacement = np.zeros((4, 3))
        displacement[1, 0] = 0.01
        von_mises = np.array([1e6])

        out_path = tmp_path / "test_structural.vtk"
        result = write_structural_vtk_arrays(
            points, cells, displacement, von_mises, out_path
        )
        assert result == out_path
        assert out_path.exists()
        # VTK may be written in binary; read back with meshio to verify
        import meshio

        mesh = meshio.read(str(out_path))
        assert "Displacement" in mesh.point_data
        assert "VonMisesStress" in mesh.cell_data
        np.testing.assert_allclose(
            mesh.point_data["Displacement"], displacement, atol=1e-12
        )
        np.testing.assert_allclose(
            mesh.cell_data["VonMisesStress"][0], von_mises, atol=1e-12
        )
