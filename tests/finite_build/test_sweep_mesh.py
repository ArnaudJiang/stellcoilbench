"""Unit tests for _sweep_mesh._finite_build_coils_to_msh_sweep."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simsopt.geo import create_equally_spaced_curves
from simsopt.field import Current, coils_via_symmetries


def _make_minimal_tetra_msh(path: Path, cell_type: str = "tetra") -> Path:
    """Create a minimal meshio .msh file with tetra or tetra10 cells.

    Parameters
    ----------
    path : Path
        Output .msh file path.
    cell_type : str
        Either "tetra" or "tetra10".

    Returns
    -------
    Path
        The written path.
    """
    import meshio

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    if cell_type == "tetra":
        cells = [("tetra", np.array([[0, 1, 2, 3]], dtype=np.int64))]
    else:
        # tetra10: 4 corners + 6 edge midpoints (indices 4-9)
        extra_pts = np.array(
            [
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ],
            dtype=float,
        )
        points = np.vstack([points, extra_pts])
        cells = [("tetra10", np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64))]
    mesh = meshio.Mesh(points, cells)
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(str(path), mesh, file_format="gmsh22")
    return path


def _make_mesh_without_tetra(path: Path) -> Path:
    """Create a mesh with only triangle cells (no tetra/tetra10)."""
    import meshio

    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
        dtype=float,
    )
    cells = [("triangle", np.array([[0, 1, 2]], dtype=np.int64))]
    mesh = meshio.Mesh(points, cells)
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(str(path), mesh, file_format="gmsh22")
    return path


@pytest.fixture
def simple_coils():
    """Create a minimal coil set for testing."""
    base_curves = create_equally_spaced_curves(
        2, 1, stellsym=False, R0=1.7, R1=0.3, order=4, numquadpoints=64
    )
    base_currents = [Current(1e6), Current(-1e6)]
    return coils_via_symmetries(base_curves, base_currents, 1, False)


class TestFiniteBuildCoilsToMshSweep:
    """Tests for _finite_build_coils_to_msh_sweep fallback."""

    @pytest.mark.parametrize("cell_type", ["tetra", "tetra10"], ids=["tetra", "tetra10"])
    def test_valid_coil_produces_combined_mesh(
        self, simple_coils: list, tmp_path: Path, cell_type: str
    ) -> None:
        """Valid coil with mocked _surface_sweep_to_msh produces combined mesh."""
        pytest.importorskip("meshio")
        import meshio

        from stellcoilbench.finite_build._sweep_mesh import _finite_build_coils_to_msh_sweep

        call_count = 0

        def mock_surface_sweep(gamma, gammadash, *, width, height, mesh_size):
            nonlocal call_count
            del gamma, gammadash, width, height, mesh_size
            p = tmp_path / f"coil_{call_count}.msh"
            call_count += 1
            return _make_minimal_tetra_msh(p, cell_type=cell_type)

        with patch(
            "stellcoilbench.finite_build._sweep_mesh._surface_sweep_to_msh",
            side_effect=mock_surface_sweep,
        ), patch("stellcoilbench.finite_build._sweep_mesh.proc0_print"):
            result = _finite_build_coils_to_msh_sweep(
                simple_coils,
                tmp_path / "coils.msh",
                width=0.02,
                height=0.02,
                mesh_size=0.03,
            )

        assert result is not None
        msh_path, coil_indices = result
        assert msh_path.exists()
        assert msh_path.suffix == ".msh"
        assert coil_indices == [0, 1]
        m = meshio.read(str(msh_path))
        assert m.points.shape[0] > 0
        # tetra10 input is converted to tetra (corners only) in output
        expected = "tetra" if cell_type == "tetra10" else cell_type
        assert any(cb.type == expected for cb in m.cells)

    def test_coil_invalid_gamma_gammadash_skipped(
        self, simple_coils: list, tmp_path: Path
    ) -> None:
        """Coil with invalid gamma/gammadash (len mismatch or <2 points) is skipped."""
        pytest.importorskip("meshio")
        from stellcoilbench.finite_build._sweep_mesh import _finite_build_coils_to_msh_sweep

        # Replace first coil with invalid curve: gamma has 1 point
        invalid_curve = MagicMock()
        invalid_curve.gamma.return_value = np.array([[0.0, 0.0, 0.0]])
        invalid_curve.gammadash.return_value = np.array([[1.0, 0.0, 0.0]])
        invalid_coil = MagicMock()
        invalid_coil.curve = invalid_curve

        coils = [invalid_coil, simple_coils[1]]

        def mock_surface_sweep(gamma, gammadash, *, width, height, mesh_size):
            del gamma, gammadash, width, height, mesh_size
            p = tmp_path / "coil_1.msh"
            return _make_minimal_tetra_msh(p, cell_type="tetra")

        with patch(
            "stellcoilbench.finite_build._sweep_mesh._surface_sweep_to_msh",
            side_effect=mock_surface_sweep,
        ), patch("stellcoilbench.finite_build._sweep_mesh.proc0_print"):
            result = _finite_build_coils_to_msh_sweep(
                coils,
                tmp_path / "coils.msh",
                width=0.02,
                height=0.02,
                mesh_size=0.03,
            )

        assert result is not None
        _, coil_indices = result
        assert coil_indices == [1]  # Only second coil meshed

    def test_meshio_read_failure_coil_skipped(
        self, simple_coils: list, tmp_path: Path
    ) -> None:
        """When meshio.read fails on a coil's .msh file, that coil is skipped."""
        import meshio

        pytest.importorskip("meshio")
        from stellcoilbench.finite_build._sweep_mesh import _finite_build_coils_to_msh_sweep

        call_count = 0

        def mock_surface_sweep(gamma, gammadash, *, width, height, mesh_size):
            nonlocal call_count
            del gamma, gammadash, width, height, mesh_size
            p = tmp_path / f"coil_{call_count}.msh"
            call_count += 1
            _make_minimal_tetra_msh(p, cell_type="tetra")
            return p

        read_count = 0
        orig_read = meshio.read

        def mock_meshio_read(path):
            nonlocal read_count
            read_count += 1
            if read_count == 1:
                raise RuntimeError("Simulated meshio read failure")
            return orig_read(path)

        with patch(
            "stellcoilbench.finite_build._sweep_mesh._surface_sweep_to_msh",
            side_effect=mock_surface_sweep,
        ), patch("meshio.read", side_effect=mock_meshio_read), patch(
            "stellcoilbench.finite_build._sweep_mesh.proc0_print"
        ):
            result = _finite_build_coils_to_msh_sweep(
                simple_coils,
                tmp_path / "coils.msh",
                width=0.02,
                height=0.02,
                mesh_size=0.03,
            )

        assert result is not None
        _, coil_indices = result
        assert coil_indices == [1]  # First coil skipped due to read failure

    def test_mesh_no_tetra_tetra10_triggers_skip(
        self, simple_coils: list, tmp_path: Path
    ) -> None:
        """Mesh with no tetra/tetra10 cells triggers 'no tetra' branch and skips coil."""
        pytest.importorskip("meshio")
        from stellcoilbench.finite_build._sweep_mesh import _finite_build_coils_to_msh_sweep

        def mock_surface_sweep(gamma, gammadash, *, width, height, mesh_size):
            del gamma, gammadash, width, height, mesh_size
            p = tmp_path / "coil_tri.msh"
            return _make_mesh_without_tetra(p)

        with patch(
            "stellcoilbench.finite_build._sweep_mesh._surface_sweep_to_msh",
            side_effect=mock_surface_sweep,
        ), patch("stellcoilbench.finite_build._sweep_mesh.proc0_print"):
            result = _finite_build_coils_to_msh_sweep(
                simple_coils,
                tmp_path / "coils.msh",
                width=0.02,
                height=0.02,
                mesh_size=0.03,
            )

        # All coils return triangle-only mesh -> no tetra -> all_points empty -> None
        assert result is None

    def test_empty_all_points_returns_none(self, tmp_path: Path) -> None:
        """When all coils fail (sweep returns None or all skipped), result is None."""
        pytest.importorskip("meshio")
        from stellcoilbench.finite_build._sweep_mesh import _finite_build_coils_to_msh_sweep

        base_curves = create_equally_spaced_curves(
            2, 1, stellsym=False, R0=1.7, R1=0.3, order=4, numquadpoints=64
        )
        base_currents = [Current(1e6), Current(-1e6)]
        coils = coils_via_symmetries(base_curves, base_currents, 1, False)

        with patch(
            "stellcoilbench.finite_build._sweep_mesh._surface_sweep_to_msh",
            return_value=None,
        ), patch("stellcoilbench.finite_build._sweep_mesh.proc0_print"):
            result = _finite_build_coils_to_msh_sweep(
                coils,
                tmp_path / "coils.msh",
                width=0.02,
                height=0.02,
                mesh_size=0.03,
            )

        assert result is None

    def test_combined_mesh_zero_points_returns_none(
        self, simple_coils: list, tmp_path: Path
    ) -> None:
        """When combined mesh has 0 points (edge case), returns None."""
        import meshio

        pytest.importorskip("meshio")
        from stellcoilbench.finite_build._sweep_mesh import _finite_build_coils_to_msh_sweep

        def mock_surface_sweep(gamma, gammadash, *, width, height, mesh_size):
            del gamma, gammadash, width, height, mesh_size
            p = tmp_path / "coil.msh"
            _make_minimal_tetra_msh(p, cell_type="tetra")
            return p

        # Return mesh with tetra block but 0 points so vstack yields 0 total rows
        empty_mesh = meshio.Mesh(
            np.empty((0, 3)),
            [("tetra", np.empty((0, 4), dtype=np.int64))],
        )

        def mock_read(path):
            del path
            return empty_mesh

        with patch(
            "stellcoilbench.finite_build._sweep_mesh._surface_sweep_to_msh",
            side_effect=mock_surface_sweep,
        ), patch("meshio.read", side_effect=mock_read), patch(
            "stellcoilbench.finite_build._sweep_mesh.proc0_print"
        ):
            result = _finite_build_coils_to_msh_sweep(
                simple_coils,
                tmp_path / "coils.msh",
                width=0.02,
                height=0.02,
                mesh_size=0.03,
            )

        assert result is None

    def test_no_meshio_returns_none(self, tmp_path: Path) -> None:
        """When meshio import fails, sweep returns None."""
        from stellcoilbench.finite_build._sweep_mesh import (
            _finite_build_coils_to_msh_sweep,
        )

        base_curves = create_equally_spaced_curves(
            2, 1, stellsym=False, R0=1.7, R1=0.3, order=4, numquadpoints=64
        )
        base_currents = [Current(1e6), Current(-1e6)]
        coils = coils_via_symmetries(base_curves, base_currents, 1, False)

        orig_import = __import__

        def mock_import(name: str, *args: object, **kwargs: object):
            if name == "meshio":
                raise ImportError("No module named 'meshio'")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = _finite_build_coils_to_msh_sweep(
                coils,
                tmp_path / "coils.msh",
                width=0.02,
                height=0.02,
                mesh_size=0.03,
            )

        assert result is None
