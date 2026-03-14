"""Unit tests for finite-build coil geometry (VTK output used by post_processing)."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from simsopt.geo import create_equally_spaced_curves
from simsopt.field import Current, coils_via_symmetries

from stellcoilbench.finite_build import (
    _compute_cross_section_frame,
    _finite_build_coils_to_msh_parastell,
    _finite_build_coils_to_vtk_parastell,
    _last_parastell_error,
    _write_vtk_unstructured,
    finite_build_coils_to_msh,
    finite_build_coils_to_vtk,
)


class TestComputeCrossSectionFrame:
    """Tests for _compute_cross_section_frame (used by sweep → VTK)."""

    def test_tangent_z_reference_orthogonal(self):
        """Frame vectors should be orthonormal."""
        tangent = np.array([1.0, 0.0, 0.0])
        normal, binormal = _compute_cross_section_frame(tangent)
        assert np.isclose(np.dot(tangent, normal), 0.0)
        assert np.isclose(np.dot(tangent, binormal), 0.0)
        assert np.isclose(np.dot(normal, binormal), 0.0)
        assert np.isclose(np.linalg.norm(normal), 1.0)
        assert np.isclose(np.linalg.norm(binormal), 1.0)

    def test_zero_tangent_raises(self):
        """Zero-length tangent should raise ValueError."""
        with pytest.raises(ValueError, match="zero length"):
            _compute_cross_section_frame(np.array([0.0, 0.0, 0.0]))


class TestWriteVtkUnstructured:
    """Tests for _write_vtk_unstructured."""

    def test_adds_vtk_suffix_if_missing(self, tmp_path):
        """Output path without .vtk gets suffix added."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        faces = np.array([[0, 1, 2]])
        out_path = tmp_path / "output"
        _write_vtk_unstructured(vertices, faces, out_path)
        assert (tmp_path / "output.vtk").exists()
        assert "POINTS 3 float" in (tmp_path / "output.vtk").read_text()


class TestFiniteBuildCoilsToVtk:
    """Tests for finite_build_coils_to_vtk (core post_processing VTK output)."""

    @pytest.fixture
    def simple_coils(self):
        """Create a minimal coil set for testing."""
        base_curves = create_equally_spaced_curves(
            2, 1, stellsym=False, R0=1.7, R1=0.3, order=4, numquadpoints=64
        )
        base_currents = [Current(1e6), Current(-1e6)]
        return coils_via_symmetries(base_curves, base_currents, 1, False)

    def test_writes_vtk_file(self, simple_coils, tmp_path):
        """finite_build_coils_to_vtk should write a valid VTK file."""
        out_path = finite_build_coils_to_vtk(
            simple_coils,
            tmp_path / "finite_build_coils",
            width=0.02,
            height=0.02,
        )
        assert out_path.exists()
        assert out_path.suffix == ".vtk"
        content = out_path.read_text()
        assert "vtk DataFile" in content
        assert "POINTS" in content
        assert "CELLS" in content

    def test_empty_coils_raises(self, tmp_path):
        """Empty coil list should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            finite_build_coils_to_vtk([], tmp_path / "out")


class TestFiniteBuildCoilsToMsh:
    """Tests for finite_build_coils_to_msh (ParaStell-first, stub fallback)."""

    @pytest.fixture
    def simple_coils(self):
        """Create a minimal coil set for testing."""
        base_curves = create_equally_spaced_curves(
            2, 1, stellsym=False, R0=1.7, R1=0.3, order=4, numquadpoints=64
        )
        base_currents = [Current(1e6), Current(-1e6)]
        return coils_via_symmetries(base_curves, base_currents, 1, False)

    def test_finite_build_coils_to_msh_returns_result_or_none(
        self, simple_coils, tmp_path
    ):
        """finite_build_coils_to_msh tries ParaStell, returns (path, indices) or None."""
        result = finite_build_coils_to_msh(
            simple_coils,
            tmp_path / "coils.msh",
            width=0.02,
            height=0.02,
            mesh_size=0.03,
        )
        if result is not None:
            msh_path, coil_indices = result
            assert msh_path.exists()
            assert msh_path.suffix == ".msh"
            assert coil_indices == list(range(len(simple_coils)))


class TestFiniteBuildCoilsToMshSweep:
    """Tests for _finite_build_coils_to_msh_sweep fallback."""

    def test_finite_build_coils_to_msh_sweep_no_meshio_returns_none(
        self, tmp_path: Path
    ) -> None:
        """When meshio import fails, sweep returns None."""
        from simsopt.geo import create_equally_spaced_curves
        from simsopt.field import Current, coils_via_symmetries

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


class TestParaStellExports:
    """Tests for ParaStell-related exports."""

    def test_parastell_symbols_importable(self):
        """_finite_build_coils_to_vtk_parastell, _finite_build_coils_to_msh_parastell, _last_parastell_error are importable."""
        assert callable(_finite_build_coils_to_vtk_parastell)
        assert callable(_finite_build_coils_to_msh_parastell)
        assert _last_parastell_error is None or isinstance(_last_parastell_error, str)
