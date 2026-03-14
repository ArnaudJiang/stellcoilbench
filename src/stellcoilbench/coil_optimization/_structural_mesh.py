"""Gmsh-based coil mesh generation for FEM structural analysis.

Provides functions to create tetrahedral meshes from coil centerlines
via surface sweep, or torus approximations for tests.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from stellcoilbench.finite_build._core import (
    _compute_cross_section_frame,
    sweep_rectangular_cross_section,
)
from stellcoilbench.finite_build._gmsh import gmsh_context
from stellcoilbench.mpi_utils import proc0_print

_N_SECTIONS_THRUSECTIONS: int = 8
"""Number of cross-sections for addThruSections (fewer reduces PLC/self-intersection risk)."""


def _try_add_thru_sections_zframe(
    gamma: np.ndarray,
    gammadash: np.ndarray,
    width: float,
    height: float,
    mesh_size: float,
) -> Path | None:
    """Create a tetrahedral coil mesh via Gmsh addThruSections with Z-axis frame.

    Uses the same cross-section frame as built-in sweep (_compute_cross_section_frame)
    to build rectangular wires at sampled sections, then addThruSections to create
    the solid via OpenCASCADE—avoiding surface parametrization issues.

    Parameters
    ----------
    gamma : np.ndarray
        Curve points, shape (n_points, 3) [m].
    gammadash : np.ndarray
        Curve derivatives (tangents), shape (n_points, 3).
    width, height : float
        Cross-section dimensions [m].
    mesh_size : float
        Characteristic element size [m].

    Returns
    -------
    Path or None
        Path to temporary .msh file if successful; None on failure.
    """
    try:
        import gmsh  # type: ignore[import-untyped]
    except ImportError:
        return None

    gamma = np.asarray(gamma)
    gammadash = np.asarray(gammadash)
    n_along = len(gamma)
    if n_along != len(gammadash) or n_along < 2:
        return None

    # Close the curve: append first point (matching sweep_rectangular_cross_section)
    if n_along > 1:
        gamma = np.vstack([gamma, gamma[0:1]])
        gammadash = np.vstack([gammadash, gammadash[0:1]])
        n_along = len(gamma)

    n_sections = min(_N_SECTIONS_THRUSECTIONS, n_along - 1)
    if n_sections < 2:
        return None

    # Sample section indices (include first and last for closed curve)
    indices = np.round(
        np.linspace(0, n_along - 1, n_sections + 1, endpoint=True)
    ).astype(int)
    indices = np.clip(indices, 0, n_along - 1)

    w2 = float(width) / 2
    h2 = float(height) / 2
    wire_tags: list[int] = []

    try:
        with gmsh_context():
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("thrusections_coil")

            for idx in indices:
                center = np.asarray(gamma[idx], dtype=float)
                tangent = np.asarray(gammadash[idx], dtype=float)
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm < 1e-14:
                    tangent = (
                        gammadash[idx + 1] if idx < n_along - 1 else gammadash[idx - 1]
                    )
                    tangent_norm = np.linalg.norm(tangent)
                if tangent_norm < 1e-14:
                    return None
                tangent = tangent / tangent_norm

                normal, binormal = _compute_cross_section_frame(tangent)
                # Rectangle corners: ±w2 along normal, ±h2 along binormal
                corners = np.array(
                    [
                        center - w2 * normal - h2 * binormal,
                        center + w2 * normal - h2 * binormal,
                        center + w2 * normal + h2 * binormal,
                        center - w2 * normal + h2 * binormal,
                    ]
                )
                pt_tags = [
                    gmsh.model.occ.addPoint(float(c[0]), float(c[1]), float(c[2]))
                    for c in corners
                ]
                line_tags = []
                for j in range(4):
                    line_tags.append(
                        gmsh.model.occ.addLine(pt_tags[j], pt_tags[(j + 1) % 4])
                    )
                wire_tag = gmsh.model.occ.addWire(line_tags)
                wire_tags.append(wire_tag)

            gmsh.model.occ.addThruSections(wire_tags, makeSolid=True)
            gmsh.model.occ.synchronize()

            vols = gmsh.model.getEntities(3)
            if not vols:
                return None

            gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], tag=1)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            gmsh.model.mesh.generate(3)

            _, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
            if sum(len(t) for t in elem_tags) == 0:
                return None

            fd = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
            msh_path = Path(fd.name)
            fd.close()
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(str(msh_path))
            return msh_path
    except Exception as e:
        proc0_print(
            f"[finite-build diagnostic] addThruSections (Z-frame) failed: "
            f"{type(e).__name__}: {e}"
        )
        return None


def _surface_sweep_to_msh_stl(
    gamma: np.ndarray,
    gammadash: np.ndarray,
    width: float,
    height: float,
    mesh_size: float,
) -> Path | None:
    """Create mesh via STL→classifySurfaces→volume (fallback when addThruSections fails)."""
    try:
        import gmsh  # type: ignore[import-untyped]
        import meshio  # noqa: F401
    except ImportError:
        return None

    vertices, faces = sweep_rectangular_cross_section(gamma, gammadash, width, height)
    if vertices.size == 0 or faces.size == 0:
        return None

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as fd:
        stl_path = Path(fd.name)
    try:
        mesh = meshio.Mesh(points=vertices, cells=[("triangle", faces)])
        meshio.write(str(stl_path), mesh, file_format="stl")
    except Exception as e:
        proc0_print(
            f"[finite-build diagnostic] sweep STL write failed: {type(e).__name__}: {e}"
        )
        stl_path.unlink(missing_ok=True)
        return None

    try:
        with gmsh_context():
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.clear()
            gmsh.merge(str(stl_path))
            gmsh.model.mesh.classifySurfaces(
                170 * math.pi / 180,
                boundary=True,
                forReparametrization=False,
                curveAngle=180 * math.pi / 180,
            )
            gmsh.model.mesh.createGeometry()
            surfaces = gmsh.model.getEntities(2)
            if not surfaces:
                return None
            surf_tags = [e[1] for e in surfaces]
            sl_tag = gmsh.model.geo.addSurfaceLoop(surf_tags)
            gmsh.model.geo.addVolume([sl_tag])
            gmsh.model.geo.synchronize()
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            gmsh.model.mesh.generate(3)
            _, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
            if sum(len(t) for t in elem_tags) == 0:
                return None
            fd = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
            msh_path = Path(fd.name)
            fd.close()
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(str(msh_path))
            return msh_path
    except Exception as e:
        proc0_print(
            f"[finite-build diagnostic] sweep STL→volume failed: "
            f"{type(e).__name__}: {e}"
        )
        return None
    finally:
        stl_path.unlink(missing_ok=True)


def _surface_sweep_to_msh(
    gamma: np.ndarray,
    gammadash: np.ndarray,
    width: float,
    height: float,
    mesh_size: float,
) -> Path | None:
    """Create a tetrahedral mesh from a swept rectangular cross-section surface.

    Tries addThruSections (Z-frame) first; falls back to STL→volume if it fails.

    Parameters
    ----------
    gamma : np.ndarray
        Curve points, shape (n_points, 3) [m].
    gammadash : np.ndarray
        Curve derivatives (tangents), shape (n_points, 3).
    width, height : float
        Cross-section dimensions [m].
    mesh_size : float
        Characteristic element size [m].

    Returns
    -------
    Path or None
        Path to temporary .msh file if successful; None on failure.
    """
    result = _try_add_thru_sections_zframe(
        gamma, gammadash, width=width, height=height, mesh_size=mesh_size
    )
    if result is not None:
        return result
    return _surface_sweep_to_msh_stl(
        gamma, gammadash, width=width, height=height, mesh_size=mesh_size
    )


def _generate_rectangular_torus_mesh_gmsh(
    x: float,
    y: float,
    z: float,
    R_major: float,
    width: float,
    height: float,
    mesh_size: float,
) -> Path:
    """Create a torus mesh with rectangular cross-section via Gmsh addPipe.

    Uses OpenCASCADE pipe: rectangle profile swept along circular wire.
    Matches finite-build (rectangular winding pack) geometry used in post-processing.

    Parameters
    ----------
    x, y, z : float
        Center of the torus (hole center) [m].
    R_major : float
        Major radius (distance from center to tube center) [m].
    width, height : float
        Rectangular cross-section dimensions [m].
    mesh_size : float
        Characteristic element size [m].

    Returns
    -------
    Path
        Path to the temporary .msh file. Caller is responsible for cleanup.
    """
    import gmsh  # type: ignore[import-untyped]

    with gmsh_context():
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("rect_torus")
        # Rectangle profile in x-y plane, centered; will be rotated to x-z (perp to circle tangent)
        rect = gmsh.model.occ.addRectangle(
            -width / 2,
            -height / 2,
            0.0,
            width,
            height,
        )
        # Rotate so profile lies in x-z plane (perpendicular to circle tangent at start)
        gmsh.model.occ.rotate([(2, rect)], 0, 0, 0, 1, 0, 0, np.pi / 2)
        # Translate to circle start (R_major, 0, 0) in local frame, then to (x,y,z)
        gmsh.model.occ.translate([(2, rect)], x + R_major, y, z)
        # Circle curve then wire (addPipe expects a wire)
        circle = gmsh.model.occ.addCircle(x, y, z, R_major)
        wire = gmsh.model.occ.addWire([circle])
        # Sweep rectangle along circle wire
        gmsh.model.occ.addPipe([(2, rect)], wire)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
        gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], tag=1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.mesh.generate(3)

        fd = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
        msh_path = Path(fd.name)
        fd.close()
        try:
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(str(msh_path))
            return msh_path
        except Exception:
            msh_path.unlink(missing_ok=True)
            raise


def _generate_torus_mesh_gmsh(
    x: float,
    y: float,
    z: float,
    R_major: float,
    r_minor: float,
    mesh_size: float,
) -> Path:
    """Create a torus mesh with circular cross-section (legacy).

    Use _generate_rectangular_torus_mesh_gmsh for rectangular cross-section
    matching post-processed finite-build geometry.
    """
    import gmsh  # type: ignore[import-untyped]

    with gmsh_context():
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("torus")
        gmsh.model.occ.addTorus(x, y, z, R_major, r_minor)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
        gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], tag=1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.mesh.generate(3)

        fd = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
        msh_path = Path(fd.name)
        fd.close()
        try:
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(str(msh_path))
            return msh_path
        except Exception:
            msh_path.unlink(missing_ok=True)
            raise
