"""Core finite-build coil geometry: cross-section frame, sweep, and VTK export.

Moved from __init__.py per Second Code Reduction Survey item 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from stellcoilbench.constants import MIN_POINTS_ALONG_CURVE
from stellcoilbench.mpi_utils import proc0_print

from ._vtk import _write_vtk_unstructured

# Default finite-build cross-section at reactor scale (35 cm).
# Re-exported from __init__.py; defined here to avoid circular import.
DEFAULT_CROSS_SECTION_M = 0.35


def _compute_cross_section_frame(
    tangent: np.ndarray,
    reference: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normal and binormal vectors for the cross-section plane.

    Uses a reference vector (default Z-axis) to define a rotation-minimizing
    frame. This avoids Frenet-Serret frame twisting in high-torsion regions.

    Parameters
    ----------
    tangent : np.ndarray
        Unit tangent vector (3,).
    reference : np.ndarray
        Reference vector for frame construction (default: Z-axis).

    Returns
    -------
    normal : np.ndarray
        Unit vector in cross-section plane, perpendicular to tangent.
    binormal : np.ndarray
        Unit vector completing right-handed frame: binormal = tangent × normal.
    """
    tangent = np.asarray(tangent, dtype=float)
    reference = np.asarray(reference, dtype=float)
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-14:
        raise ValueError("Tangent vector has zero length")
    tangent = tangent / tangent_norm

    # normal = cross(tangent, reference) gives vector in cross-section plane
    cross_tr = np.cross(tangent, reference)
    cross_norm = np.linalg.norm(cross_tr)

    if cross_norm < 1e-10:
        # Tangent parallel to reference; use X-axis instead
        reference = np.array([1.0, 0.0, 0.0])
        cross_tr = np.cross(tangent, reference)
        cross_norm = np.linalg.norm(cross_tr)
        if cross_norm < 1e-10:
            reference = np.array([0.0, 1.0, 0.0])
            cross_tr = np.cross(tangent, reference)
            cross_norm = np.linalg.norm(cross_tr)

    normal = cross_tr / cross_norm
    binormal = np.cross(tangent, normal)
    binormal_norm = np.linalg.norm(binormal)
    if binormal_norm > 1e-14:
        binormal = binormal / binormal_norm

    return normal, binormal


def _unit_vector(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm > 1e-14:
        return vector / norm
    if fallback is None:
        raise ValueError("Cannot normalize near-zero vector")
    return _unit_vector(np.asarray(fallback, dtype=float))


def _rotate_vector(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit_vector(axis)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def _initial_parallel_normal(
    tangent: np.ndarray,
    reference: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    tangent = _unit_vector(tangent)
    reference = np.asarray(reference, dtype=float)
    if abs(float(np.dot(tangent, _unit_vector(reference)))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    normal = reference - np.dot(reference, tangent) * tangent
    if np.linalg.norm(normal) < 1e-12:
        reference = np.array([0.0, 1.0, 0.0])
        normal = reference - np.dot(reference, tangent) * tangent
    return _unit_vector(normal)


def _parallel_transport_frames(
    tangents: np.ndarray,
    *,
    closed: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return smooth Bishop-frame normals/binormals along a sampled curve."""
    tangents = np.asarray(tangents, dtype=float)
    unit_tangents = np.array([_unit_vector(t) for t in tangents])
    normal = _initial_parallel_normal(unit_tangents[0])
    normals = [normal]
    binormals = [_unit_vector(np.cross(unit_tangents[0], normal))]

    for idx in range(1, len(unit_tangents)):
        t_prev = unit_tangents[idx - 1]
        t_curr = unit_tangents[idx]
        axis = np.cross(t_prev, t_curr)
        axis_norm = np.linalg.norm(axis)
        transported = normals[-1]
        if axis_norm > 1e-12:
            angle = float(np.arctan2(axis_norm, np.dot(t_prev, t_curr)))
            transported = _rotate_vector(transported, axis / axis_norm, angle)
        transported = transported - np.dot(transported, t_curr) * t_curr
        normal = _unit_vector(transported, fallback=normals[-1])
        normals.append(normal)
        binormals.append(_unit_vector(np.cross(t_curr, normal)))

    normals_arr = np.asarray(normals)
    binormals_arr = np.asarray(binormals)
    if closed and len(unit_tangents) > 2:
        t0 = unit_tangents[0]
        n0 = normals_arr[0]
        n_end = normals_arr[-1] - np.dot(normals_arr[-1], t0) * t0
        if np.linalg.norm(n_end) > 1e-12:
            n_end = _unit_vector(n_end)
            b0 = _unit_vector(np.cross(t0, n0))
            holonomy = float(np.arctan2(np.dot(n_end, b0), np.dot(n_end, n0)))
            fractions = np.linspace(0.0, 1.0, len(unit_tangents))
            for idx, frac in enumerate(fractions):
                correction = -holonomy * frac
                normals_arr[idx] = _rotate_vector(
                    normals_arr[idx], unit_tangents[idx], correction
                )
                normals_arr[idx] = _unit_vector(
                    normals_arr[idx]
                    - np.dot(normals_arr[idx], unit_tangents[idx]) * unit_tangents[idx]
                )
                binormals_arr[idx] = _unit_vector(
                    np.cross(unit_tangents[idx], normals_arr[idx])
                )
    return normals_arr, binormals_arr


def sweep_rectangular_cross_section(
    gamma: np.ndarray,
    gammadash: np.ndarray,
    width: float,
    height: float,
    n_cross: int = 4,
    frame: str = "parallel",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sweep a rectangular cross-section along a curve to create a surface mesh.

    Parameters
    ----------
    gamma : np.ndarray
        Curve points, shape (n_points, 3).
    gammadash : np.ndarray
        Curve derivatives (tangents), shape (n_points, 3).
    width : float
        Cross-section width [m] along the first in-plane direction.
    height : float
        Cross-section height [m] along the second in-plane direction.
    n_cross : int
        Number of vertices per cross-section (4 for rectangle corners).
    frame : {"parallel", "z-reference"}
        Cross-section frame. ``parallel`` uses a Bishop frame with closed-loop
        holonomy correction; ``z-reference`` preserves the legacy global-Z frame.

    Returns
    -------
    vertices : np.ndarray
        Mesh vertices, shape (n_vertices, 3).
    faces : np.ndarray
        Triangle faces as vertex indices, shape (n_faces, 3).

    Notes
    -----
    The first point is appended to gamma/gammadash to close the curve, so the
    last segment connects back to the start for a fully connected loop.
    """
    gamma = np.asarray(gamma)
    gammadash = np.asarray(gammadash)
    n_along = len(gamma)
    if n_along != len(gammadash):
        raise ValueError("gamma and gammadash must have same length")

    # Append first point to close the curve so the sweep is fully connected
    if n_along > 1:
        gamma = np.vstack([gamma, gamma[0:1]])
        gammadash = np.vstack([gammadash, gammadash[0:1]])
        n_along = len(gamma)

    # Rectangle corners in local frame: exact (±w/2, ±h/2) for a true rectangular cross-section
    w2 = float(width) / 2
    h2 = float(height) / 2
    local_corners = np.array(
        [
            [-w2, -h2],
            [w2, -h2],
            [w2, h2],
            [-w2, h2],
        ],
        dtype=float,
    )

    tangents = []
    for i in range(n_along):
        tangent = np.asarray(gammadash[i], dtype=float)
        tangent_len = np.linalg.norm(tangent)
        if tangent_len < 1e-14:
            tangent = (
                gammadash[(i + 1) % n_along] if i < n_along - 1 else gammadash[i - 1]
            )
            tangent_len = np.linalg.norm(tangent)
        tangents.append(tangent / tangent_len)

    frame_key = frame.lower().replace("_", "-")
    if frame_key in {"parallel", "bishop"}:
        normals, binormals = _parallel_transport_frames(np.asarray(tangents), closed=True)
    elif frame_key in {"z-reference", "z", "legacy"}:
        frame_pairs = [_compute_cross_section_frame(tangent) for tangent in tangents]
        normals = np.asarray([pair[0] for pair in frame_pairs])
        binormals = np.asarray([pair[1] for pair in frame_pairs])
    else:
        raise ValueError(f"Unknown finite-build frame: {frame}")

    vertices_list = []
    for i in range(n_along):
        normal = normals[i]
        binormal = binormals[i]
        center = gamma[i]

        for u, v in local_corners:
            point = center + u * normal + v * binormal
            vertices_list.append(point)

    vertices = np.array(vertices_list)

    # Build triangle faces: each segment between consecutive cross-sections
    # gives 4 quads, each split into 2 triangles. With first point appended,
    # the last segment connects back to the start for a fully closed loop.
    faces = []
    for i in range(n_along - 1):
        base_curr = i * 4
        base_next = (i + 1) * 4
        for j in range(4):
            j_next = (j + 1) % 4
            v0 = base_curr + j
            v1 = base_curr + j_next
            v2 = base_next + j_next
            v3 = base_next + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    return vertices, np.array(faces)


def finite_build_coils_to_vtk(
    coils: List,
    output_path: Union[str, Path],
    width: Optional[float] = None,
    height: Optional[float] = None,
    width_per_coil: Optional[List[float]] = None,
    height_per_coil: Optional[List[float]] = None,
    n_along: Optional[int] = None,
    use_stellaris_default: bool = True,
    min_mesh_size: Optional[float] = None,
    max_mesh_size: Optional[float] = None,
    frame: str = "parallel",
) -> Path:
    """
    Generate finite-build coil geometry and export to VTK.

    Sweeps a rectangular cross-section along each coil centerline and writes
    a combined VTK file. Cross-section dimensions can be:
    - Uniform (width, height) for all coils
    - Per-coil (width_per_coil, height_per_coil)
    - Stellaris-derived: sqrt(N_turns) * 20 mm square when N_turns available
    - Default: 35 cm × 35 cm (reactor scale).

    Parameters
    ----------
    coils : List
        List of simsopt Coil objects (with .curve attribute).
    output_path : Path or str
        Output file path (e.g. "finite_build_coils" -> finite_build_coils.vtk).
    width : float, optional
        Uniform cross-section width [m] for all coils.
    height : float, optional
        Uniform cross-section height [m] for all coils.
    width_per_coil : List[float], optional
        Per-coil width [m]. Length must match number of coils.
    height_per_coil : List[float], optional
        Per-coil height [m]. Length must match number of coils.
    n_along : int, optional
        Number of points along each coil for sampling. If None, uses the
        curve's existing quadrature points.
    use_stellaris_default : bool, default=True
        If no dimensions given, use DEFAULT_CROSS_SECTION_M (35 cm).
    min_mesh_size : float, optional
        Gmsh minimum element size [m] for ParaStell tetrahedral mesh.
        Used only when ParaStell succeeds (no per-coil dimensions).
    max_mesh_size : float, optional
        Gmsh maximum element size [m] for ParaStell tetrahedral mesh.
        Used only when ParaStell succeeds (no per-coil dimensions).

    Returns
    -------
    Path
        Path to the written VTK file.

    Raises
    ------
    ValueError
        If coil list is empty or dimension arrays have wrong length.
    """
    if not coils:
        raise ValueError("coils list cannot be empty")

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".vtk":
        output_path = output_path.with_suffix(".vtk")

    # When no per-coil dimensions, try ParaStell first (tetrahedral mesh)
    if width_per_coil is None and height_per_coil is None:
        try:
            from ._parastell import _finite_build_coils_to_vtk_parastell

            fb_w = width if width is not None else DEFAULT_CROSS_SECTION_M
            fb_h = height if height is not None else fb_w
            parastell_path = output_path.parent / (
                output_path.stem + "_parastell" + output_path.suffix
            )
            result = _finite_build_coils_to_vtk_parastell(
                coils,
                parastell_path,
                width=fb_w,
                height=fb_h,
                min_mesh_size=min_mesh_size,
                max_mesh_size=max_mesh_size,
                save_msh=True,
            )
            if result is not None:
                proc0_print(f"[finite-build diagnostic] ParaStell: wrote {result}")
                return result
        except ImportError:
            pass
        from ._parastell import _last_parastell_error

        if _last_parastell_error:
            proc0_print(
                f"[finite-build diagnostic] ParaStell failed: {_last_parastell_error}"
            )
            if (
                "pymoab" in (_last_parastell_error or "").lower()
                or "moab" in (_last_parastell_error or "").lower()
            ):
                proc0_print(
                    "[finite-build diagnostic] For structural analysis, run "
                    "`bash tools/install_parastell_in_vmec.sh` with stellcoilbench_vmec "
                    "active, or use environment-parastell.yml."
                )
        proc0_print(
            "[finite-build diagnostic] ParaStell unavailable, using built-in sweep"
        )

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    proc0_print(f"[finite-build diagnostic] Built-in sweep: {len(coils)} coil(s)")
    for coil_idx, coil in enumerate(coils):
        curve = coil.curve
        gamma = np.asarray(curve.gamma(), dtype=float).reshape(-1, 3)
        proc0_print(
            f"[finite-build diagnostic]   coil {coil_idx}: gamma has"
            f" {len(gamma)} points, gammadash has"
            f" {len(np.asarray(curve.gammadash(), dtype=float).reshape(-1, 3))} points"
        )
        gammadash = np.asarray(curve.gammadash(), dtype=float).reshape(-1, 3)

        # Use at least MIN_POINTS_ALONG_CURVE for accurate rectangular sweep representation
        effective_n_along = (
            n_along if n_along is not None else max(len(gamma), MIN_POINTS_ALONG_CURVE)
        )
        if effective_n_along != len(gamma):
            # Resample curve for accurate finite cross-section representation
            t_orig = np.linspace(0, 1, len(gamma), endpoint=False)
            t_new = np.linspace(0, 1, effective_n_along, endpoint=False)
            gamma = np.column_stack(
                [np.interp(t_new, t_orig, gamma[:, k]) for k in range(3)]
            )
            # Tangent: interpolate direction, then renormalize (linear interp of derivative)
            gammadash = np.column_stack(
                [np.interp(t_new, t_orig, gammadash[:, k]) for k in range(3)]
            )
            # Renormalize tangents to unit length for consistent frame
            norms = np.linalg.norm(gammadash, axis=1, keepdims=True)
            norms = np.where(norms < 1e-14, 1.0, norms)
            gammadash = gammadash / norms

        if width_per_coil is not None and height_per_coil is not None:
            if len(width_per_coil) != len(coils) or len(height_per_coil) != len(coils):
                raise ValueError(
                    "width_per_coil and height_per_coil must have length equal to number of coils"
                )
            w = width_per_coil[coil_idx]
            h = height_per_coil[coil_idx]
        elif width is not None and height is not None:
            w = width
            h = height
        else:
            w = width if width is not None else DEFAULT_CROSS_SECTION_M
            h = height if height is not None else w

        vertices, faces = sweep_rectangular_cross_section(
            gamma, gammadash, width=w, height=h, frame=frame
        )
        all_vertices.append(vertices)
        all_faces.append(faces + vertex_offset)
        vertex_offset += len(vertices)

    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.vstack(all_faces)

    proc0_print(
        f"[finite-build diagnostic] Built-in sweep output: vertices.shape="
        f"{combined_vertices.shape}, faces.shape={combined_faces.shape}"
    )
    _write_vtk_unstructured(
        combined_vertices,
        combined_faces,
        output_path,
        title="finite-build coils",
    )
    return output_path


def finite_build_coils_to_msh(
    coils: List,
    msh_path: Union[str, Path],
    width: float,
    height: float,
    mesh_size: float,
    min_mesh_size: Optional[float] = None,
    max_mesh_size: Optional[float] = None,
) -> Optional[Tuple[Path, List[int]]]:
    """
    Generate tetrahedral .msh for finite-build coils.

    Tries ParaStell first. If ParaStell is unavailable or fails, falls back
    to sweep-based mesh generation (stub: returns None for now).

    Parameters
    ----------
    coils : List
        List of simsopt Coil objects.
    msh_path : Path or str
        Output .msh file path.
    width : float
        Cross-section width [m].
    height : float
        Cross-section height [m].
    mesh_size : float
        Characteristic element size [m].
    min_mesh_size : float, optional
        Gmsh minimum element size [m]. If None, uses mesh_size.
    max_mesh_size : float, optional
        Gmsh maximum element size [m]. If None, uses mesh_size.

    Returns
    -------
    tuple[Path, list[int]] or None
        (msh_path, list of coil indices) if successful; None on failure.
    """
    try:
        from ._parastell import _finite_build_coils_to_msh_parastell

        result = _finite_build_coils_to_msh_parastell(
            coils,
            Path(msh_path),
            width=width,
            height=height,
            mesh_size=mesh_size,
            min_mesh_size=min_mesh_size,
            max_mesh_size=max_mesh_size,
        )
        if result is not None:
            return result
    except ImportError:
        pass
    from ._parastell import _last_parastell_error

    if _last_parastell_error:
        proc0_print(
            f"[finite-build diagnostic] ParaStell MSH failed: {_last_parastell_error}"
        )
        if (
            "pymoab" in (_last_parastell_error or "").lower()
            or "moab" in (_last_parastell_error or "").lower()
        ):
            proc0_print(
                "[finite-build diagnostic] For structural analysis, run "
                "`bash tools/install_parastell_in_vmec.sh` with stellcoilbench_vmec "
                "active, or use environment-parastell.yml."
            )
    # Sweep-to-volume fallback when ParaStell returns None
    from ._sweep_mesh import _finite_build_coils_to_msh_sweep

    return _finite_build_coils_to_msh_sweep(
        coils, Path(msh_path), width, height, mesh_size
    )
