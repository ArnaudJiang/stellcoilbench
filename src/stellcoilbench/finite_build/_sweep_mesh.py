"""Sweep-to-volume mesh fallback for finite-build coil geometry."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from stellcoilbench.coil_optimization._structural_mesh import _surface_sweep_to_msh
from stellcoilbench.mpi_utils import proc0_print


def _finite_build_coils_to_msh_sweep(
    coils: List,
    msh_path: Path,
    width: float,
    height: float,
    mesh_size: float,
) -> Optional[tuple[Path, list[int]]]:
    """Generate tetrahedral coil mesh via sweep-to-volume fallback.

    For each coil, obtains gamma and gammadash from the curve, calls
    _surface_sweep_to_msh to create a per-coil tetrahedral mesh via
    swept surface + Gmsh volume fill. Combines successful per-coil meshes
    (stack points with offset, assign gmsh:physical per block) and writes
    to the output .msh file.

    Parameters
    ----------
    coils : List
        simsopt Coil objects (unique coils only).
    msh_path : Path
        Output .msh file path.
    width, height : float
        Cross-section dimensions [m].
    mesh_size : float
        Characteristic element size [m].

    Returns
    -------
    tuple[Path, list[int]] or None
        (Path to written .msh, list of successfully-meshed coil indices) if
        successful; None on total failure.
    """
    try:
        import meshio  # noqa: F401
    except ImportError:
        return None

    all_points: List[np.ndarray] = []
    all_cells: List[np.ndarray] = []
    meshed_coil_indices: List[int] = []
    node_offset = 0

    proc0_print(
        f"[finite-build diagnostic] sweep-to-volume fallback: {len(coils)} coil(s)"
    )

    for coil_idx, coil in enumerate(coils):
        curve = coil.curve
        gamma = np.asarray(curve.gamma(), dtype=float).reshape(-1, 3)
        gammadash = np.asarray(curve.gammadash(), dtype=float).reshape(-1, 3)
        if len(gamma) != len(gammadash) or len(gamma) < 2:
            proc0_print(
                f"[finite-build diagnostic]   coil {coil_idx}: invalid gamma/gammadash, skipping"
            )
            continue

        tmp_msh = _surface_sweep_to_msh(
            gamma, gammadash, width=width, height=height, mesh_size=mesh_size
        )
        if tmp_msh is None:
            proc0_print(
                f"[finite-build diagnostic]   coil {coil_idx}: sweep-to-msh failed, skipping"
            )
            continue

        try:
            try:
                m = meshio.read(str(tmp_msh))
            except Exception as e:
                proc0_print(
                    f"[finite-build diagnostic]   coil {coil_idx}: meshio.read failed"
                    f" ({type(e).__name__}: {e}), skipping coil"
                )
                continue
            for cb in m.cells:
                if cb.type == "tetra":
                    pts = m.points
                    cells = cb.data + node_offset
                    all_points.append(pts)
                    all_cells.append(cells)
                    meshed_coil_indices.append(coil_idx)
                    node_offset += len(pts)
                    proc0_print(
                        f"[finite-build diagnostic]   coil {coil_idx}: read mesh"
                        f" pts.shape={pts.shape}, cells.shape={cells.shape}"
                    )
                    break
                if cb.type == "tetra10":
                    pts = m.points
                    cells = cb.data[:, :4] + node_offset
                    all_points.append(pts)
                    all_cells.append(cells)
                    meshed_coil_indices.append(coil_idx)
                    node_offset += len(pts)
                    proc0_print(
                        f"[finite-build diagnostic]   coil {coil_idx}: read mesh (tetra10)"
                        f" pts.shape={pts.shape}, cells.shape={cells.shape}"
                    )
                    break
            else:
                proc0_print(
                    f"[finite-build diagnostic]   coil {coil_idx}: no tetra/tetra10 in mesh,"
                    f" cell types: {[cb.type for cb in m.cells]}"
                )
        finally:
            tmp_msh.unlink(missing_ok=True)

    if not all_points or not all_cells:
        proc0_print(
            f"[finite-build diagnostic] sweep fallback aborted: all_points has "
            f"{len(all_points)} entries, all_cells has {len(all_cells)}"
        )
        return None

    combined_points = np.vstack(all_points)
    if combined_points.shape[0] == 0:
        proc0_print(
            "[finite-build diagnostic] WARNING: combined mesh has 0 points, skipping"
        )
        return None

    cell_blocks = [("tetra", cells) for cells in all_cells]
    cell_data = {
        "gmsh:physical": [
            np.full(len(cells), i + 1, dtype=np.int32)
            for i, cells in enumerate(all_cells)
        ]
    }
    out_mesh = meshio.Mesh(combined_points, cell_blocks, cell_data=cell_data)
    msh_path = Path(msh_path)
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(str(msh_path), out_mesh, file_format="gmsh22")
    proc0_print(
        f"[finite-build diagnostic] sweep meshed coil indices: {meshed_coil_indices} "
        f"(of {len(coils)} requested)"
    )
    return (msh_path, meshed_coil_indices)
