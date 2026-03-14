"""Shared helpers for the structural-analysis backends.

These are pure-NumPy utilities used by both the DOLFINx and scikit-fem
backend modules, factored here to avoid circular imports.

I/O helpers: output directory preparation, mesh extraction.
BC helpers: fixed-support z-threshold computation.
Physics computations live in :mod:`._physics`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Re-export physics functions for backward compatibility (used by _skfem, _dolfinx,
# _pipeline, coil_optimization/_structural_objective, finite_build/_gmsh, tests).
from ._physics import (  # noqa: F401
    MU0,
    _G_helper,
    _build_coil_centerline_data,
    _compute_B0,
    _compute_B_internal,
    _compute_Breg_for_coil,
    _compute_coil_frame,
    _compute_jcross_b,
    _lame_parameters,
)

# ---------------------------------------------------------------------------
# BC and I/O helpers
# ---------------------------------------------------------------------------

# Fraction of z-range used for fixed-support BC: nodes with z <= z_min + BC_Z_FRACTION * z_range are pinned.
BC_Z_FRACTION: float = 0.15


def _compute_z_threshold_for_fixed_support(
    z_min: float,
    z_max: float,
    bc_z_fraction: float | None = None,
    *,
    range_if_zero: float = 0.0,
) -> float:
    """Compute the z-threshold for fixed-support BC.

    Nodes with z <= threshold are pinned. Formula:
    threshold = z_min + bc_z_fraction * (z_max - z_min).

    When z_max <= z_min (flat geometry), uses range_if_zero as the effective
    z-range multiplier: per-block logic uses range_if_zero=1.0; global fallback
    uses range_if_zero=0.0 (threshold = z_min).

    Parameters
    ----------
    z_min : float
        Minimum z coordinate of the block/domain.
    z_max : float
        Maximum z coordinate of the block/domain.
    bc_z_fraction : float, optional
        Fraction of z-range to pin. Defaults to :const:`BC_Z_FRACTION`.
    range_if_zero : float, default 0.0
        Effective z-range when z_max <= z_min: threshold = z_min + bc_z_fraction * range_if_zero.
        Use 1.0 for per-coil blocks, 0.0 for global fallback.

    Returns
    -------
    float
        z-threshold; nodes with z <= threshold are fixed.
    """
    frac = bc_z_fraction if bc_z_fraction is not None else BC_Z_FRACTION
    z_range = z_max - z_min
    if z_range > 0:
        return z_min + frac * z_range
    return z_min + frac * range_if_zero


def _get_mesh_block_tag_summary(mesh: object) -> str | None:
    """Return a one-line summary of block tags if mesh has cell tags.

    Works with both DOLFINx and scikit-fem meshes. Used for BC validation.
    """
    cell_tags = getattr(mesh, "_structural_cell_tags", None)
    if cell_tags is None:
        return None
    try:
        if hasattr(cell_tags, "values"):
            tags = np.asarray(cell_tags.values)
        elif hasattr(cell_tags, "__len__"):
            tags = np.asarray(cell_tags)
        else:
            return None
    except Exception:
        return None
    unique = np.unique(tags)
    n = len(unique)
    if n == 0:
        return None
    return f"Mesh block tags: {n} unique (ids: {list(unique)})"


def _prepare_structural_output_dir(output_dir: Path | str) -> Path:
    """Ensure the structural-analysis output directory exists.

    Shared by both DOLFINx and scikit-fem export routines.

    Parameters
    ----------
    output_dir : Path or str
        Target directory for output files.

    Returns
    -------
    Path
        The path, with parents created.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _extract_tet_blocks_from_meshio(
    msh_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract tetrahedral cells from a meshio-readable mesh file.

    Supports tetra, tetra10, tetra14, tetra20. For higher-order elements,
    only the corner nodes (first 4) are used to produce linear tets.

    Parameters
    ----------
    msh_path : Path
        Path to the mesh file (e.g. Gmsh .msh).

    Returns
    -------
    cells : np.ndarray
        Cell connectivity, shape (n_cells, 4), dtype int.
    block_ids : np.ndarray
        Block tag per cell (1-based), shape (n_cells,), dtype np.int32.
    points : np.ndarray
        Vertex coordinates, shape (n_vertices, 3), dtype np.float64.

    Raises
    ------
    ValueError
        If no tetrahedral cells are found in the mesh.
    """
    import meshio

    m = meshio.read(str(msh_path))
    tet_blocks: list[np.ndarray] = []
    for cell_block in m.cells:
        if cell_block.type == "tetra":
            tet_blocks.append(cell_block.data)
        elif cell_block.type in ("tetra10", "tetra14", "tetra20"):
            # Use corner nodes only for higher-order tets
            tet_blocks.append(cell_block.data[:, :4])
    if not tet_blocks:
        raise ValueError(
            f"No tetrahedral cells found in {msh_path}; "
            f"cell types: {[b.type for b in m.cells]}"
        )

    cells_tet = np.vstack(tet_blocks)
    block_ids = np.concatenate(
        [np.full(len(b), i + 1, dtype=np.int32) for i, b in enumerate(tet_blocks)]
    )
    # If meshio merged tetra blocks (gmsh round-trip) but preserved gmsh:physical
    # in cell_data, use that for per-coil block IDs so per-coil BC activates.
    phy_key = "gmsh:physical"
    if len(tet_blocks) == 1 and phy_key in m.cell_data:
        phy_arrays = m.cell_data[phy_key]
        tet_idx = next(
            (
                i
                for i, cb in enumerate(m.cells)
                if cb.type in ("tetra", "tetra10", "tetra14", "tetra20")
            ),
            -1,
        )
        if tet_idx >= 0 and tet_idx < len(phy_arrays):
            phy = np.asarray(phy_arrays[tet_idx], dtype=np.int32)
            if phy.size == cells_tet.shape[0] and np.any(phy > 0):
                block_ids = np.where(phy >= 1, phy, phy + 1).astype(np.int32)
    vertices = np.unique(cells_tet)
    old_to_new = np.full(m.points.shape[0], -1, dtype=np.int64)
    old_to_new[vertices] = np.arange(len(vertices), dtype=np.int64)
    points = np.asarray(m.points[vertices], dtype=np.float64)
    cells = np.asarray(old_to_new[cells_tet], dtype=np.int64)

    return cells, block_ids, points
