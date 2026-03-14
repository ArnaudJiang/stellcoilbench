"""Gmsh utilities for finite-build coil geometry (context manager, rescaling)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def gmsh_context() -> Iterator[None]:
    """Context manager that initializes Gmsh on entry and finalizes on exit.

    Ensures gmsh.finalize() is always called, even on exception, avoiding
    leftover Gmsh state that can break subsequent mesh generation.

    Sets OMP/MKL/OPENBLAS thread limits before use to avoid Gmsh/OpenMP
    segfaults on macOS (GitHub #1807).
    """
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    import gmsh  # type: ignore[import-untyped]

    gmsh.initialize()
    try:
        yield
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass


def _rescale_msh_points(msh_path: Path, scale: float) -> None:
    """Multiply all node coordinates in a Gmsh ``.msh`` file by *scale* (in-place).

    Supports Gmsh MSH format v4.1.  Nodes are inside a ``$Nodes`` /
    ``$EndNodes`` block.

    Parameters
    ----------
    msh_path : Path
        Path to a Gmsh ``.msh`` file.
    scale : float
        Multiplicative factor applied to every coordinate.
    """
    lines = msh_path.read_text().splitlines()
    out: list[str] = []
    in_nodes = False
    entity_remaining = 0
    skip_entity_header = False
    for line in lines:
        if line.strip() == "$Nodes":
            in_nodes = True
            out.append(line)
            continue
        if line.strip() == "$EndNodes":
            in_nodes = False
            out.append(line)
            continue
        if not in_nodes:
            out.append(line)
            continue
        parts = line.split()
        if entity_remaining == 0 and len(parts) == 4:
            # Entity header: entityDim entityTag parametric numNodesInBlock
            entity_remaining = int(parts[3])
            skip_entity_header = True
            out.append(line)
            continue
        if entity_remaining == 0 and len(parts) == 2:
            # Section header: numEntityBlocks numNodes
            out.append(line)
            continue
        if skip_entity_header and entity_remaining > 0:
            # Node tag lines (just integer IDs)
            try:
                int(parts[0])
                if len(parts) == 1:
                    out.append(line)
                    continue
            except ValueError:
                pass
            # Could be a coordinate line if tags and coords are interleaved
            skip_entity_header = False
        if len(parts) == 3:
            try:
                coords = [float(p) * scale for p in parts]
                out.append(f"{coords[0]:.16e} {coords[1]:.16e} {coords[2]:.16e}")
                entity_remaining -= 1
                continue
            except ValueError:
                pass
        out.append(line)
    msh_path.write_text("\n".join(out) + "\n")
