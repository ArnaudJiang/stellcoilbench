"""VTK output utilities for finite-build coil geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def _write_vtk_unstructured(
    vertices: np.ndarray,
    faces: np.ndarray,
    filepath: Union[str, Path],
    title: str = "finite-build coils",
) -> None:
    """
    Write vertices and triangle faces to VTK unstructured grid file.

    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertices, shape (n_vertices, 3).
    faces : np.ndarray
        Triangle faces as vertex indices, shape (n_faces, 3).
    filepath : Path or str
        Output file path. .vtk suffix added if missing.
    title : str, optional
        Title string written in the VTK header.
    """
    filepath = Path(filepath)
    if filepath.suffix.lower() != ".vtk":
        filepath = filepath.with_suffix(".vtk")

    n_points = len(vertices)
    n_cells = len(faces)
    # Each triangle: 3 vertex indices + 3 for the count
    cell_data_size = n_cells * 4

    with open(filepath, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {n_points} float\n")
        for row in vertices:
            f.write(f"{row[0]:.6e} {row[1]:.6e} {row[2]:.6e}\n")
        f.write(f"CELLS {n_cells} {cell_data_size}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        f.write(f"CELL_TYPES {n_cells}\n")
        for _ in range(n_cells):
            f.write("5\n")  # VTK_TRIANGLE = 5


def _rescale_vtk_points(vtk_path: Path, scale: float) -> None:
    """Multiply all point coordinates in a VTK file by *scale* (in-place).

    Parameters
    ----------
    vtk_path : Path
        Path to a legacy ASCII VTK file.
    scale : float
        Multiplicative factor applied to every coordinate.
    """
    lines = vtk_path.read_text().splitlines(keepends=True)
    out: list[str] = []
    coords_remaining = 0
    for line in lines:
        if coords_remaining <= 0 and line.strip().startswith("POINTS"):
            parts = line.split()
            coords_remaining = int(parts[1]) * 3
            out.append(line)
            continue
        if coords_remaining > 0:
            tokens = line.split()
            scaled_tokens: list[str] = []
            for tok in tokens:
                if coords_remaining > 0:
                    scaled_tokens.append(f"{float(tok) * scale:.16e}")
                    coords_remaining -= 1
                else:
                    scaled_tokens.append(tok)
            out.append(" ".join(scaled_tokens) + "\n")
            continue
        out.append(line)
    vtk_path.write_text("".join(out))
