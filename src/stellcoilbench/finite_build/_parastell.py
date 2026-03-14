"""ParaStell-based finite-build coil geometry generation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from stellcoilbench.constants import (
    DEFAULT_MAX_TETRAHEDRAL_MESH_SIZE_M,
    DEFAULT_MIN_TETRAHEDRAL_MESH_SIZE_M,
)
from stellcoilbench.utils import suppress_output

from ._vtk import _rescale_vtk_points
from ._gmsh import _rescale_msh_points

# Set when ParaStell import fails (for diagnostics)
_last_parastell_error: Optional[str] = None


def _write_parastell_filament_file(
    coils: List,
    filepath: Path,
    scale: float = 100.0,
) -> None:
    """
    Write simsopt coils to ParaStell filament format.

    Format: x y z current per line. Each coil ends with a line where current=0.
    ParaStell's reader appends the first point when it sees current=0 to close loops.
    Coords are in meters; scale multiplies them (ParaStell uses scale=100 for m->cm).

    Parameters
    ----------
    coils : List
        List of simsopt Coil objects (with .curve and .current attributes).
    filepath : Path
        Output file path.
    scale : float, optional
        Coordinate scaling factor (default 100 for meters to cm).
    """
    with open(filepath, "w") as f:
        f.write("stellcoilbench coils\n")
        f.write("simsopt export\n")
        f.write("begin filament\n")
        for coil in coils:
            curve = coil.curve
            gamma = np.asarray(curve.gamma(), dtype=float).reshape(-1, 3)
            current = abs(float(coil.current.get_value()))
            for pt in gamma:
                f.write(f"  {pt[0]:.16e} {pt[1]:.16e} {pt[2]:.16e} {current:.16e}\n")
            # ParaStell: s=0 signals end of filament; reader appends first point to close
            if len(gamma) > 0:
                f.write(
                    f"  {gamma[0][0]:.16e} {gamma[0][1]:.16e} {gamma[0][2]:.16e} 0.0\n"
                )


def _finite_build_coils_to_vtk_parastell(
    coils: List,
    output_path: Path,
    width: float,
    height: float,
    min_mesh_size: Optional[float] = None,
    max_mesh_size: Optional[float] = None,
    save_msh: bool = True,
) -> Optional[Path]:
    """
    Use ParaStell to generate finite-build coil geometry and tetrahedral mesh.

    Builds CadQuery solids from filament data, meshes with Gmsh (3D tetrahedra),
    and writes VTK. Cross-section is oriented using coil center-of-mass.

    Parameters
    ----------
    coils : List
        List of simsopt Coil objects.
    output_path : Path
        Output VTK file path (should include _parastell in stem).
    width : float
        Cross-section width [m].
    height : float
        Cross-section height [m].
    min_mesh_size : float, optional
        Gmsh minimum element size [m].
    max_mesh_size : float, optional
        Gmsh maximum element size [m].
    save_msh : bool, optional
        If True, also write a ``.msh`` file alongside the VTK output.
        The ``.msh`` retains Gmsh physical groups needed by FEM solvers
        (e.g. DOLFINx).

    Returns
    -------
    Path or None
        Path to written VTK file if successful; None if ParaStell or Gmsh
        unavailable or if build/mesh fails.
    """
    import os

    global _last_parastell_error
    try:
        from parastell.magnet_coils import MagnetSetFromFilaments  # type: ignore[import-untyped]
    except ImportError as e:  # pragma: no cover
        _last_parastell_error = str(e)
        return None

    try:
        import gmsh  # type: ignore[import-untyped]
    except ImportError as e:  # pragma: no cover
        _last_parastell_error = f"gmsh: {e}"
        return None

    # Avoid Gmsh/OpenMP multithreading segfault (GitHub #1807) on macOS/Python 3.12
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".vtk":
        output_path = output_path.with_suffix(".vtk")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ms_min_m = (
        DEFAULT_MIN_TETRAHEDRAL_MESH_SIZE_M if min_mesh_size is None else min_mesh_size
    )
    ms_max_m = (
        DEFAULT_MAX_TETRAHEDRAL_MESH_SIZE_M if max_mesh_size is None else max_mesh_size
    )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            filament_file = tmpdir / "coils_filament.txt"
            _write_parastell_filament_file(coils, filament_file, scale=100.0)

            # ParaStell width/thickness in cm; our width/height in m
            width_cm = width * 100
            thickness_cm = height * 100

            with suppress_output():
                ms = MagnetSetFromFilaments(
                    str(filament_file),
                    width=width_cm,
                    thickness=thickness_cm,
                    toroidal_extent=360.0,
                    case_thickness=0.0,
                    scale=100.0,
                    start_line=3,
                )
                ms.populate_magnet_coils()
                ms.build_magnet_coils()

            if not ms.coil_solids or len(ms.coil_solids) == 0:  # pragma: no cover
                _last_parastell_error = "ParaStell build produced no coil solids"
                return None

            # ParaStell/Gmsh expect mesh sizes in [cm]; convert from internal [m]
            try:
                with suppress_output():
                    ms.mesh_magnets_gmsh(
                        min_mesh_size=ms_min_m * 100,
                        max_mesh_size=ms_max_m * 100,
                    )
                    gmsh.write(str(output_path))
                    if save_msh:
                        msh_path = output_path.with_suffix(".msh")
                        gmsh.write(str(msh_path))
            finally:
                # Always finalize so next ParaStell call gets clean Gmsh state
                try:
                    gmsh.clear()
                    gmsh.finalize()
                except Exception:
                    pass

        _rescale_vtk_points(output_path, 1.0 / 100.0)
        if save_msh:
            msh_path = output_path.with_suffix(".msh")
            if msh_path.exists():
                _rescale_msh_points(msh_path, 1.0 / 100.0)

        return output_path
    except Exception as e:  # pragma: no cover
        _last_parastell_error = str(e)
        return None


def _finite_build_coils_to_msh_parastell(
    coils: List,
    msh_path: Path,
    width: float,
    height: float,
    mesh_size: float,
    min_mesh_size: Optional[float] = None,
    max_mesh_size: Optional[float] = None,
) -> Optional[Tuple[Path, List[int]]]:
    """
    Use ParaStell to build and mesh finite-build coils, writing a Gmsh .msh file.

    Wrapper around the ParaStell build + mesh logic. When successful, all coils
    are meshed together and returned as coil indices [0, 1, ..., n-1].

    Parameters
    ----------
    coils : List
        List of simsopt Coil objects.
    msh_path : Path
        Output .msh file path.
    width : float
        Cross-section width [m].
    height : float
        Cross-section height [m].
    mesh_size : float
        Characteristic element size [m] (used for both min/max when not specified).
    min_mesh_size : float, optional
        Gmsh minimum element size [m]. If None, uses mesh_size.
    max_mesh_size : float, optional
        Gmsh maximum element size [m]. If None, uses mesh_size.

    Returns
    -------
    tuple[Path, list[int]] or None
        (msh_path, list of coil indices) if successful; None on failure.
    """
    import os

    global _last_parastell_error
    try:
        from parastell.magnet_coils import MagnetSetFromFilaments  # type: ignore[import-untyped]
    except ImportError as e:  # pragma: no cover
        _last_parastell_error = str(e)
        return None

    try:
        import gmsh  # type: ignore[import-untyped]
    except ImportError as e:  # pragma: no cover
        _last_parastell_error = f"gmsh: {e}"
        return None

    # Avoid Gmsh/OpenMP multithreading segfault
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    msh_path = Path(msh_path)
    if msh_path.suffix.lower() != ".msh":
        msh_path = msh_path.with_suffix(".msh")
    msh_path.parent.mkdir(parents=True, exist_ok=True)

    ms_min = min_mesh_size if min_mesh_size is not None else mesh_size
    ms_max = max_mesh_size if max_mesh_size is not None else mesh_size

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            filament_file = tmpdir / "coils_filament.txt"
            _write_parastell_filament_file(coils, filament_file, scale=100.0)

            width_cm = width * 100
            thickness_cm = height * 100

            with suppress_output():
                ms = MagnetSetFromFilaments(
                    str(filament_file),
                    width=width_cm,
                    thickness=thickness_cm,
                    toroidal_extent=360.0,
                    case_thickness=0.0,
                    scale=100.0,
                    start_line=3,
                )
                ms.populate_magnet_coils()
                ms.build_magnet_coils()

            if not ms.coil_solids or len(ms.coil_solids) == 0:  # pragma: no cover
                _last_parastell_error = "ParaStell build produced no coil solids"
                return None

            try:
                with suppress_output():
                    ms.mesh_magnets_gmsh(
                        min_mesh_size=ms_min * 100,
                        max_mesh_size=ms_max * 100,
                    )
                    gmsh.write(str(msh_path))
            finally:
                try:
                    gmsh.clear()
                    gmsh.finalize()
                except Exception:
                    pass

        _rescale_msh_points(msh_path, 1.0 / 100.0)
        coil_indices = list(range(len(coils)))
        return (msh_path, coil_indices)
    except Exception as e:  # pragma: no cover
        _last_parastell_error = str(e)
        return None
