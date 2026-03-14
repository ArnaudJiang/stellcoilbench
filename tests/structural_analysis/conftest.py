"""Shared fixtures and helpers for structural_analysis tests."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from typing import Callable

# Artifacts dir for convergence plots (shared with test_fem_benchmarks)
_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "fem_benchmarks"


def _mms_derive(
    u_syms: list,
    E: float,
    nu: float,
) -> tuple[Callable, Callable]:
    """
    Given symbolic u = [u_x(x,y,z), u_y, u_z], derive f = -div(σ(u)) and
    return lambdified (body_force_fn, u_exact_fn).
    """
    import sympy as sp

    x, y, z = sp.symbols("x y z", real=True)

    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    grad_u = sp.Matrix([[sp.diff(u_syms[i], v) for v in (x, y, z)] for i in range(3)])
    eps = (grad_u + grad_u.T) / 2
    tr_eps = eps.trace()
    sigma = lam * tr_eps * sp.eye(3) + 2 * mu * eps

    div_sigma = sp.Matrix(
        [
            sum(sp.diff(sigma[i, j], v) for j, v in enumerate((x, y, z)))
            for i in range(3)
        ]
    )
    f_vec = -div_sigma

    f_lam = sp.lambdify(
        (x, y, z),
        [sp.simplify(f_vec[i]) for i in range(3)],
        modules="numpy",
    )
    u_lam = sp.lambdify(
        (x, y, z),
        [u_syms[i] for i in range(3)],
        modules="numpy",
    )

    def body_force_fn(coords: np.ndarray) -> np.ndarray:
        out = np.zeros_like(coords)
        fx, fy, fz = f_lam(coords[:, 0], coords[:, 1], coords[:, 2])
        for c, val in enumerate([fx, fy, fz]):
            out[:, c] = np.asarray(val, dtype=np.float64).ravel()
        return out

    def u_exact_fn(coords: np.ndarray) -> np.ndarray:
        out = np.zeros_like(coords)
        ux, uy, uz = u_lam(coords[:, 0], coords[:, 1], coords[:, 2])
        for c, val in enumerate([ux, uy, uz]):
            out[:, c] = np.asarray(val, dtype=np.float64).ravel()
        return out

    return body_force_fn, u_exact_fn


def _mms_full_dirichlet(E: float, nu: float):
    """MMS with u_exact=0 on all faces of [0,1]³ (all-Dirichlet compatible)."""
    import sympy as sp

    x, y, z = sp.symbols("x y z", real=True)
    A = sp.Float(1e-4)
    base = sp.sin(sp.pi * x) * sp.sin(sp.pi * y) * sp.sin(sp.pi * z)
    u_syms = [A * base, A * base, A * base]
    return _mms_derive(u_syms, E, nu)


def _mms_production_bc(E: float, nu: float):
    """MMS compatible with production BCs (bottom-15% clamped, rest free)."""
    import sympy as sp

    x, y, z = sp.symbols("x y z", real=True)
    A = sp.Float(1e-4)
    pxy = x * (1 - x) * y * (1 - y)
    u_syms = [A * pxy * z**2, A / 2 * pxy * z**2, A / 2 * pxy * z**2]
    return _mms_derive(u_syms, E, nu)


def _mms_smooth_neumann(E: float, nu: float):
    """MMS with smooth polynomial, all free faces except bottom clamped."""
    import sympy as sp

    x, y, z = sp.symbols("x y z", real=True)
    A = sp.Float(1e-4)
    base = sp.sin(sp.pi * x) * sp.sin(sp.pi * y) * sp.sin(sp.pi * z)
    u_syms = [A * base, sp.Float(0.8) * A * base, sp.Float(0.6) * A * base]
    return _mms_derive(u_syms, E, nu)


def _require_both_backends() -> None:
    """Skip the calling test unless both DOLFINx and scikit-fem are importable."""
    pytest.importorskip("dolfinx", reason="DOLFINx not installed")
    pytest.importorskip("skfem", reason="scikit-fem not installed")
    pytest.importorskip("meshio", reason="meshio not installed")


def _build_box_msh(tmp_path: Path, nx: int = 3) -> Path:
    """Build a unit-cube tetrahedral mesh with Gmsh and write it to tmp_path."""
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("box")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()

    vols = gmsh.model.getEntities(3)
    gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], tag=1)

    lc = 1.0 / nx
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(3)

    msh_path = tmp_path / "box.msh"
    gmsh.write(str(msh_path))
    gmsh.finalize()
    return msh_path


def _build_solenoid_msh(
    tmp_path: Path,
    r_i: float = 0.1,
    r_e: float = 0.2,
    height: float = 0.1,
    n_r: int = 4,
    n_theta: int = 8,
    n_z: int = 4,
) -> Path:
    """Build a cylindrical annulus mesh with Gmsh (OpenCASCADE).

    Creates an annulus r_i <= r <= r_e, 0 <= z <= height, centered on z-axis.
    Tags physical groups: volume (tag 1), bottom face z=0 (tag 2), top face z=height (tag 3).

    Parameters
    ----------
    tmp_path : Path
        Directory for the output .msh file.
    r_i : float
        Inner radius [m].
    r_e : float
        Outer radius [m].
    height : float
        Cylinder height [m].
    n_r, n_theta, n_z : int
        Mesh density hints (characteristic length derived from these).

    Returns
    -------
    Path
        Path to the generated .msh file.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("solenoid")
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, r_e)
    outer_tag = gmsh.model.occ.getEntities(3)[-1][1]
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, r_i)
    inner_tag = gmsh.model.occ.getEntities(3)[-1][1]
    out, _ = gmsh.model.occ.cut([(3, outer_tag)], [(3, inner_tag)])
    gmsh.model.occ.synchronize()

    vol_tag = out[0][1]
    gmsh.model.addPhysicalGroup(3, [vol_tag], tag=1)
    gmsh.model.setPhysicalName(3, 1, "Volume")

    boundary = gmsh.model.getBoundary([(3, vol_tag)], oriented=False)
    bottom_surfs = []
    top_surfs = []
    for dim, tag in boundary:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(com[2]) < 1e-12:
            bottom_surfs.append(tag)
        elif abs(com[2] - height) < 1e-12:
            top_surfs.append(tag)
    if bottom_surfs:
        gmsh.model.addPhysicalGroup(2, bottom_surfs, tag=2)
        gmsh.model.setPhysicalName(2, 2, "Bottom")
    if top_surfs:
        gmsh.model.addPhysicalGroup(2, top_surfs, tag=3)
        gmsh.model.setPhysicalName(2, 3, "Top")

    h_r = (r_e - r_i) / max(n_r, 1)
    h_theta = (2 * 3.14159 * r_e) / max(n_theta, 1)
    h_z = height / max(n_z, 1)
    lc = min(h_r, h_theta, h_z)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc * 1.5)
    gmsh.model.mesh.generate(3)

    msh_path = tmp_path / "solenoid.msh"
    gmsh.write(str(msh_path))
    gmsh.finalize()
    return msh_path


def _save_fem_artifact_fig(fig, name: str) -> None:
    """Save figure to artifacts dir for FEM convergence plots."""
    import matplotlib.pyplot as plt

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _ARTIFACTS_DIR / name
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")
