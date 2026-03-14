"""DOLFINx vs scikit-fem feasibility tests.

Benchmark and accuracy tests from the structural speedup plan.
Requires both dolfinx and skfem (skipped if either missing).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tests.structural_analysis.conftest import _build_box_msh


def _body_force_fn(coords: np.ndarray) -> np.ndarray:
    """Simple body force for benchmark."""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    return 1e6 * np.column_stack([x * (1 - x), y * (1 - y), z * (1 - z)])


def _run_skfem_n_times(msh_path: Path, n: int, E: float, nu: float) -> float:
    """Run skfem solves; return total time."""
    from stellcoilbench.structural_analysis import (
        _load_mesh_skfem,
        _solve_elasticity_skfem,
    )

    sk_mesh = _load_mesh_skfem(msh_path)
    force = np.zeros((sk_mesh.p.shape[1], 3))
    t0 = time.perf_counter()
    for _ in range(n):
        _ = _solve_elasticity_skfem(sk_mesh, force, E, nu, body_force_fn=_body_force_fn)
    return time.perf_counter() - t0


def _run_dolfinx_n_times(msh_path: Path, n: int, E: float, nu: float) -> float:
    """Run DOLFINx solves (mesh load once); return total time."""
    import basix.ufl
    import dolfinx
    import dolfinx.fem
    import dolfinx.fem.petsc
    import ufl
    from mpi4py import MPI

    from stellcoilbench.structural_analysis._common import BC_Z_FRACTION

    try:
        from dolfinx.io.gmsh import read_from_msh
    except ImportError:
        from dolfinx.io.gmshio import read_from_msh

    result = read_from_msh(str(msh_path), MPI.COMM_WORLD, rank=0, gdim=3)
    mesh = result.mesh if hasattr(result, "mesh") else result[0]
    coords = mesh.geometry.x.copy()
    q_degree = 2

    q_el = basix.ufl.quadrature_element(
        "tetrahedron", value_shape=(3,), degree=q_degree, scheme="default"
    )
    Q = dolfinx.fem.functionspace(mesh, q_el)
    f_func = dolfinx.fem.Function(Q, name="BodyForce")
    q_coords = Q.tabulate_dof_coordinates()
    f_func.x.array[:] = _body_force_fn(q_coords).flatten()
    f_func.x.scatter_forward()

    el = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    V = dolfinx.fem.functionspace(mesh, el)
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_val = E / (2 * (1 + nu))

    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return lam * ufl.nabla_div(w) * ufl.Identity(3) + 2 * mu_val * epsilon(w)

    a = ufl.inner(sigma(u_trial), epsilon(v_test)) * ufl.dx
    dx_q = ufl.Measure(
        "dx",
        domain=mesh,
        metadata={"quadrature_degree": q_degree, "quadrature_scheme": "default"},
    )
    L_form = ufl.inner(f_func, v_test) * dx_q

    fdim = mesh.topology.dim - 1
    threshold = coords[:, 2].min() + BC_Z_FRACTION * (
        coords[:, 2].max() - coords[:, 2].min()
    )
    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: x[2] <= threshold
    )
    bc_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    zero = dolfinx.fem.Constant(mesh, np.zeros(3, dtype=dolfinx.default_scalar_type))
    bcs = [dolfinx.fem.dirichletbc(zero, bc_dofs, V)]

    import inspect as _inspect

    kw = {"bcs": bcs, "petsc_options": {"ksp_type": "preonly", "pc_type": "lu"}}
    if (
        "petsc_options_prefix"
        in _inspect.signature(dolfinx.fem.petsc.LinearProblem.__init__).parameters
    ):
        kw["petsc_options_prefix"] = "xval_"
    problem = dolfinx.fem.petsc.LinearProblem(a, L_form, **kw)

    t0 = time.perf_counter()
    for _ in range(n):
        _ = problem.solve()
    return time.perf_counter() - t0


def test_dolfinx_faster_than_skfem(tmp_path: Path) -> None:
    """DOLFINx should be faster than skfem for assembly+solve (mesh loaded once)."""
    pytest.importorskip("dolfinx", reason="DOLFINx required")
    pytest.importorskip("skfem", reason="skfem required")
    pytest.importorskip("gmsh", reason="gmsh required")

    msh_path = _build_box_msh(tmp_path, nx=6)
    E, nu = 100e9, 0.3
    n_solves = 5

    devnull = open(os.devnull, "w")
    old_err = sys.stderr
    sys.stderr = devnull
    try:
        t_skfem = _run_skfem_n_times(msh_path, n_solves, E, nu)
        t_dolfinx = _run_dolfinx_n_times(msh_path, n_solves, E, nu)
    finally:
        sys.stderr = old_err
        devnull.close()

    speedup = t_skfem / t_dolfinx if t_dolfinx > 0 else 0
    assert speedup >= 1.2, (
        f"DOLFINx should be ≥1.2× faster: skfem={t_skfem:.2f}s, dolfinx={t_dolfinx:.2f}s (speedup={speedup:.2f})"
    )
