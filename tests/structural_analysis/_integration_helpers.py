"""Shared helpers for structural_analysis integration tests (MMS, cross-validation)."""

from __future__ import annotations

import inspect

import numpy as np
from pathlib import Path

from stellcoilbench.structural_analysis._common import BC_Z_FRACTION


def _import_ok(module: str) -> bool:
    """Return True if module can be imported."""
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _solve_with_dolfinx(
    msh_path: Path,
    body_force_fn,
    E: float,
    nu: float,
    degree: int = 1,
    bc_mode: str = "bottom",
    q_degree: int = 2,
):
    """Solve linear elasticity with DOLFINx."""
    import basix.ufl
    import dolfinx
    import dolfinx.fem
    import dolfinx.fem.petsc
    import ufl
    from mpi4py import MPI

    try:
        from dolfinx.io.gmsh import read_from_msh
    except ImportError:
        from dolfinx.io.gmshio import read_from_msh

    result = read_from_msh(str(msh_path), MPI.COMM_WORLD, rank=0, gdim=3)
    mesh = result.mesh if hasattr(result, "mesh") else result[0]
    coords = mesh.geometry.x.copy()

    q_el = basix.ufl.quadrature_element(
        "tetrahedron", value_shape=(3,), degree=q_degree, scheme="default"
    )
    Q = dolfinx.fem.functionspace(mesh, q_el)
    f_func = dolfinx.fem.Function(Q, name="BodyForce")
    q_coords = Q.tabulate_dof_coordinates()
    force_vals = body_force_fn(q_coords)
    f_func.x.array[:] = force_vals.flatten()
    f_func.x.scatter_forward()

    el = basix.ufl.element("Lagrange", "tetrahedron", degree, shape=(3,))
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

    tdim = mesh.topology.dim
    fdim = tdim - 1

    if bc_mode == "all":
        boundary_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.full(x.shape[1], True)
        )
    else:
        z_min = coords[:, 2].min()
        z_range = coords[:, 2].max() - z_min
        threshold = z_min + BC_Z_FRACTION * z_range
        boundary_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: x[2] <= threshold
        )

    bc_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    zero = dolfinx.fem.Constant(mesh, np.zeros(3, dtype=dolfinx.default_scalar_type))
    bcs = [dolfinx.fem.dirichletbc(zero, bc_dofs, V)]

    lp_sig = inspect.signature(dolfinx.fem.petsc.LinearProblem.__init__)
    kw = {"bcs": bcs, "petsc_options": {"ksp_type": "preonly", "pc_type": "lu"}}
    if "petsc_options_prefix" in lp_sig.parameters:
        kw["petsc_options_prefix"] = "xval_"
    problem = dolfinx.fem.petsc.LinearProblem(a, L_form, **kw)
    uh = problem.solve()

    eps_u = ufl.sym(ufl.grad(uh))
    sig = lam * ufl.nabla_div(uh) * ufl.Identity(3) + 2 * mu_val * eps_u
    s_dev = sig - (1.0 / 3.0) * ufl.tr(sig) * ufl.Identity(3)
    vm_ufl = ufl.sqrt(1.5 * ufl.inner(s_dev, s_dev))

    S_el = basix.ufl.element("DG", "tetrahedron", 0)
    S_space = dolfinx.fem.functionspace(mesh, S_el)
    ip_s = S_space.element.interpolation_points
    if callable(ip_s):
        ip_s = ip_s()
    vm_expr = dolfinx.fem.Expression(vm_ufl, ip_s)
    vm_field = dolfinx.fem.Function(S_space)
    vm_field.interpolate(vm_expr)

    u_all_dofs = uh.x.array.reshape(-1, 3)
    dof_coords = V.tabulate_dof_coordinates()
    if u_all_dofs.shape[0] != coords.shape[0]:
        from scipy.spatial import cKDTree

        tree = cKDTree(dof_coords)
        _, idx = tree.query(coords)
        u_nodes = u_all_dofs[idx]
    else:
        u_nodes = u_all_dofs
    vm_cells = vm_field.x.array.copy()
    return u_nodes, vm_cells, coords


def _solve_with_skfem(msh_path: Path, body_force_fn, E: float, nu: float):
    """Solve linear elasticity with scikit-fem."""
    from stellcoilbench.structural_analysis import (
        _compute_von_mises_skfem,
        _load_mesh_skfem,
        _solve_elasticity_skfem,
    )

    sk_mesh = _load_mesh_skfem(msh_path)
    coords = sk_mesh.p.T
    force_array_placeholder = np.zeros((coords.shape[0], 3))
    u_nodes = _solve_elasticity_skfem(
        sk_mesh, force_array_placeholder, E, nu, body_force_fn=body_force_fn
    )
    vm_cells = _compute_von_mises_skfem(sk_mesh, u_nodes, E, nu)
    return u_nodes, vm_cells, coords


def _solve_with_skfem_allbc(msh_path: Path, body_force_fn, E: float, nu: float):
    """Solve with scikit-fem, clamping ALL boundary nodes (for MMS full-Dirichlet)."""
    import skfem as _skfem
    from skfem import condense, solve as skfem_solve
    from skfem.assembly import BilinearForm, LinearForm
    from skfem.element.discrete_field import DiscreteField
    from skfem.helpers import ddot, sym_grad, trace

    from stellcoilbench.structural_analysis import _load_mesh_skfem
    from stellcoilbench.structural_analysis._common import _lame_parameters

    sk_mesh = _load_mesh_skfem(msh_path)
    coords = sk_mesh.p.T

    lam, mu = _lame_parameters(E, nu)
    e = _skfem.ElementTetP1()
    elem = _skfem.ElementVector(e)
    ib = _skfem.Basis(sk_mesh, elem, intorder=4)

    @BilinearForm
    def stiffness(u, v, w):
        eps_u = sym_grad(u)
        eps_v = sym_grad(v)
        return lam * trace(eps_u) * trace(eps_v) + 2 * mu * ddot(eps_u, eps_v)

    K = stiffness.assemble(ib)

    gc = ib.global_coordinates()
    n_elem, n_qpts = gc.shape[1], gc.shape[2]
    q_coords = np.asarray(gc.transpose(1, 2, 0).reshape(-1, 3), dtype=np.float64)
    force = np.asarray(body_force_fn(q_coords), dtype=np.float64)
    force_at_qp = force.reshape(n_elem, n_qpts, 3).transpose(2, 0, 1)
    bf_qp = DiscreteField(force_at_qp)

    @LinearForm
    def body_load(v, w):
        return sum(v[c] * w["bf"][c] for c in range(3))

    f_vec = body_load.assemble(ib, bf=bf_qp)

    n_nodes = sk_mesh.p.shape[1]
    boundary_nodes = sk_mesh.boundary_nodes()
    fixed_dofs = np.concatenate([3 * boundary_nodes + c for c in range(3)])
    u_full = skfem_solve(*condense(K, f_vec, D=fixed_dofs, expand=True))
    u_nodes = u_full.reshape(n_nodes, 3)

    from stellcoilbench.structural_analysis import _compute_von_mises_skfem

    vm_cells = _compute_von_mises_skfem(sk_mesh, u_nodes, E, nu)
    return u_nodes, vm_cells, coords
