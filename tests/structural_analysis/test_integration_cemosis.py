"""Cemosis-style solenoid benchmark integration tests."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from typing import Callable

from conftest import _build_solenoid_msh, _save_fem_artifact_fig


def _analytical_u_r_solenoid(
    r: np.ndarray,
    j_theta: float,
    B_z: float,
    E: float,
    nu: float,
    r_i: float,
    r_e: float,
) -> np.ndarray:
    """
    Analytical radial displacement for infinite solenoid with uniform j_θ and B_z.

    Solves the ODE from Wilson87/Montgomery69 (Cemosis docs):
        d/dr(r du_r/dr) - u_r/r = -K * r * j_θ * B_z
    with K = (1+ν)(1-2ν)/(E(1-ν)). For uniform j and B_z (Δb_z=0), the particular
    solution is u_p(r) = -K * j_θ * B_z * r²/3. Homogeneous: u_h = C1*r + C2/r.
    Boundary conditions: σ_rr = 0 at r_i and r_e (free curved surfaces).

    Parameters
    ----------
    r : np.ndarray
        Radial coordinate(s) [m].
    j_theta : float
        Azimuthal current density [A/m²].
    B_z : float
        Axial magnetic field [T].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson ratio.
    r_i, r_e : float
        Inner and outer radii [m].

    Returns
    -------
    np.ndarray
        Radial displacement u_r(r) [m], same shape as r.
    """
    r = np.asarray(r, dtype=float)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_val = E / (2 * (1 + nu))
    K = (1 + nu) * (1 - 2 * nu) / (E * (1 - nu))

    # Particular solution: u_p = -K * j_theta * B_z * r^2 / 3
    u_p = -K * j_theta * B_z * r**2 / 3.0

    # σ_rr = 2(λ+μ)C1 - 2μ C2/r² - K*j_θ*B_z*r*(3λ+4μ)/3
    # At r_i and r_e: σ_rr = 0
    coeff = K * j_theta * B_z * (3 * lam + 4 * mu_val) / 3.0
    a11 = 2 * (lam + mu_val)
    a12_i = -2 * mu_val / (r_i**2)
    a12_e = -2 * mu_val / (r_e**2)
    rhs_i = coeff * r_i
    rhs_e = coeff * r_e
    det = a11 * (a12_i - a12_e)
    C2 = (a11 * (rhs_i - rhs_e)) / det
    C1 = (rhs_i - a12_i * C2) / a11

    u_r = C1 * r + np.where(np.abs(r) > 1e-14, C2 / r, 0.0) + u_p
    return u_r


def _solenoid_body_force(coords: np.ndarray, j_theta: float, B_z: float) -> np.ndarray:
    """
    J×B body force for solenoid: j = (0, j_θ, 0), B = (0, 0, B_z) → f = j_θ*B_z*(x/r, y/r, 0).

    Radially outward. For r≈0, returns zero to avoid singularity.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) Cartesian coordinates [m].
    j_theta : float
        Azimuthal current density [A/m²].
    B_z : float
        Axial magnetic field [T].

    Returns
    -------
    np.ndarray
        (N, 3) body force density [N/m³].
    """
    x, y = coords[:, 0], coords[:, 1]
    r = np.sqrt(x**2 + y**2)
    scale = np.where(r > 1e-12, j_theta * B_z / r, 0.0)
    f = np.zeros_like(coords)
    f[:, 0] = scale * x
    f[:, 1] = scale * y
    f[:, 2] = 0.0
    return f


def _solve_solenoid_dolfinx(
    msh_path: Path,
    body_force_fn: Callable[[np.ndarray], np.ndarray],
    E: float,
    nu: float,
    height: float,
    degree: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve linear elasticity on solenoid mesh with top and bottom clamped (Cemosis BC).

    Dirichlet u=0 on z<=eps and z>=height-eps; free on curved surfaces.

    Parameters
    ----------
    msh_path : Path
        Path to .msh file.
    body_force_fn : callable
        body_force_fn(coords) -> (N, 3) body force [N/m³].
    E, nu : float
        Material properties.
    height : float
        Cylinder height [m] for BC location.
    degree : int
        Finite element degree (1 = P1, 2 = P2, etc.).

    Returns
    -------
    u_nodes : ndarray (n_nodes, 3)
    vm_cells : ndarray (n_cells,)
    coords : ndarray (n_nodes, 3)
    """
    import inspect

    import basix.ufl
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

    q_degree = 2
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

    eps_bc = 1e-6 * height

    def on_clamped(x):
        # x has shape (3, n): columns are facet midpoints, row 2 is z
        z = x[2]
        return (z <= eps_bc) | (z >= height - eps_bc)

    tdim = mesh.topology.dim
    fdim = tdim - 1
    boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, on_clamped)
    bc_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    zero = dolfinx.fem.Constant(mesh, np.zeros(3, dtype=dolfinx.default_scalar_type))
    bcs = [dolfinx.fem.dirichletbc(zero, bc_dofs, V)]

    kw = {"bcs": bcs, "petsc_options": {"ksp_type": "preonly", "pc_type": "lu"}}
    if (
        "petsc_options_prefix"
        in inspect.signature(dolfinx.fem.petsc.LinearProblem.__init__).parameters
    ):
        kw["petsc_options_prefix"] = "solenoid_"
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


class TestCemosisSolenoidBenchmark:
    """Cemosis-style solenoid benchmark: J×B body force + linear elasticity.

    Infinite solenoid analytical solution (Wilson87/Montgomery69) for radial
    displacement; compare FEM at mid-plane z=height/2. Uses height=0.5 m so
    mid-plane is far from clamped ends and approaches the infinite-solenoid limit.
    """

    def test_solenoid_radial_displacement(self, tmp_path):
        """Build mesh, solve elasticity with clamped top/bottom, compare to analytical u_r."""
        pytest.importorskip("dolfinx", reason="DOLFINx required for Cemosis benchmark")
        pytest.importorskip("gmsh", reason="Gmsh required for mesh generation")

        r_i, r_e = 0.1, 0.2
        height = 0.5  # Taller cylinder so mid-plane is far from clamped ends
        E, nu = 128e9, 0.33
        j_theta = 1e7  # A/m²
        B_z = 5.0  # T

        msh_path = _build_solenoid_msh(
            tmp_path, r_i=r_i, r_e=r_e, height=height, n_r=5, n_theta=10, n_z=8
        )

        def body_force(coords):
            return _solenoid_body_force(coords, j_theta, B_z)

        u_nodes, _, coords = _solve_solenoid_dolfinx(
            msh_path, body_force, E, nu, height
        )

        # Extract radial displacement at mid-plane (z ≈ height/2)
        z_mid = height / 2
        eps_z = 0.05 * height  # 5% band for more stable sampling
        mid_mask = (coords[:, 2] >= z_mid - eps_z) & (coords[:, 2] <= z_mid + eps_z)

        x, y = coords[mid_mask, 0], coords[mid_mask, 1]
        r_nodes = np.sqrt(x**2 + y**2)
        u_x, u_y = u_nodes[mid_mask, 0], u_nodes[mid_mask, 1]
        u_r_fem = np.where(r_nodes > 1e-12, (x * u_x + y * u_y) / r_nodes, 0.0)

        u_r_analytical = _analytical_u_r_solenoid(
            r_nodes, j_theta, B_z, E, nu, r_i, r_e
        )

        rel_err = np.abs(u_r_fem - u_r_analytical) / (np.abs(u_r_analytical) + 1e-20)
        max_rel_err = float(np.max(rel_err))

        assert max_rel_err < 0.10, (
            f"Max relative error {max_rel_err:.2%} exceeds 10% threshold; "
            f"mid-plane nodes: {np.sum(mid_mask)}"
        )

    def test_solenoid_convergence_h_p(self, tmp_path):
        """Solenoid h- and p-convergence study; save plot to artifacts."""
        pytest.importorskip("dolfinx", reason="DOLFINx required")
        pytest.importorskip("gmsh", reason="Gmsh required")

        r_i, r_e = 0.1, 0.2
        height = 0.5
        E, nu = 128e9, 0.33
        j_theta = 1e7
        B_z = 5.0

        def body_force(coords):
            return _solenoid_body_force(coords, j_theta, B_z)

        def _compute_rms_rel_err(msh_path: Path, degree: int) -> float:
            """RMS relative error in u_r at mid-plane using fixed radial grid.

            Interpolates FEM solution onto 15 points from r_i to r_e (theta=0,
            z=height/2) for consistent sampling across mesh densities.
            """
            u_nodes, _, coords = _solve_solenoid_dolfinx(
                msh_path, body_force, E, nu, height, degree=degree
            )
            from scipy.interpolate import LinearNDInterpolator

            z_mid = height / 2
            r_grid = np.linspace(r_i, r_e, 15)
            x_grid = np.column_stack(
                [r_grid, np.zeros_like(r_grid), np.full_like(r_grid, z_mid)]
            )
            interp_ux = LinearNDInterpolator(coords, u_nodes[:, 0])
            interp_uy = LinearNDInterpolator(coords, u_nodes[:, 1])
            u_x_at = interp_ux(x_grid)
            u_y_at = interp_uy(x_grid)
            valid = np.isfinite(u_x_at) & np.isfinite(u_y_at)
            if np.all(valid):
                u_r_fem = np.where(r_grid > 1e-12, u_x_at, 0.0)  # theta=0 => u_r=u_x
                u_r_analytical = _analytical_u_r_solenoid(
                    r_grid, j_theta, B_z, E, nu, r_i, r_e
                )
            else:
                # Fallback: mid-plane band (5%) if interpolation fails at boundary
                eps_z = 0.05 * height
                mid_mask = (coords[:, 2] >= z_mid - eps_z) & (
                    coords[:, 2] <= z_mid + eps_z
                )
                x, y = coords[mid_mask, 0], coords[mid_mask, 1]
                r_nodes = np.sqrt(x**2 + y**2)
                u_x, u_y = u_nodes[mid_mask, 0], u_nodes[mid_mask, 1]
                u_r_fem = np.where(r_nodes > 1e-12, (x * u_x + y * u_y) / r_nodes, 0.0)
                u_r_analytical = _analytical_u_r_solenoid(
                    r_nodes, j_theta, B_z, E, nu, r_i, r_e
                )
            rel_err = np.abs(u_r_fem - u_r_analytical) / (
                np.abs(u_r_analytical) + 1e-20
            )
            return float(np.sqrt(np.mean(rel_err**2)))

        # h-convergence: vary mesh density
        h_configs = [
            (3, 6, 4),
            (4, 8, 5),
            (5, 10, 8),
            (6, 12, 10),
        ]
        h_vals = []  # approximate lc
        h_errors = []
        for n_r, n_theta, n_z in h_configs:
            msh_path = _build_solenoid_msh(
                tmp_path,
                r_i=r_i,
                r_e=r_e,
                height=height,
                n_r=n_r,
                n_theta=n_theta,
                n_z=n_z,
            )
            lc = min((r_e - r_i) / n_r, 2 * 3.14159 * r_e / n_theta, height / n_z)
            h_vals.append(lc)
            h_errors.append(_compute_rms_rel_err(msh_path, degree=1))

        # p-convergence: fixed mesh, vary degree (P1, P2; vertex dofs extracted for P2+)
        msh_p = _build_solenoid_msh(
            tmp_path, r_i=r_i, r_e=r_e, height=height, n_r=5, n_theta=10, n_z=8
        )
        p_degrees = [1, 2]
        p_errors = [_compute_rms_rel_err(msh_p, degree=d) for d in p_degrees]

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_h, ax_p = axes

        ax_h.semilogy(h_vals, h_errors, "o-")
        ax_h.set_xlabel("h (approximate mesh size)")
        ax_h.set_ylabel("RMS relative error in u_r")
        ax_h.set_title("Solenoid: h-convergence (P1)")

        if p_errors:
            ax_p.semilogy(p_degrees[: len(p_errors)], p_errors, "o-")
        ax_p.set_xlabel("Element degree (p)")
        ax_p.set_ylabel("RMS relative error in u_r")
        ax_p.set_title("Solenoid: p-convergence")

        fig.suptitle("Cemosis solenoid benchmark: h- and p-convergence")
        fig.tight_layout()
        _save_fem_artifact_fig(fig, "solenoid_convergence.png")

        # Sanity: finest h-config should have error < 10%
        assert h_errors[-1] < 0.10, f"Finest h-convergence error {h_errors[-1]:.2%}"
