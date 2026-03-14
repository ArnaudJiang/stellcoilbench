"""scikit-fem fallback backend for structural analysis.

All functions in this module require scikit-fem and meshio.  The parent
``__init__.py`` only calls into this module when ``_SKFEM_AVAILABLE``
is ``True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from simsopt.field import BiotSavart, Coil

from ._common import (
    _compute_jcross_b,
    _compute_z_threshold_for_fixed_support,
    _extract_tet_blocks_from_meshio,
    _lame_parameters,
    _prepare_structural_output_dir,
)
from .._optional_imports import optional_import

skfem = optional_import("skfem", "", fallback=None)  # type: ignore[assignment]
_meshio = optional_import("meshio", "", fallback=None)


def _load_mesh_skfem(msh_path: Path) -> "skfem.MeshTet1":
    """
    Read a Gmsh ``.msh`` file via meshio into a scikit-fem ``MeshTet1``.

    Parameters
    ----------
    msh_path : Path
        Path to a Gmsh ``.msh`` file.

    Returns
    -------
    skfem.MeshTet1
        Tetrahedral mesh.
    """
    import skfem as _skfem

    cells, block_ids, points = _extract_tet_blocks_from_meshio(msh_path)
    # MeshTet1 expects points (3, n_vertices) and cells (4, n_cells)
    mesh = _skfem.MeshTet1(points.T, cells.T.astype(np.intp))
    mesh._structural_cell_tags = block_ids  # type: ignore[attr-defined]
    return mesh


def _solve_elasticity_skfem(
    sk_mesh: "skfem.Mesh",
    body_force_array: np.ndarray,
    E: float,
    nu: float,
    *,
    coils: Optional[list["Coil"]] = None,
    bs: Optional["BiotSavart"] = None,
    cross_section_area: Optional[float] = None,
    width: float = 0.05,
    height: float = 0.05,
    body_force_fn: Optional[object] = None,
    cached_K=None,
    cached_ib=None,
    cached_fixed_dofs=None,
    return_assembly: bool = False,
    cached_K_ff=None,
    cached_free_dofs: Optional[np.ndarray] = None,
    bs_mutual_list: Optional[list] = None,
    cached_coil_frames: Optional[list] = None,
    cached_Breg_list: Optional[list] = None,
    mesh_coils: Optional[list["Coil"]] = None,
    all_coils: Optional[list["Coil"]] = None,
) -> np.ndarray:
    """
    Solve 3-D linear elasticity with scikit-fem.

    When ``coils``, ``bs``, and ``cross_section_area`` are provided, the
    body force is evaluated at quadrature points for proper h-convergence.
    When ``body_force_fn`` is provided (callable coords -> force), it is
    evaluated at quadrature points.  Otherwise, ``body_force_array`` at
    mesh nodes is interpolated to quadrature (legacy behavior).

    Parameters
    ----------
    sk_mesh : skfem.MeshTet1
        Tetrahedral mesh.
    body_force_array : np.ndarray
        Body-force density at mesh nodes, shape ``(n_nodes, 3)`` [N/m³].
        Ignored when coils/bs/cross_section_area are provided.
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson ratio.
    coils, bs, cross_section_area : optional
        When all provided, J × B is evaluated at quadrature points.
    body_force_fn : callable, optional
        ``(coords: ndarray (n, 3)) -> ndarray (n, 3)``.  When provided,
        evaluated at quadrature points for h-convergence.
    cached_K, cached_ib, cached_fixed_dofs : optional
        When all three are provided, skip stiffness assembly and Basis creation.
        Used for FD gradient speedup (K barely changes for small perturbations).
    cached_K_ff : scipy.sparse matrix, optional
        Pre-condensed stiffness (free DOFs only), sorted for solver. When
        provided with ``cached_free_dofs``, skips condense and csr_sort per solve.
    cached_free_dofs : np.ndarray, optional
        Free DOF indices for condensed solve. Required when ``cached_K_ff`` used.
    return_assembly : bool, optional
        When True and not using cached assembly, return
        ``(u_array, K, ib, fixed_dofs, K_ff, free_dofs)`` for caching.

    Returns
    -------
    u : np.ndarray
        Displacement vector at mesh nodes, shape ``(n_nodes, 3)``.
    """
    import skfem as _skfem
    from skfem.assembly import BilinearForm, LinearForm
    from skfem.element.discrete_field import DiscreteField
    from skfem.helpers import ddot, sym_grad, trace
    from skfem import solve as skfem_solve, condense

    use_cached = (
        cached_K is not None and cached_ib is not None and cached_fixed_dofs is not None
    )
    if use_cached:
        K = cached_K
        ib = cached_ib
    else:
        e = _skfem.ElementTetP1()
        elem = _skfem.ElementVector(e)
        ib = _skfem.Basis(sk_mesh, elem)
        lam, mu = _lame_parameters(E, nu)

        @BilinearForm
        def stiffness(u, v, w):
            eps_u = sym_grad(u)
            eps_v = sym_grad(v)
            return lam * trace(eps_u) * trace(eps_v) + 2 * mu * ddot(eps_u, eps_v)

        K = stiffness.assemble(ib)

    # Body force at quadrature points for proper h-convergence
    if coils is not None and bs is not None and cross_section_area is not None:
        gc = ib.global_coordinates()  # (3, n_elem, n_qpts)
        n_elem, n_qpts = gc.shape[1], gc.shape[2]
        q_coords = gc.transpose(1, 2, 0).reshape(-1, 3)
        jcross_kwargs = {
            "width": width,
            "height": height,
            "use_regularized": True,
        }
        if bs_mutual_list is not None:
            jcross_kwargs["bs_mutual_list"] = bs_mutual_list
        if cached_coil_frames is not None:
            jcross_kwargs["cached_coil_frames"] = cached_coil_frames
        if cached_Breg_list is not None:
            jcross_kwargs["cached_Breg_list"] = cached_Breg_list
        if mesh_coils is not None:
            jcross_kwargs["mesh_coils"] = mesh_coils
        if all_coils is not None:
            jcross_kwargs["all_coils"] = all_coils
        force = _compute_jcross_b(
            q_coords,
            coils,
            bs,
            cross_section_area,
            **jcross_kwargs,
        )
        force_at_qp = force.reshape(n_elem, n_qpts, 3).transpose(2, 0, 1)
        bf_qp = DiscreteField(force_at_qp)
    elif body_force_fn is not None:
        gc = ib.global_coordinates()
        n_elem, n_qpts = gc.shape[1], gc.shape[2]
        q_coords = np.asarray(
            gc.transpose(1, 2, 0).reshape(-1, 3),
            dtype=np.float64,
            order="C",
        )
        force = np.asarray(body_force_fn(q_coords), dtype=np.float64)
        if force.shape != (q_coords.shape[0], 3):
            raise ValueError(f"body_force_fn must return (n, 3), got {force.shape}")
        force_at_qp = force.reshape(n_elem, n_qpts, 3).transpose(2, 0, 1)
        bf_qp = DiscreteField(force_at_qp)
    else:
        n_nodes = sk_mesh.p.shape[1]
        if body_force_array.shape[0] != n_nodes or body_force_array.shape[1] != 3:
            raise ValueError(
                f"body_force_array shape {body_force_array.shape} does not match "
                f"(n_nodes={n_nodes}, 3)."
            )
        bf_dofs = np.zeros(ib.N)
        for i in range(n_nodes):
            bf_dofs[3 * i : 3 * i + 3] = body_force_array[i]
        bf_qp = ib.interpolate(bf_dofs)

    @LinearForm
    def body_load(v, w):
        return sum(v[c] * w["bf"][c] for c in range(3))

    f_vec = body_load.assemble(ib, bf=bf_qp)

    n_nodes = sk_mesh.p.shape[1]

    # Pin nodes: per-coil z-threshold (z <= z_min_i + 0.15*(z_max_i - z_min_i))
    # or global fallback when block tags unavailable.
    if use_cached:
        fixed_dofs = cached_fixed_dofs
    else:
        z = sk_mesh.p[2, :]
        cell_tags = getattr(sk_mesh, "_structural_cell_tags", None)
        cells_mat = sk_mesh.t.T  # (n_elem, 4) node indices per cell

        block_z_range: dict[int, tuple[float, float]] = {}
        if cell_tags is not None and len(cell_tags) == cells_mat.shape[0]:
            unique_tags = np.unique(cell_tags)
            if len(unique_tags) > 1:
                for tag in unique_tags:
                    mask = cell_tags == tag
                    node_ids = np.unique(cells_mat[mask].ravel())
                    z_vals = z[node_ids]
                    block_z_range[int(tag)] = (
                        float(np.min(z_vals)),
                        float(np.max(z_vals)),
                    )

        # Assign each node to a block (via first cell containing it)
        node_to_block: dict[int, int] = {}
        for c in range(cells_mat.shape[0]):
            tag = (
                int(cell_tags[c])
                if (cell_tags is not None and c < len(cell_tags))
                else 1
            )
            for n in cells_mat[c]:
                node_to_block.setdefault(int(n), tag)

        fixed_nodes_list: list[int] = []
        if block_z_range and node_to_block:
            for ni in range(n_nodes):
                tag = node_to_block.get(ni, 1)
                if tag not in block_z_range:
                    continue
                z_min_i, z_max_i = block_z_range[tag]
                threshold = _compute_z_threshold_for_fixed_support(
                    z_min_i, z_max_i, range_if_zero=1.0
                )
                if z[ni] <= threshold:
                    fixed_nodes_list.append(ni)
        else:
            z_min, z_max = float(z.min()), float(z.max())
            threshold = _compute_z_threshold_for_fixed_support(z_min, z_max)
            fixed_nodes_list = list(np.where(z <= threshold)[0])

        if not fixed_nodes_list:
            fixed_nodes_list = list(np.argsort(z)[: max(1, n_nodes // 10)])

        fixed_nodes = np.array(fixed_nodes_list, dtype=np.intp)
        fixed_dofs = np.concatenate([3 * fixed_nodes + c for c in range(3)])

    use_cached_K_ff = (
        use_cached and cached_K_ff is not None and cached_free_dofs is not None
    )
    if use_cached_K_ff:
        # Fast path: skip condense (avoids repeated csr_sort)
        from scipy.sparse.linalg import spsolve

        f_vec_flat = np.asarray(f_vec).ravel()
        f_f = f_vec_flat[cached_free_dofs]
        u_f = spsolve(cached_K_ff, f_f)
        u_full = np.zeros(ib.N, dtype=np.float64)
        u_full[cached_free_dofs] = np.asarray(u_f).ravel()
    else:
        u_full = skfem_solve(*condense(K, f_vec, D=fixed_dofs, expand=True))

    u_out = np.asarray(u_full).ravel().reshape(n_nodes, 3)
    if return_assembly and not use_cached:
        K_ff, _f_f, _x_full, free_dofs = condense(K, f_vec, D=fixed_dofs, expand=True)
        K_ff = K_ff.tocsr()
        K_ff.sort_indices()
        return u_out, K, ib, fixed_dofs, K_ff, np.asarray(free_dofs, dtype=np.intp)
    return u_out


def _compute_von_mises_skfem(
    sk_mesh: "skfem.Mesh",
    u_array: np.ndarray,
    E: float,
    nu: float,
) -> np.ndarray:
    r"""Compute element-wise Von Mises stress from nodal displacements.

    Strain tensor:

    .. math::
        \varepsilon = \tfrac{1}{2}(\nabla\mathbf{u} + \nabla\mathbf{u}^T)

    Cauchy stress (isotropic):

    .. math::
        \sigma = \lambda\mathrm{tr}(\varepsilon)I + 2\mu\varepsilon

    Von Mises equivalent stress (deviatoric :math:`\mathbf{s} = \sigma
    - \tfrac{1}{3}\mathrm{tr}(\sigma)I`):

    .. math::
        \sigma_{\mathrm{vm}} = \sqrt{\tfrac{3}{2}\,\mathbf{s}:\mathbf{s}}

    Uses fully vectorized numpy over all tetrahedra (no Python element loop).

    Parameters
    ----------
    sk_mesh : skfem.MeshTet1
        Tetrahedral mesh.
    u_array : np.ndarray
        Nodal displacements, shape ``(n_nodes, 3)``.
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson ratio.

    Returns
    -------
    np.ndarray
        Von Mises stress per element, shape ``(n_elements,)``.
    """
    lam, mu = _lame_parameters(E, nu)

    cells = sk_mesh.t.T  # (n_elem, 4)

    # Gather nodal coordinates and displacements for all elements at once
    pts = sk_mesh.p[:, cells].transpose(1, 2, 0)  # (n_elem, 4, 3)
    u_el = u_array[cells]  # (n_elem, 4, 3)

    # Jacobian: X[e] = pts[e,1:,:] - pts[e,0,:]  →  (n_elem, 3, 3)
    X = pts[:, 1:, :] - pts[:, 0:1, :]

    # Batch-invert the transposed Jacobians
    # inv_Xt[e] = inv(X[e].T)  →  (n_elem, 3, 3)
    Xt = X.transpose(0, 2, 1)
    det = np.linalg.det(Xt)
    degenerate = np.abs(det) < 1e-30
    # Replace degenerate Jacobians with identity to avoid singular matrix errors
    Xt_safe = Xt.copy()
    Xt_safe[degenerate] = np.eye(3)
    inv_Xt = np.linalg.inv(Xt_safe)  # (n_elem, 3, 3)

    # Displacement differences: du[e,i,:] = u_el[e,i+1,:] - u_el[e,0,:]
    du = u_el[:, 1:, :] - u_el[:, 0:1, :]  # (n_elem, 3, 3)

    # grad_u[e] = sum_i outer(du[e,i], dN[e,i])
    # where dN[e] = inv_Xt[e] of shape (3,3), row i = grad(N_{i+1})
    # grad_u[e,j,k] = sum_i du[e,i,j] * inv_Xt[e,i,k]
    grad_u = np.einsum("eij,eik->ejk", du, inv_Xt)  # (n_elem, 3, 3)

    # Strain tensor: ε = 0.5*(grad_u + grad_u^T)
    eps = 0.5 * (grad_u + grad_u.transpose(0, 2, 1))  # (n_elem, 3, 3)

    # Stress: σ = λ tr(ε) I + 2μ ε
    tr_eps = np.trace(eps, axis1=1, axis2=2)  # (n_elem,)
    eye3 = np.eye(3)[np.newaxis, :, :]  # (1, 3, 3)
    sig = lam * tr_eps[:, np.newaxis, np.newaxis] * eye3 + 2 * mu * eps

    # Deviatoric stress: s = σ − (1/3) tr(σ) I
    tr_sig = np.trace(sig, axis1=1, axis2=2)  # (n_elem,)
    s_dev = sig - (1.0 / 3.0) * tr_sig[:, np.newaxis, np.newaxis] * eye3

    # Von Mises: σ_vm = sqrt(3/2 s:s)
    vm = np.sqrt(1.5 * np.sum(s_dev * s_dev, axis=(1, 2)))  # (n_elem,)

    # Zero out degenerate elements
    vm[degenerate] = 0.0

    return vm


def _export_skfem(
    sk_mesh: "skfem.Mesh",
    u_array: np.ndarray,
    vm: np.ndarray,
    output_dir: Path,
) -> dict[str, str]:
    """Write scikit-fem results to VTK via meshio.

    Writes ``structural_results.vtk`` in *output_dir* with point data
    ``Displacement`` and cell data ``VonMisesStress``.

    Parameters
    ----------
    sk_mesh : skfem.MeshTet1
        Tetrahedral mesh.
    u_array : np.ndarray
        Nodal displacement, shape ``(n_nodes, 3)``.
    vm : np.ndarray
        Per-element Von Mises stress, shape ``(n_elements,)``.
    output_dir : Path
        Directory for output.

    Returns
    -------
    dict[str, str]
        Key ``"structural_vtk"`` with path to the written file.
    """
    import meshio as _meshio

    output_dir = _prepare_structural_output_dir(output_dir)

    cells = [("tetra", sk_mesh.t.T)]
    points = sk_mesh.p.T

    vtk_path = output_dir / "structural_results.vtk"
    m = _meshio.Mesh(
        points,
        cells,
        point_data={"Displacement": u_array},
        cell_data={"VonMisesStress": [vm]},
    )
    _meshio.vtk.write(str(vtk_path), m)
    return {"structural_vtk": str(vtk_path)}
