"""DOLFINx (FEniCSx) backend for structural analysis.

All functions in this module require DOLFINx and its dependencies
(basix, ufl, mpi4py, PETSc).  The parent ``__init__.py`` only calls
into this module when ``_DOLFINX_AVAILABLE`` is ``True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..mpi_utils import proc0_print

from ._common import (
    BC_Z_FRACTION,
    _compute_jcross_b,
    _compute_z_threshold_for_fixed_support,
    _extract_tet_blocks_from_meshio,
    _lame_parameters,
    _prepare_structural_output_dir,
)

if TYPE_CHECKING:
    from mpi4py import MPI
    from simsopt.field import BiotSavart, Coil

try:
    import dolfinx  # type: ignore[import-untyped]
    import dolfinx.fem  # type: ignore[import-untyped]
    import dolfinx.fem.petsc  # type: ignore[import-untyped]
    import dolfinx.io  # type: ignore[import-untyped]
    import dolfinx.mesh  # type: ignore[import-untyped]
    import basix.ufl  # type: ignore[import-untyped]
    import ufl  # type: ignore[import-untyped]
    from mpi4py import MPI as _MPI  # noqa: N811

    from .._optional_imports import optional_import

    _read_from_msh = optional_import("dolfinx.io.gmsh", "read_from_msh", fallback=None)
    if _read_from_msh is None:
        _read_from_msh = optional_import(
            "dolfinx.io.gmshio", "read_from_msh", fallback=None
        )
except ImportError:
    pass


def _create_mesh_from_points_cells(
    x: np.ndarray,
    cells: np.ndarray,
    *,
    comm: "MPI.Comm | None" = None,
    block_ids: Optional[np.ndarray] = None,
) -> "dolfinx.mesh.Mesh":
    """Create a DOLFINx mesh from vertex coordinates and cell connectivity.

    Used for deformed coil meshes where topology is fixed but node positions change.
    Uses COMM_SELF by default so only the calling rank participates (safe for
    optimization loops where only rank 0 evaluates objectives).

    Parameters
    ----------
    x : np.ndarray
        Vertex coordinates, shape (n_vertices, 3), dtype np.float64.
    cells : np.ndarray
        Cell connectivity, shape (n_cells, 4), dtype np.int64. 0-based indices.
    comm : MPI.Comm, optional
        MPI communicator. Default COMM_SELF for rank-local mesh.
    block_ids : np.ndarray, optional
        Block tag per cell (1-based), shape (n_cells,), dtype np.int32. When
        provided, mesh._structural_cell_tags is set for per-coil BC support.

    Returns
    -------
    dolfinx.mesh.Mesh
        Tetrahedral mesh.
    """
    if comm is None:
        comm = _MPI.COMM_SELF
    x = np.asarray(x, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    ufl_tet = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
    mesh = dolfinx.mesh.create_mesh(
        comm,
        cells,
        ufl_tet,
        x,
        partitioner=dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.none),
    )
    if block_ids is not None and block_ids.shape[0] == cells.shape[0]:
        tdim = mesh.topology.dim
        n_cells_local = mesh.topology.index_map(tdim).size_local
        im = mesh.topology.index_map(tdim)
        local_indices = np.arange(n_cells_local, dtype=np.int32)
        global_indices = im.local_to_global(local_indices)
        tag_values = np.asarray(
            [block_ids[gi] for gi in global_indices],
            dtype=np.int32,
        )
        cell_tags = dolfinx.mesh.meshtags(
            mesh,
            tdim,
            local_indices,
            tag_values,
        )
        mesh._structural_cell_tags = cell_tags  # type: ignore[attr-defined]
    return mesh


def _load_mesh_dolfinx_tet_only(
    msh_path: Path,
    *,
    comm: "MPI.Comm | None" = None,
) -> tuple["dolfinx.mesh.Mesh", "dolfinx.mesh.MeshTags", "dolfinx.mesh.MeshTags"]:
    """Load mesh via meshio, extract tetrahedra only, create DOLFINx mesh.

    Fallback for Gmsh meshes with mixed cell types (e.g. tet+prism)
    that cause 'Non-matching UFL cell and mesh cell shapes' with DOLFINx's
    native tetrahedral elements. Uses :func:`_extract_tet_blocks_from_meshio`
    to keep only tetrahedral cells; facet tags are returned empty.

    Parameters
    ----------
    msh_path : Path
        Path to the ``.msh`` file.
    comm : MPI.Comm, optional
        MPI communicator for mesh creation. Default COMM_SELF (rank-local mesh),
        required when loading on rank 0 only to avoid collective deadlock.

    Returns
    -------
    tuple
        ``(mesh, cell_tags, facet_tags)``. Same signature as
        :func:`_load_mesh_dolfinx`.
    """
    if comm is None:
        comm = _MPI.COMM_SELF
    cells, block_ids, x = _extract_tet_blocks_from_meshio(msh_path)

    ufl_tet = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
    mesh = dolfinx.mesh.create_mesh(
        comm,
        cells,
        ufl_tet,
        x,
        partitioner=dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.none),
    )
    tdim = mesh.topology.dim
    fdim = tdim - 1
    n_cells_local = mesh.topology.index_map(tdim).size_local
    im = mesh.topology.index_map(tdim)
    local_indices = np.arange(n_cells_local, dtype=np.int32)
    global_indices = im.local_to_global(local_indices)
    tag_values = np.array(
        [block_ids[gi] for gi in global_indices],
        dtype=np.int32,
    )
    cell_tags = dolfinx.mesh.meshtags(
        mesh,
        tdim,
        np.arange(n_cells_local, dtype=np.int32),
        tag_values,
    )
    facet_tags = dolfinx.mesh.meshtags(
        mesh, fdim, np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    )
    return mesh, cell_tags, facet_tags


def _load_mesh_dolfinx(
    msh_path: Path,
    *,
    comm: "MPI.Comm | None" = None,
) -> tuple["dolfinx.mesh.Mesh", "dolfinx.mesh.MeshTags", "dolfinx.mesh.MeshTags"]:
    """Read a Gmsh ``.msh`` file into a DOLFINx mesh.

    If the mesh has no physical groups (common with some mesh generators),
    synthetic groups are added for all 3-D and 2-D entities so
    that DOLFINx's ``model_to_mesh`` / ``read_from_msh`` can proceed.

    If the mesh has mixed cell types (e.g. tet+prism), falls back to
    meshio-based tet-only extraction to avoid 'Non-matching UFL cell'
    errors.

    Parameters
    ----------
    msh_path : Path
        Path to a Gmsh ``.msh`` file (version 2 or 4).
    comm : MPI.Comm, optional
        MPI communicator for mesh creation. Default COMM_SELF (rank-local mesh),
        required when loading on rank 0 only to avoid collective deadlock.

    Returns
    -------
    mesh : dolfinx.mesh.Mesh
        The finite-element mesh.
    cell_tags : dolfinx.mesh.MeshTags
        Cell (volume) markers from Gmsh physical groups.
    facet_tags : dolfinx.mesh.MeshTags
        Facet (surface) markers from Gmsh physical groups.
    """
    if comm is None:
        comm = _MPI.COMM_SELF
    try:
        result = _read_from_msh(str(msh_path), comm, rank=0, gdim=3)
    except RuntimeError as exc:
        if "physical groups" not in str(exc).lower():
            raise
        result = _load_mesh_adding_physical_groups(msh_path, comm=comm)

    if hasattr(result, "mesh"):
        return result.mesh, result.cell_tags, result.facet_tags
    return result


def _load_mesh_adding_physical_groups(
    msh_path: Path,
    *,
    comm: "MPI.Comm | None" = None,
) -> tuple["dolfinx.mesh.Mesh", "dolfinx.mesh.MeshTags", "dolfinx.mesh.MeshTags"]:
    """Fallback loader that injects physical groups into the gmsh model.

    Some mesh generators write ``.msh`` files
    without physical groups.  DOLFINx requires them.  This function
    opens the file with the gmsh API, creates a physical group for
    every 3-D entity (volumes) and every 2-D entity (surfaces), and
    then calls DOLFINx's ``model_to_mesh``.

    Parameters
    ----------
    msh_path : Path
        Path to the ``.msh`` file.
    comm : MPI.Comm, optional
        MPI communicator for mesh creation. Default COMM_SELF (rank-local mesh),
        required when loading on rank 0 only to avoid collective deadlock.

    Returns
    -------
    tuple
        ``(mesh, cell_tags, facet_tags)`` — same as
        :func:`_load_mesh_dolfinx`.
    """
    if comm is None:
        comm = _MPI.COMM_SELF
    import gmsh  # type: ignore[import-untyped]

    from stellcoilbench.finite_build._gmsh import gmsh_context

    try:
        from dolfinx.io.gmsh import model_to_mesh as _model_to_mesh
    except ImportError:
        from dolfinx.io.gmshio import model_to_mesh as _model_to_mesh

    with gmsh_context():
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.open(str(msh_path))

        vol_entities = gmsh.model.getEntities(dim=3)
        if vol_entities:
            tags_3d = [e[1] for e in vol_entities]
            gmsh.model.addPhysicalGroup(3, tags_3d, tag=1)
            gmsh.model.setPhysicalName(3, 1, "Volume")

        surf_entities = gmsh.model.getEntities(dim=2)
        if surf_entities:
            tags_2d = [e[1] for e in surf_entities]
            gmsh.model.addPhysicalGroup(2, tags_2d, tag=1)
            gmsh.model.setPhysicalName(2, 1, "Surface")

        result = _model_to_mesh(
            gmsh.model,
            comm,
            rank=0,
            gdim=3,
        )

    if hasattr(result, "mesh"):
        return result.mesh, result.cell_tags, result.facet_tags
    return result


def _compute_body_force_dolfinx(
    coils: list[Coil],
    bs: BiotSavart,
    mesh: "dolfinx.mesh.Mesh",
    cross_section_area: float,
    *,
    width: float = 0.05,
    height: float = 0.05,
    use_regularized: bool = True,
    mesh_coils: Optional[list[Coil]] = None,
    all_coils: Optional[list[Coil]] = None,
    cached_coil_frames: Optional[list] = None,
    cached_Breg_list: Optional[list] = None,
    bs_mutual_list: Optional[list] = None,
    q_degree: int = 4,
) -> "dolfinx.fem.Function":
    """
    Compute J × B Lorentz body-force on a DOLFINx mesh.

    Evaluates the body force at quadrature points (not mesh nodes) so that
    the weak-form load integral converges under h-refinement.  Uses a
    quadrature element with degree 4 (raised from 2) to better capture the
    J×B variation across the winding-pack cross-section — in particular the
    self-field B0, which is zero at the centerline and varies as
    log/arctan across the cross-section width and is under-sampled at degree 2.

    The current density is assumed uniform across each coil cross-section:
    ``J = I / A`` (scalar magnitude, directed along the coil tangent at the
    nearest centerline point).  When ``mesh_coils`` and ``all_coils`` are
    provided, mutual B includes contributions from all coils (including
    symmetry copies); otherwise only unique coils are used.

    Parameters
    ----------
    coils : list
        simsopt ``Coil`` objects.
    bs : BiotSavart
        Magnetic field object (already configured with the same coils).
    mesh : dolfinx.mesh.Mesh
        Tetrahedral mesh of the winding pack.
    cross_section_area : float
        Winding-pack cross-section area [m²] (width × height).
    mesh_coils : list, optional
        Coils on which mesh points lie (e.g. unique coils).  When provided
        with ``all_coils``, ensures correct J assignment and mutual field
        from all coils including symmetry copies.
    all_coils : list, optional
        Full coil set including symmetry copies (e.g. bs.coils).  When
        provided with ``mesh_coils``, mutual B includes all coils.
    cached_coil_frames : list, optional
        Precomputed coil frames from :func:`_compute_coil_frame`. When
        provided with ``cached_Breg_list``, skips recomputation in J×B.
    cached_Breg_list : list, optional
        Precomputed regularized B at centerline points. Length must match
        ``mesh_coils`` when mesh_coils is used.
    bs_mutual_list : list, optional
        Prebuilt BiotSavart objects per unique coil (each excluding self).
        Avoids repeated BiotSavart construction in J×B.
    q_degree : int, optional
        Quadrature degree for integration (1=centroid, 4=default). Affects
        both the quadrature element and the weak-form measure. Default 4.

    Returns
    -------
    f_q : dolfinx.fem.Function
        Vector function (3-component) on a quadrature element representing
        the body-force density [N/m³] at quadrature points.  Has attribute
        ``_quadrature_degree`` (int) for use in the weak-form integration
        measure.
    """
    q_el = basix.ufl.quadrature_element(
        "tetrahedron",
        value_shape=(3,),
        degree=q_degree,
        scheme="default",
    )
    Q = dolfinx.fem.functionspace(mesh, q_el)
    f_q = dolfinx.fem.Function(Q, name="LorentzForce")

    q_coords = Q.tabulate_dof_coordinates()  # (n_total_qpts, 3)
    jcross_kwargs: dict[str, Any] = {
        "width": width,
        "height": height,
        "use_regularized": use_regularized,
    }
    if mesh_coils is not None:
        jcross_kwargs["mesh_coils"] = mesh_coils
    if all_coils is not None:
        jcross_kwargs["all_coils"] = all_coils
    if cached_coil_frames is not None:
        jcross_kwargs["cached_coil_frames"] = cached_coil_frames
    if cached_Breg_list is not None:
        jcross_kwargs["cached_Breg_list"] = cached_Breg_list
    if bs_mutual_list is not None:
        jcross_kwargs["bs_mutual_list"] = bs_mutual_list

    force = _compute_jcross_b(
        q_coords,
        coils,
        bs,
        cross_section_area,
        **jcross_kwargs,
    )

    f_q.x.array[:] = force.flatten()
    f_q.x.scatter_forward()
    f_q._quadrature_degree = q_degree  # type: ignore[attr-defined]
    return f_q


def _log_per_coil_bc_status(
    block_z_range: dict[int, tuple[float, float]],
    *,
    _last_logged: dict[str, int] | None = None,
) -> None:
    """Log per-coil BC block count; warn if single block (may indicate missing tags).

    Uses module-level cache to avoid spamming on repeated solves (e.g. convergence study).
    """
    n = len(block_z_range)
    tags = sorted(block_z_range.keys())
    cache = _log_per_coil_bc_status.__dict__.setdefault("_last_n", -1)
    if n == cache:
        return
    _log_per_coil_bc_status._last_n = n

    if n > 1:
        proc0_print(
            f"[structural] Per-coil spring BC: {n} blocks (tags {tags}); "
            "each coil has independent z-threshold."
        )
    else:
        proc0_print(
            f"[structural] Per-coil spring BC: 1 block (tag {tags[0]}). "
            "If this is a multi-coil mesh with displacement blow-up, ensure the .msh "
            "has gmsh:physical per coil (sweep fallback and ParaStell write these)."
        )


def _get_support_facet_indices(
    mesh: "dolfinx.mesh.Mesh",
    tdim: int,
    fdim: int,
) -> np.ndarray:
    """Return exterior facet indices in the spring/Dirichlet support region.

    Identifies boundary facets whose centroid z-coordinate falls within the
    bottom ``BC_Z_FRACTION`` of each coil's z-range (per-coil) or the global
    mesh z-range (fallback).  This is the geometry shared by both the
    spring-foundation BC path and the legacy hard-Dirichlet BC path.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Tetrahedral FEM mesh (may contain multiple disconnected coil bodies).
    tdim : int
        Topological dimension (3 for tets).
    fdim : int
        Facet dimension (2 for triangular boundary faces).

    Returns
    -------
    np.ndarray
        1-D int32 array of exterior facet indices to be used as support.
    """
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(fdim, 0)
    mesh.topology.create_connectivity(tdim, 0)

    f2c = mesh.topology.connectivity(fdim, tdim)
    f2v = mesh.topology.connectivity(fdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)
    coords = mesh.geometry.x
    cell_tags = getattr(mesh, "_structural_cell_tags", None)

    # Build per-block z_min, z_max when we have multiple blocks
    block_z_range: dict[int, tuple[float, float]] = {}
    tag_vals: np.ndarray = np.array([], dtype=np.int32)
    tag_idx: np.ndarray = np.array([], dtype=np.int32)
    if (
        cell_tags is not None
        and hasattr(cell_tags, "indices")
        and hasattr(cell_tags, "values")
    ):
        tag_vals = np.asarray(cell_tags.values)
        tag_idx = np.asarray(cell_tags.indices)
        unique_tags = np.unique(tag_vals)
        if len(unique_tags) > 1:
            tag_to_cells: dict[int, list[int]] = {t: [] for t in unique_tags}
            for idx, val in zip(tag_idx, tag_vals):
                tag_to_cells.setdefault(int(val), []).append(int(idx))
            for tag, cell_list in tag_to_cells.items():
                if not cell_list:
                    continue
                z_vals_c = []
                for c in cell_list:
                    verts = c2v.links(c)
                    z_vals_c.extend(coords[verts, 2])
                z_arr = np.array(z_vals_c)
                block_z_range[tag] = (float(np.min(z_arr)), float(np.max(z_arr)))

    exterior = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    facets_to_pin: list[int] = []

    if block_z_range:
        # Per-coil thresholds
        for fi in exterior:
            cells_adj = f2c.links(fi)
            if len(cells_adj) == 0:
                continue
            c = int(cells_adj[0])
            pos = np.where(tag_idx == c)[0]
            if len(pos) == 0:
                continue
            tag = int(tag_vals[pos[0]])
            if tag not in block_z_range:
                continue
            z_min_i, z_max_i = block_z_range[tag]
            threshold = _compute_z_threshold_for_fixed_support(
                z_min_i, z_max_i, range_if_zero=1.0
            )
            verts = f2v.links(fi)
            centroid_z = float(np.mean(coords[verts, 2]))
            if centroid_z <= threshold:
                facets_to_pin.append(int(fi))
    else:
        # Fallback: global z-threshold
        z_min = float(coords[:, 2].min())
        z_max = float(coords[:, 2].max())
        threshold = _compute_z_threshold_for_fixed_support(z_min, z_max)
        for fi in exterior:
            verts = f2v.links(fi)
            centroid_z = float(np.mean(coords[verts, 2]))
            if centroid_z <= threshold:
                facets_to_pin.append(int(fi))

    if not facets_to_pin:
        # Safety: pin at least the lowest facets if none selected
        cent_z_list = [
            (int(fi), float(np.mean(coords[f2v.links(fi), 2]))) for fi in exterior
        ]
        cent_z_list.sort(key=lambda x: x[1])
        n_pin = max(1, len(cent_z_list) // 10)
        facets_to_pin = [f for f, _ in cent_z_list[:n_pin]]

    return np.array(facets_to_pin, dtype=np.int32)


def _build_global_support_bcs(
    mesh: "dolfinx.mesh.Mesh",
    V: "dolfinx.fem.FunctionSpace",
    tdim: int,
    fdim: int,
) -> list:
    """Build Dirichlet BCs that pin the bottom BC_Z_FRACTION per coil.

    Uses per-coil z-thresholds: for each mesh block (coil), pins boundary
    facets whose centroid satisfies ``z <= z_min_i + BC_Z_FRACTION * (z_max_i - z_min_i)``
    where z_min_i and z_max_i are computed from that coil's vertex coordinates.
    This guarantees each coil has 15% of its own z-range fixed, preventing
    floating coils that would cause displacement blow-up.

    If cell tags are unavailable or all identical (single block), falls
    back to global z-threshold.

    .. note::
        This is the **legacy** hard-Dirichlet BC path.  The default solver
        path uses :func:`_get_support_facet_indices` with a spring-foundation
        Robin BC instead (see ``use_spring_bc`` in
        :func:`_solve_elasticity_dolfinx`), which eliminates the clamped-free
        stress singularity at the BC boundary edge.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Tetrahedral FEM mesh (may contain multiple disconnected coil
        bodies).
    V : dolfinx.fem.FunctionSpace
        CG-1 vector function space for displacements.
    tdim : int
        Topological dimension of the mesh (3 for tets).
    fdim : int
        Facet dimension (2 for triangular boundary faces).

    Returns
    -------
    list[dolfinx.fem.DirichletBC]
        A single-element list containing the assembled Dirichlet BC
        that clamps the support regions to zero displacement.
    """
    boundary_facets = _get_support_facet_indices(mesh, tdim, fdim)
    bc_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    zero = dolfinx.fem.Constant(
        mesh,
        np.zeros(3, dtype=dolfinx.default_scalar_type),
    )
    return [dolfinx.fem.dirichletbc(zero, bc_dofs, V)]


# Alias for backward compatibility (tests); implementation uses per-coil logic.
_build_per_coil_support_bcs = _build_global_support_bcs

# PETSc options for structural elasticity solve. Direct LU is used exclusively;
# benchmarking showed iterative solvers (CG/GMRES) do not beat LU at current mesh sizes.
_STRUCTURAL_PETSC_OPTS: dict[str, str] = {"ksp_type": "preonly", "pc_type": "lu"}


def _solve_elasticity_dolfinx(
    mesh: "dolfinx.mesh.Mesh",
    body_force: "dolfinx.fem.Function",
    E: float,
    nu: float,
    *,
    facet_tags: Optional["dolfinx.mesh.MeshTags"] = None,
    polynomial_degree: int = 1,
    use_spring_bc: bool = True,
) -> "dolfinx.fem.Function":
    """Solve the 3-D linear-elasticity problem on *mesh* with DOLFINx.

    Assembles and solves the weak form

        ∫ σ(u):ε(v) dx + ∫_{∂Ω} k_s(x) u·v ds  = ∫ f·v dx

    where σ = λ tr(ε) I + 2μ ε  (isotropic Hooke's law), *f* is
    the Lorentz body-force density J × B, and the second term is
    the tapered Winkler spring-foundation Robin BC (active when
    ``use_spring_bc=True``).  The stiffness k_s(x) uses a quadratic
    taper: k_s(z) = k_max * max(0, (z_thr - z)/depth)², integrated over
    all exterior facets; k_s is zero above z_threshold, giving a smooth
    C¹ transition that eliminates the Robin-to-Neumann singularity at
    the BC edge.  A direct LU solver (PETSc MUMPS/SuperLU) is used.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Tetrahedral mesh (may contain multiple disconnected coil
        bodies).
    body_force : dolfinx.fem.Function
        Body-force density [N/m³] on a quadrature element from
        :func:`_compute_body_force_dolfinx`.
    E : float
        Young's modulus [Pa].  Default for Cu/Nb₃Sn composite winding
        pack is ~100 GPa.
    nu : float
        Poisson ratio (default 0.3).
    facet_tags : dolfinx.mesh.MeshTags, optional
        Gmsh-derived facet markers.  Reserved for future user-defined BC
        regions; not used by the internal support strategies.
    use_spring_bc : bool, optional
        When ``True`` (default), applies a tapered Winkler spring-foundation
        Robin BC: k_s(z) = k_max * max(0, (z_thr - z)/depth)² over all
        exterior facets (k_s zeros out above z_threshold).  Uses per-coil
        z-thresholds when cell_tags exist (multi-coil mesh), preventing
        displacement blow-up in coils at higher z.  The smooth transition
        eliminates the clamped-free stress singularity.

        When ``False``, falls back to the legacy hard Dirichlet BC
        (``u = 0`` on support facets) via :func:`_build_global_support_bcs`.
        The legacy path produces a clamped-free stress singularity under
        mesh refinement and is kept only for comparison.

    Returns
    -------
    uh : dolfinx.fem.Function
        Displacement field (vector CG-1) in metres.
    """
    el = basix.ufl.element("Lagrange", "tetrahedron", polynomial_degree, shape=(3,))
    V = dolfinx.fem.functionspace(mesh, el)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    lam, mu = _lame_parameters(E, nu)

    def epsilon(w: "ufl.Argument") -> "ufl.core.expr.Expr":
        return ufl.sym(ufl.grad(w))

    def sigma(w: "ufl.Argument") -> "ufl.core.expr.Expr":
        return lam * ufl.nabla_div(w) * ufl.Identity(3) + 2 * mu * epsilon(w)

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    q_degree = getattr(body_force, "_quadrature_degree", 4)
    dx_q = ufl.Measure(
        "dx",
        domain=mesh,
        metadata={"quadrature_degree": q_degree, "quadrature_scheme": "default"},
    )
    L = ufl.inner(body_force, v) * dx_q

    tdim = mesh.topology.dim
    fdim = tdim - 1

    if use_spring_bc:
        # Spring-foundation Robin BC with spatially graded (tapered) k_s(z).
        # Smooth quadratic taper: k_s(z) = k_max * max(0, (z_thr - z)/depth)^2
        # integrated over ALL exterior facets — k_s itself zeros out above z_threshold.
        # The smooth C¹ transition at z_threshold eliminates the Robin→Neumann singularity
        # that caused mesh-dependent stress divergence with the uniform-k_s formulation.
        # Uses per-coil z-thresholds only; global fallback removed to avoid incorrect BCs on
        # coils at higher z. Requires mesh._structural_cell_tags from the pipeline.
        # Uses DoF coordinates and dofmap (not geometry/topology vertex indices) to avoid
        # DoF/geometry index mismatch that caused unphysical displacement blow-up.
        bcs: list = []
        CG1_s = dolfinx.fem.functionspace(
            mesh, basix.ufl.element("Lagrange", "tetrahedron", 1)
        )
        k_s_fn = dolfinx.fem.Function(CG1_s, name="SpringStiffness")
        dof_coords = CG1_s.tabulate_dof_coordinates()
        n_dofs = dof_coords.shape[0]
        mesh.topology.create_connectivity(tdim, 0)
        cell_tags = getattr(mesh, "_structural_cell_tags", None)
        block_z_range: dict[int, tuple[float, float]] = {}
        dof_to_block: np.ndarray = np.full(n_dofs, -1, dtype=np.int32)
        if (
            cell_tags is not None
            and hasattr(cell_tags, "indices")
            and hasattr(cell_tags, "values")
        ):
            tag_idx = np.asarray(cell_tags.indices)
            tag_vals = np.asarray(cell_tags.values)
            unique_tags = np.unique(tag_vals)
            if len(unique_tags) >= 1:
                tag_to_cells: dict[int, list[int]] = {int(t): [] for t in unique_tags}
                for idx, val in zip(tag_idx, tag_vals):
                    tag_to_cells.setdefault(int(val), []).append(int(idx))
                for tag, cell_list in tag_to_cells.items():
                    if not cell_list:
                        continue
                    # Use DoF coordinates (not geometry coords) for consistent indexing
                    z_vals_c = np.concatenate(
                        [dof_coords[CG1_s.dofmap.cell_dofs(c), 2] for c in cell_list]
                    )
                    block_z_range[tag] = (
                        float(np.min(z_vals_c)),
                        float(np.max(z_vals_c)),
                    )
                supp_depths = {
                    t: max(
                        BC_Z_FRACTION * (block_z_range[t][1] - block_z_range[t][0]),
                        0.01,
                    )
                    for t in block_z_range
                }
                z_thresholds = {
                    t: block_z_range[t][0] + supp_depths[t]
                    for t in block_z_range
                }
                for c in range(mesh.topology.index_map(tdim).size_local):
                    dofs = CG1_s.dofmap.cell_dofs(c)
                    t = int(tag_vals[np.where(tag_idx == c)[0][0]])
                    for d in dofs:
                        d = int(d)
                        blocks_of_d = dof_to_block[d]
                        if blocks_of_d == -1:
                            dof_to_block[d] = t
                        else:
                            if t != blocks_of_d:
                                best = min(
                                    (blocks_of_d, t),
                                    key=lambda b: z_thresholds.get(b, np.inf),
                                )
                                dof_to_block[d] = best
                for d in range(n_dofs):
                    t = dof_to_block[d]
                    if t >= 0 and t in block_z_range:
                        z_min_i, z_max_i = block_z_range[t]
                        depth = supp_depths[t]
                        z_thr = z_thresholds[t]
                        k_max_i = E / depth
                        z_d = dof_coords[d, 2]
                        taper = max(0.0, (z_thr - z_d) / depth)
                        k_s_fn.x.array[d] = k_max_i * taper**2
                    else:
                        raise ValueError(
                            "Per-coil spring BC requires mesh._structural_cell_tags with "
                            "block IDs; DoF has no block. Ensure mesh is loaded via "
                            "load_coil_mesh (sets cell_tags from gmsh:physical)."
                        )
                # Log per-coil BC status; warn if single block (may indicate missing gmsh:physical)
                _log_per_coil_bc_status(block_z_range)
                # Add spring-foundation Robin term to bilinear form (integrated over exterior facets)
                ds = ufl.Measure("ds", domain=mesh)
                a = a + k_s_fn * ufl.inner(u, v) * ds
        if not block_z_range:
            raise ValueError(
                "Per-coil spring BC requires mesh._structural_cell_tags with block IDs. "
                "No cell tags or block data found. Ensure mesh is loaded via load_coil_mesh "
                "and the .msh file has gmsh:physical (or per-block tet) so each coil has a "
                "unique tag."
            )
    else:
        # Legacy: hard Dirichlet BC (u=0) on support facets.
        # Produces a clamped-free stress singularity — kept for comparison only.
        bcs = _build_global_support_bcs(mesh, V, tdim, fdim)

    petsc_opts = _STRUCTURAL_PETSC_OPTS
    import inspect as _inspect

    _lp_sig = _inspect.signature(dolfinx.fem.petsc.LinearProblem.__init__)
    _kw: dict[str, Any] = {
        "bcs": bcs,
        "petsc_options": petsc_opts,
    }
    if "petsc_options_prefix" in _lp_sig.parameters:
        _kw["petsc_options_prefix"] = "structural_"
    problem = dolfinx.fem.petsc.LinearProblem(a, L, **_kw)
    uh = problem.solve()
    uh.name = "Displacement"
    return uh


def _compute_stress_dolfinx(
    mesh: "dolfinx.mesh.Mesh",
    u: "dolfinx.fem.Function",
    E: float,
    nu: float,
) -> tuple["dolfinx.fem.Function", "dolfinx.fem.Function"]:
    r"""Compute the Cauchy stress tensor and Von Mises stress from displacement.

    Cauchy stress (isotropic Hooke's law):

    .. math::
        \sigma = \lambda\mathrm{tr}(\varepsilon)I + 2\mu\varepsilon

    Von Mises equivalent stress:

    .. math::
        \sigma_{\mathrm{vm}} = \sqrt{\tfrac{3}{2}\,\mathbf{s}:\mathbf{s}}
        \quad\text{where}\quad
        \mathbf{s} = \sigma - \tfrac{1}{3}\mathrm{tr}(\sigma)I

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Tetrahedral mesh.
    u : dolfinx.fem.Function
        Displacement field (vector CG-1).
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson ratio.

    Returns
    -------
    tuple
        ``(sigma_field, vm_field)``. Both on DG-0 spaces (cell-wise constant).
    """
    lam, mu_val = _lame_parameters(E, nu)

    eps = ufl.sym(ufl.grad(u))
    sig = lam * ufl.nabla_div(u) * ufl.Identity(3) + 2 * mu_val * eps

    # Stress tensor → DG-0 tensor space
    T_el = basix.ufl.element("DG", "tetrahedron", 0, shape=(3, 3), symmetry=True)
    T_space = dolfinx.fem.functionspace(mesh, T_el)

    ip_t = T_space.element.interpolation_points
    if callable(ip_t):
        ip_t = ip_t()  # pre-0.10: method
    sigma_expr = dolfinx.fem.Expression(sig, ip_t)
    sigma_field = dolfinx.fem.Function(T_space, name="CauchyStress")
    sigma_field.interpolate(sigma_expr)

    # Von Mises stress: σ_vm = sqrt(3/2 s:s),  s = σ − (1/3)tr(σ)I
    s_dev = sig - (1.0 / 3.0) * ufl.tr(sig) * ufl.Identity(3)
    vm_expr_ufl = ufl.sqrt(1.5 * ufl.inner(s_dev, s_dev))

    S_el = basix.ufl.element("DG", "tetrahedron", 0)
    S_space = dolfinx.fem.functionspace(mesh, S_el)

    ip_s = S_space.element.interpolation_points
    if callable(ip_s):
        ip_s = ip_s()
    vm_expression = dolfinx.fem.Expression(vm_expr_ufl, ip_s)
    vm_field = dolfinx.fem.Function(S_space, name="VonMisesStress")
    vm_field.interpolate(vm_expression)

    return sigma_field, vm_field


def _interpolate_displacement_to_linear(
    mesh: "dolfinx.mesh.Mesh",
    u: "dolfinx.fem.Function",
) -> "dolfinx.fem.Function":
    """Interpolate displacement to CG-1 for export (XDMF/VTK require mesh-degree match).

    The mesh geometry is linear (degree 1). When u is in a higher-order space (p=2),
    interpolate to CG-1 so the function degree matches the mesh degree.
    """
    el_linear = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    V_linear = dolfinx.fem.functionspace(mesh, el_linear)
    u_linear = dolfinx.fem.Function(V_linear, name="Displacement")
    u_linear.interpolate(u)
    return u_linear


def _export_dolfinx(
    mesh: "dolfinx.mesh.Mesh",
    u: "dolfinx.fem.Function",
    vm_field: "dolfinx.fem.Function",
    output_dir: Path,
) -> dict[str, str]:
    """Write displacement and Von Mises stress to XDMF files.

    Writes ``displacement.xdmf`` and ``von_mises_stress.xdmf`` for ParaView
    or other XDMF-capable viewers. The mesh geometry is linear (degree 1);
    if the displacement is in a higher-order space (e.g. p=2), it is
    interpolated to CG-1 before export so the function degree matches the
    mesh degree as required by the XDMF writer.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Distributed mesh.
    u : dolfinx.fem.Function
        Displacement vector field (any Lagrange degree).
    vm_field : dolfinx.fem.Function
        Von Mises scalar stress field.
    output_dir : Path
        Directory for output files.

    Returns
    -------
    dict[str, str]
        Keys ``"displacement_xdmf"`` and ``"von_mises_xdmf"`` with file paths.
    """
    output_dir = _prepare_structural_output_dir(output_dir)
    paths: dict[str, str] = {}

    # XDMF requires function degree to match mesh degree (1 for linear mesh).
    # Interpolate p>1 displacement to CG-1; harmless when u is already p=1.
    u_export = _interpolate_displacement_to_linear(mesh, u)

    disp_path = output_dir / "displacement.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, str(disp_path), "w") as xf:
        xf.write_mesh(mesh)
        xf.write_function(u_export)
    paths["displacement_xdmf"] = str(disp_path)

    vm_path = output_dir / "von_mises_stress.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, str(vm_path), "w") as xf:
        xf.write_mesh(mesh)
        xf.write_function(vm_field)
    paths["von_mises_xdmf"] = str(vm_path)

    return paths
