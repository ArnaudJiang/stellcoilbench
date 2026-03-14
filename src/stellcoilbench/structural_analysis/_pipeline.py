"""
Structural analysis pipeline — mesh loading, elasticity solve, stress computation, VTK/XDMF export.

Contains the core FEM steps: load_coil_mesh, compute_lorentz_body_force,
solve_linear_elasticity, compute_stress_field, export_results, write_structural_vtk.
Extracted from __init__.py to keep the package root thin.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..constants import WP_POISSON_RATIO, WP_YOUNGS_MODULUS_PA
from .._optional_imports import optional_import
from ._common import _compute_jcross_b

if TYPE_CHECKING:
    import dolfinx  # noqa: F401
    import skfem  # noqa: F401
    from simsopt.field import BiotSavart, Coil

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
_last_import_error: Optional[str] = None

_dolfinx_mod = optional_import("dolfinx", "", fallback=None)
_DOLFINX_AVAILABLE: bool = _dolfinx_mod is not None
if not _DOLFINX_AVAILABLE:
    _last_import_error = "dolfinx not found"

if not _DOLFINX_AVAILABLE:
    import importlib.util as _ilu

    _skfem_mod = optional_import("skfem", "", fallback=None)
    _SKFEM_AVAILABLE: bool = (
        _skfem_mod is not None and _ilu.find_spec("meshio") is not None
    )
    if not _SKFEM_AVAILABLE and _last_import_error == "dolfinx not found":
        _last_import_error = (
            "skfem/meshio not found" if _skfem_mod is None else "meshio required"
        )
else:
    _SKFEM_AVAILABLE = False


def _require_backend() -> str:
    """Return the available structural analysis backend or raise ImportError.

    Used internally by pipeline functions to select DOLFINx (preferred) or
    scikit-fem. Raises a clear ImportError with installation hints when
    neither backend is available.

    Returns
    -------
    str
        ``"dolfinx"`` or ``"skfem"``.

    Raises
    ------
    ImportError
        When neither DOLFINx nor scikit-fem is importable.
    """
    if _DOLFINX_AVAILABLE:
        return "dolfinx"
    if _SKFEM_AVAILABLE:
        return "skfem"
    msg = (
        "Structural analysis requires DOLFINx (preferred) or scikit-fem. "
        "Install with:  conda install -c conda-forge fenics-dolfinx  or  "
        "pip install scikit-fem meshio"
    )
    if _last_import_error:
        msg += f"  (import error: {_last_import_error})"
    raise ImportError(msg)


def _resolve_backend(override: str | None) -> str:
    """Return backend, honoring optional override when that backend is available.

    Parameters
    ----------
    override : str | None
        If "dolfinx" or "skfem" and that backend is available, return it.
        Otherwise fall back to _require_backend().

    Returns
    -------
    str
        ``"dolfinx"`` or ``"skfem"``.
    """
    if override == "dolfinx" and _DOLFINX_AVAILABLE:
        return "dolfinx"
    if override == "skfem" and _SKFEM_AVAILABLE:
        return "skfem"
    return _require_backend()


DEFAULT_STRUCTURAL_MESH_RESOLUTION_M: float = 0.02
"""Default mesh resolution [m] for the final post-processing structural solve.

Used only when ``run_structural_analysis`` is invoked by post-processing
(submit-case / post-process). The structural objective in the optimization
loop uses ``structural_mesh_resolution_coarse`` and
``structural_mesh_resolution_fine`` from ``coil_objective_terms`` instead.
"""


def load_coil_mesh(
    msh_path: Path,
    *,
    backend: str | None = None,
) -> "dolfinx.mesh.Mesh | skfem.MeshTet1":
    """
    Load a Gmsh ``.msh`` coil mesh using the available backend.

    Parameters
    ----------
    msh_path : Path
        Path to the ``.msh`` file.
    backend : str | None
        Override: ``"dolfinx"`` or ``"skfem"`` when that backend is available.

    Returns
    -------
    object
        Backend-specific mesh object (``dolfinx.mesh.Mesh`` or
        ``skfem.MeshTet1``).
    """
    backend = _resolve_backend(backend)
    if backend == "dolfinx":
        from mpi4py import MPI

        from ..mpi_utils import is_mpi_enabled, comm_world
        from ._dolfinx import _load_mesh_dolfinx, _load_mesh_dolfinx_tet_only

        # Use COMM_SELF when MPI size > 1 to avoid deadlock: structural PP runs
        # only on rank 0, but COMM_WORLD collectives require all ranks.
        comm = MPI.COMM_SELF if (is_mpi_enabled() and comm_world.size > 1) else None
        try:
            mesh, cell_tags, facet_tags = _load_mesh_dolfinx_tet_only(
                msh_path, comm=comm
            )
        except (ImportError, ModuleNotFoundError):
            mesh, cell_tags, facet_tags = _load_mesh_dolfinx(msh_path, comm=comm)
        mesh._structural_cell_tags = cell_tags  # type: ignore[attr-defined]
        mesh._structural_facet_tags = facet_tags  # type: ignore[attr-defined]
        return mesh
    from ._skfem import _load_mesh_skfem

    return _load_mesh_skfem(msh_path)


def compute_lorentz_body_force(
    coils: list[Coil],
    bs: BiotSavart,
    mesh: "dolfinx.mesh.Mesh | skfem.MeshTet1",
    cross_section_area: float,
    *,
    width: float = 0.05,
    height: float = 0.05,
    use_regularized: bool = True,
    mesh_coils: list[Coil] | None = None,
    all_coils: list[Coil] | None = None,
    backend: str | None = None,
    quadrature_degree: int | None = None,
) -> "dolfinx.fem.Function | np.ndarray":
    r"""Compute Lorentz body-force density J × B on the coil mesh.

    The current density is assumed uniform across the winding-pack
    cross-section, directed along the coil tangent at the nearest
    centerline point:

    .. math::
        \mathbf{J} = \frac{I}{A}\,\mathbf{t}

    The body-force density (force per unit volume) is:

    .. math::
        \mathbf{f} = \mathbf{J} \times \mathbf{B} \quad [\mathrm{N/m^3}]

    When ``use_regularized=True`` (default), the Landreman et al. (2025)
    regularized internal field model is used for self-field; BiotSavart
    provides the mutual field from other coils.

    Parameters
    ----------
    coils : list[Coil]
        simsopt ``Coil`` objects.
    bs : BiotSavart
        Magnetic field evaluator.
    mesh : dolfinx.mesh.Mesh | skfem.MeshTet1
        Tetrahedral coil mesh.
    cross_section_area : float
        Winding-pack cross-section area [m²].
    width : float, optional
        Cross-section full width [m]. Default 0.05.
    height : float, optional
        Cross-section full height [m]. Default 0.05.
    use_regularized : bool, optional
        If True, use regularized internal field for self-field. Default True.
    mesh_coils : list[Coil] | None, optional
        Coils on which mesh points lie (for multi-coil meshes).
    all_coils : list[Coil] | None, optional
        Full coil set including symmetry copies for mutual-field B.

    Returns
    -------
    dolfinx.fem.Function | np.ndarray
        Body-force density [N/m³] at quadrature points (DOLFINx) or
        mesh nodes (scikit-fem).
    """
    backend = _resolve_backend(backend)
    if backend == "dolfinx":
        from ._dolfinx import _compute_body_force_dolfinx

        q_degree = quadrature_degree if quadrature_degree is not None else 4
        return _compute_body_force_dolfinx(
            coils,
            bs,
            mesh,
            cross_section_area,
            width=width,
            height=height,
            use_regularized=use_regularized,
            mesh_coils=mesh_coils,
            all_coils=all_coils,
            q_degree=q_degree,
        )

    coords = mesh.p.T  # (n_nodes, 3)
    return _compute_jcross_b(
        coords,
        coils,
        bs,
        cross_section_area,
        width=width,
        height=height,
        use_regularized=use_regularized,
        mesh_coils=mesh_coils,
        all_coils=all_coils,
    )


def solve_linear_elasticity(
    mesh: "dolfinx.mesh.Mesh | skfem.MeshTet1",
    body_force: "dolfinx.fem.Function | np.ndarray",
    E: float = WP_YOUNGS_MODULUS_PA,
    nu: float = WP_POISSON_RATIO,
    *,
    coils: Optional[list[Coil]] = None,
    bs: Optional[BiotSavart] = None,
    cross_section_area: Optional[float] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    mesh_coils: Optional[list[Coil]] = None,
    all_coils: Optional[list[Coil]] = None,
    backend: str | None = None,
    polynomial_degree: Optional[int] = None,
    use_spring_bc: bool = True,
) -> "dolfinx.fem.Function | np.ndarray":
    r"""Solve the 3-D linear-elasticity boundary-value problem.

    Solves the weak form for displacement :math:`\mathbf{u}`:

    .. math::
        \int_\Omega \sigma(\mathbf{u}) : \varepsilon(\mathbf{v})\,dx
        = \int_\Omega \mathbf{f} \cdot \mathbf{v}\,dx

    where :math:`\sigma = \lambda\mathrm{tr}(\varepsilon)I + 2\mu\varepsilon`
    (isotropic Hooke's law), :math:`\varepsilon = \tfrac{1}{2}(\nabla\mathbf{u}
    + \nabla\mathbf{u}^T)`, and :math:`\mathbf{f}` is the Lorentz body-force
    density. With ``use_spring_bc=True`` (default), a Winkler spring-foundation
    Robin BC is applied over the bottom 15% of the z-range, eliminating
    the clamped-free stress singularity.  Set ``use_spring_bc=False`` for
    the legacy hard-Dirichlet path (:math:`\mathbf{u}=0`).

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh | skfem.MeshTet1
        Tetrahedral mesh.
    body_force : dolfinx.fem.Function | np.ndarray
        Body-force density [N/m³] from :func:`compute_lorentz_body_force`.
    E : float, optional
        Young's modulus [Pa]. Default from ``WP_YOUNGS_MODULUS_PA``.
    nu : float, optional
        Poisson ratio. Default from ``WP_POISSON_RATIO``.
    coils, bs, cross_section_area, width, height : optional
        Passed to scikit-fem backend when body_force was precomputed elsewhere.
    mesh_coils : list[Coil] | None, optional
        Coils for mesh (multi-coil case).
    all_coils : list[Coil] | None, optional
        Full coil set for mutual B in scikit-fem.
    use_spring_bc : bool, optional
        When ``True`` (default), uses a Winkler spring-foundation Robin BC
        to avoid the clamped-free stress singularity.  ``False`` selects
        the legacy hard-Dirichlet BC for comparison.  Dolfinx backend only.

    Returns
    -------
    dolfinx.fem.Function | np.ndarray
        Displacement field [m] at mesh nodes.
    """
    backend = _resolve_backend(backend)
    if backend == "dolfinx":
        from ._dolfinx import _solve_elasticity_dolfinx

        facet_tags = getattr(mesh, "_structural_facet_tags", None)
        p_deg = 1 if polynomial_degree is None else polynomial_degree
        return _solve_elasticity_dolfinx(
            mesh,
            body_force,
            E,
            nu,
            facet_tags=facet_tags,
            polynomial_degree=p_deg,
            use_spring_bc=use_spring_bc,
        )
    from ._skfem import _solve_elasticity_skfem

    _w = 0.05
    _h = 0.05
    if width is not None:
        _w = width
    elif cross_section_area and cross_section_area > 0:
        _w = np.sqrt(cross_section_area)
    if height is not None:
        _h = height
    elif cross_section_area and cross_section_area > 0:
        _h = np.sqrt(cross_section_area)
    return _solve_elasticity_skfem(
        mesh,
        body_force,
        E,
        nu,
        coils=coils,
        bs=bs,
        cross_section_area=cross_section_area,
        width=_w,
        height=_h,
        mesh_coils=mesh_coils,
        all_coils=all_coils,
    )


def compute_stress_field(
    mesh: "dolfinx.mesh.Mesh | skfem.MeshTet1",
    displacement: "dolfinx.fem.Function | np.ndarray",
    E: float = WP_YOUNGS_MODULUS_PA,
    nu: float = WP_POISSON_RATIO,
    *,
    backend: str | None = None,
) -> tuple["dolfinx.fem.Function | None", "dolfinx.fem.Function | np.ndarray"]:
    r"""Compute the Cauchy stress tensor and Von Mises stress from displacement.

    The Cauchy stress follows isotropic Hooke's law:

    .. math::
        \sigma = \lambda\mathrm{tr}(\varepsilon)I + 2\mu\varepsilon

    with strain :math:`\varepsilon = \tfrac{1}{2}(\nabla\mathbf{u}
    + \nabla\mathbf{u}^T)`. The Von Mises equivalent stress is:

    .. math::
        \sigma_{\mathrm{vm}} = \sqrt{\tfrac{3}{2}\,\mathbf{s}:\mathbf{s}}
        \quad\text{where}\quad \mathbf{s} = \sigma - \tfrac{1}{3}\mathrm{tr}
        (\sigma)I

    (deviatoric stress :math:`\mathbf{s}`).

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh | skfem.MeshTet1
        Tetrahedral mesh.
    displacement : dolfinx.fem.Function | np.ndarray
        Displacement field from :func:`solve_linear_elasticity`.
    E : float, optional
        Young's modulus [Pa].
    nu : float, optional
        Poisson ratio.

    Returns
    -------
    tuple
        ``(sigma_field, von_mises)``. DOLFINx returns full stress tensor;
        scikit-fem returns ``(None, von_mises)``.
    """
    backend = _resolve_backend(backend)
    if backend == "dolfinx":
        from ._dolfinx import _compute_stress_dolfinx

        return _compute_stress_dolfinx(mesh, displacement, E, nu)
    from ._skfem import _compute_von_mises_skfem

    vm = _compute_von_mises_skfem(mesh, displacement, E, nu)
    return None, vm


def _symmetrize_structural_mesh_to_full_coil_set(
    points: np.ndarray,
    cells: np.ndarray,
    displacement: np.ndarray,
    von_mises: np.ndarray,
    nfp: int,
    stellsym: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply toroidal and stellarator symmetry to produce full coil set mesh.

    Transforms the unique-coil mesh into symmetry copies. No FEM solve required.
    Toroidal: rotate by 2π*k/nfp around z (k=1..nfp-1). Stellarator: (x,y,z)->(x,-y,-z).

    Parameters
    ----------
    points : np.ndarray
        Mesh vertices, shape (n_nodes, 3) [m].
    cells : np.ndarray
        Tetrahedron vertex indices, shape (n_cells, 4).
    displacement : np.ndarray
        Nodal displacement, shape (n_nodes, 3).
    von_mises : np.ndarray
        Per-element Von Mises stress, shape (n_cells,).
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether stellarator symmetry is used.

    Returns
    -------
    tuple
        (points_full, cells_full, displacement_full, von_mises_full).
    """
    symmetry_factor = nfp * (2 if stellsym else 1)
    if symmetry_factor <= 1:
        return points, cells, displacement, von_mises

    pt_list: list[np.ndarray] = []
    cell_list: list[np.ndarray] = []
    disp_list: list[np.ndarray] = []
    vm_list: list[np.ndarray] = []

    def _rot_z(theta: float, p: np.ndarray) -> np.ndarray:
        """Rotate points/vectors around z-axis by theta [rad]."""
        c, s = np.cos(theta), np.sin(theta)
        out = np.empty_like(p)
        out[:, 0] = p[:, 0] * c - p[:, 1] * s
        out[:, 1] = p[:, 0] * s + p[:, 1] * c
        out[:, 2] = p[:, 2]
        return out

    def _stell(p: np.ndarray) -> np.ndarray:
        """Stellarator symmetry: (x,y,z) -> (x,-y,-z)."""
        out = np.array(p, copy=True)
        out[:, 1] = -out[:, 1]
        out[:, 2] = -out[:, 2]
        return out

    n_nodes = points.shape[0]
    vertex_offset = 0

    for i in range(symmetry_factor):
        if i < nfp:
            theta = 2.0 * np.pi * i / nfp
            pts = _rot_z(theta, points)
            disp = _rot_z(theta, displacement)
        else:
            k = i - nfp
            theta = 2.0 * np.pi * k / nfp
            pts = _stell(_rot_z(theta, points))
            disp = _stell(_rot_z(theta, displacement))
        # Von Mises is scalar invariant
        pt_list.append(pts)
        disp_list.append(disp)
        vm_list.append(von_mises)
        cell_list.append(cells + vertex_offset)
        vertex_offset += n_nodes

    points_full = np.vstack(pt_list)
    cells_full = np.vstack(cell_list)
    displacement_full = np.vstack(disp_list)
    von_mises_full = np.concatenate(vm_list)
    return points_full, cells_full, displacement_full, von_mises_full


def _extract_structural_mesh_arrays(
    mesh: "dolfinx.mesh.Mesh | skfem.MeshTet1",
    displacement: "dolfinx.fem.Function | np.ndarray",
    von_mises: "dolfinx.fem.Function | np.ndarray",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract points, cells, displacement, von_mises as numpy arrays from mesh.

    Used for symmetrization and VTK export from raw arrays.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh | skfem.MeshTet1
        Tetrahedral mesh.
    displacement : dolfinx.fem.Function | np.ndarray
        Displacement field.
    von_mises : dolfinx.fem.Function | np.ndarray
        Von Mises stress per cell.

    Returns
    -------
    tuple
        (points, cells, u_array, vm_array).
    """
    if hasattr(displacement, "x"):
        if hasattr(displacement, "function_space") and hasattr(mesh, "geometry"):
            from ._dolfinx import _interpolate_displacement_to_linear

            u_linear = _interpolate_displacement_to_linear(mesh, displacement)
            u_array = np.asarray(u_linear.x.array.reshape(-1, 3), dtype=np.float64)
        else:
            u_array = np.asarray(displacement.x.array.reshape(-1, 3), dtype=np.float64)
    else:
        u_array = np.asarray(displacement, dtype=np.float64)

    if hasattr(von_mises, "x"):
        vm_array = np.asarray(von_mises.x.array, dtype=np.float64)
    else:
        vm_array = np.asarray(von_mises, dtype=np.float64)

    if hasattr(mesh, "geometry"):
        points = np.array(mesh.geometry.x, dtype=np.float64)
        mesh.topology.create_connectivity(mesh.topology.dim, 0)
        cell_to_vertex = mesh.topology.connectivity(mesh.topology.dim, 0)
        n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        cells = np.array(
            [cell_to_vertex.links(i) for i in range(n_cells)],
            dtype=np.int64,
        )
    else:
        points = np.array(mesh.p.T, dtype=np.float64)
        cells = np.array(mesh.t.T, dtype=np.int64)

    return points, cells, u_array, vm_array


def write_structural_vtk_arrays(
    points: np.ndarray,
    cells: np.ndarray,
    displacement: np.ndarray,
    von_mises: np.ndarray,
    output_path: Path,
) -> Path:
    """Write a structural VTK from raw numpy arrays (e.g. symmetrized full coil set).

    Parameters
    ----------
    points : np.ndarray
        Mesh vertices, shape (n_nodes, 3).
    cells : np.ndarray
        Tetrahedron vertex indices, shape (n_cells, 4).
    displacement : np.ndarray
        Nodal displacement, shape (n_nodes, 3).
    von_mises : np.ndarray
        Per-element Von Mises stress, shape (n_cells,).
    output_path : Path
        Output file path.

    Returns
    -------
    Path
        The path written.
    """
    import meshio as _meshio

    output_path = Path(output_path)
    u_mag = np.linalg.norm(displacement, axis=1)
    point_data = {"Displacement": displacement, "DisplacementMagnitude": u_mag}
    cell_data: dict[str, list[np.ndarray]] = {}
    if von_mises.shape[0] == cells.shape[0]:
        cell_data["VonMisesStress"] = [von_mises]
    out_mesh = _meshio.Mesh(
        points=points,
        cells=[("tetra", cells)],
        point_data=point_data,
        cell_data=cell_data,
    )
    _meshio.vtk.write(str(output_path), out_mesh)
    return output_path


def write_structural_vtk(
    mesh: "dolfinx.mesh.Mesh | skfem.MeshTet1",
    displacement: "dolfinx.fem.Function | np.ndarray",
    von_mises: "dolfinx.fem.Function | np.ndarray",
    output_path: Path,
) -> Path:
    """Write a standalone tetrahedral VTK with displacement and Von Mises stress.

    Produces a single VTK file with:
    - Point data: ``Displacement`` (3-vector), ``DisplacementMagnitude``
    - Cell data: ``VonMisesStress`` (scalar per tetrahedron)

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh | skfem.MeshTet1
        Tetrahedral mesh.
    displacement : dolfinx.fem.Function | np.ndarray
        Displacement field, shape ``(n_nodes, 3)``.
    von_mises : dolfinx.fem.Function | np.ndarray
        Von Mises stress per cell or per node.
    output_path : Path
        Output file path (e.g. ``structural_results.vtk``).

    Returns
    -------
    Path
        The path written.
    """
    import meshio as _meshio

    output_path = Path(output_path)

    if hasattr(displacement, "x"):
        # DOLFINx Function: interpolate to CG-1 if p>1 so point_data has one value per vertex
        if hasattr(displacement, "function_space") and hasattr(mesh, "geometry"):
            from ._dolfinx import _interpolate_displacement_to_linear

            u_linear = _interpolate_displacement_to_linear(mesh, displacement)
            u_array = u_linear.x.array.reshape(-1, 3)
        else:
            u_array = displacement.x.array.reshape(-1, 3)
    else:
        u_array = np.asarray(displacement)

    if hasattr(von_mises, "x"):
        vm_array = von_mises.x.array
    else:
        vm_array = np.asarray(von_mises)

    if hasattr(mesh, "geometry"):
        points = np.array(mesh.geometry.x, dtype=np.float64)
        mesh.topology.create_connectivity(mesh.topology.dim, 0)
        cell_to_vertex = mesh.topology.connectivity(mesh.topology.dim, 0)
        n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        cells = np.array(
            [cell_to_vertex.links(i) for i in range(n_cells)],
            dtype=np.int64,
        )
    else:
        points = np.array(mesh.p.T, dtype=np.float64)
        cells = np.array(mesh.t.T, dtype=np.int64)

    u_mag = np.linalg.norm(u_array, axis=1)

    point_data = {
        "Displacement": u_array,
        "DisplacementMagnitude": u_mag,
    }
    cell_data: dict[str, list[np.ndarray]] = {}
    if vm_array.shape[0] == cells.shape[0]:
        cell_data["VonMisesStress"] = [vm_array]

    out_mesh = _meshio.Mesh(
        points=points,
        cells=[("tetra", cells)],
        point_data=point_data,
        cell_data=cell_data,
    )
    _meshio.vtk.write(str(output_path), out_mesh)
    return output_path


def export_results(
    mesh: "dolfinx.mesh.Mesh | skfem.MeshTet1",
    displacement: "dolfinx.fem.Function | np.ndarray",
    von_mises: "dolfinx.fem.Function | np.ndarray",
    output_dir: Path,
    *,
    backend: str | None = None,
) -> dict[str, str]:
    """Write displacement and Von Mises stress to backend-specific output files.

    DOLFINx writes XDMF (``displacement.xdmf``, ``von_mises_stress.xdmf``).
    scikit-fem writes a single VTK via meshio.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh | skfem.MeshTet1
        Tetrahedral mesh.
    displacement : dolfinx.fem.Function | np.ndarray
        Displacement field.
    von_mises : dolfinx.fem.Function | np.ndarray
        Von Mises stress.
    output_dir : Path
        Directory for output files.
    backend : str | None
        Override: ``"dolfinx"`` or ``"skfem"`` when that backend is available.

    Returns
    -------
    dict[str, str]
        Mapping of result name to file path (e.g. ``"displacement_xdmf"``,
        ``"von_mises_xdmf"``, ``"structural_vtk"``).
    """
    backend = _resolve_backend(backend)
    if backend == "dolfinx":
        from ._dolfinx import _export_dolfinx

        return _export_dolfinx(mesh, displacement, von_mises, output_dir)
    from ._skfem import _export_skfem

    return _export_skfem(mesh, displacement, von_mises, output_dir)
