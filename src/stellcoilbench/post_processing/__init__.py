# ruff: noqa: E402, F401
"""
Post-processing utilities for coil optimization results.

This module provides functions to analyze optimized coil configurations:
- Poincaré plots (fieldline tracing)
- QFM surface computation
- VMEC equilibrium
- Quasisymmetry and iota profiles
- Boozer surface plots
- SIMPLE particle tracing
- VTK output and B·n error visualization

MPI-parallel for VMEC and fieldline tracing; plotting on rank 0 only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from .._mpl import MATPLOTLIB_AVAILABLE, ensure_mpl_agg, get_plt

if not MATPLOTLIB_AVAILABLE:
    raise ImportError("matplotlib is required for post_processing")
ensure_mpl_agg()  # Use non-interactive backend by default
plt = get_plt()
from mpl_toolkits.mplot3d import Axes3D
from simsopt.geo import SurfaceRZFourier
from simsopt.field import BiotSavart

if TYPE_CHECKING:
    from simsopt.mhd.vmec import Vmec
    from simsopt.util.mpi import MpiPartition
    from ..config_scheme import PostProcessingConfig

DEFAULT_NPOIPER2 = 256

from ._coil_io import (
    MagneticFieldSum,
    _get_coils_from_bfield as _get_coils_from_bfield,
    get_unique_coils as get_unique_coils,
    load_bfield_from_coils_json as load_bfield_from_coils_json,
    load_coils_and_surface,
    load_coils_from_json as load_coils_from_json,
)
from ..path_utils import load_surface_with_range as load_surface_with_range


from simsopt.mhd.vmec import Vmec  # type: ignore
from simsopt.mhd import QuasisymmetryRatioResidual  # type: ignore

# MPI imports - wrapped to handle systems without MPI (e.g., ReadTheDocs)
try:
    from simsopt.util.mpi import MpiPartition  # type: ignore
except (ImportError, RuntimeError):
    MpiPartition = None  # type: ignore
from ..mpi_utils import (
    comm_world,
    is_mpi_enabled,
    is_proc0,
    proc0_print,
    proc0_warning,
)

# Timing and output-suppression utilities now live in utils.py.
# Re-export here for backward compatibility.
from ..utils import (
    clear_timing_results,
    get_timing_results,
    print_timing_summary,
    suppress_output,
    timed_section,
)

from ._qfm import compute_qfm_surface
from ._vmec import (
    compute_quasisymmetry,
    run_vmec_equilibrium,
)
from ._boozer_plots import (
    plot_boozer_surface as plot_boozer_surface,
    plot_iota_profile as plot_iota_profile,
    plot_quasisymmetry_profile as plot_quasisymmetry_profile,
)
from ._results_io import _save_post_processing_results
from ._run_helpers import (
    _apply_post_processing_config,
    _run_post_processing_steps,
)


# --- Fieldline tracing (moved to _fieldlines.py) ---
from ._fieldlines import trace_fieldlines
from ._fieldlines import TRACING_AVAILABLE

# --- Shape gradient computation (moved to _shape_gradient.py) ---
from ._shape_gradient import (
    compute_shape_gradient_single_curve,
    compute_shape_gradients,
    shape_gradient_to_vtk_data,
)

# --- SIMPLE particle tracing (moved to _simple.py) ---
from ._simple import run_simple_particle_tracing


def _compute_bdotn_metrics(
    bfield: BiotSavart,
    surface: SurfaceRZFourier,
) -> Dict[str, float]:
    """Compute B·n error metrics on the plasma surface.

    Delegates to :func:`_bdotn.compute_bdotn_on_surface`.

    Parameters
    ----------
    bfield : BiotSavart | MagneticFieldSum
        Magnetic field object whose ``set_points`` / ``B`` / ``AbsB``
        methods will be called.
    surface : SurfaceRZFourier
        Plasma surface with ``gamma``, ``unitnormal``, ``quadpoints_phi``,
        and ``quadpoints_theta`` attributes.

    Returns
    -------
    Dict[str, float]
        ``{"BdotN": <value>, "BdotN_over_B": <value>}``
    """
    from ._bdotn import compute_bdotn_on_surface

    return compute_bdotn_on_surface(bfield, surface)


def run_post_processing(
    coils_json_path: Path,
    output_dir: Path,
    case_yaml_path: Optional[Path] = None,
    plasma_surfaces_dir: Optional[Path] = None,
    run_vmec: bool = False,
    helicity_m: int = 1,
    helicity_n: int = 0,
    ns: int = 50,
    plot_boozer: bool = True,
    plot_poincare: bool = True,
    nfieldlines: int = 20,
    mpi: MpiPartition | None = None,
    run_simple: bool = False,
    simple_executable_path: Optional[Path] = None,
    run_vmec_original: bool = False,
    plot_finite_build: bool = False,
    finite_build_width: Optional[float] = None,
    finite_build_height: Optional[float] = None,
    run_structural: bool = False,
    structural_E: Optional[float] = None,
    structural_nu: Optional[float] = None,
    compute_shape_gradient: bool = False,
    export_structural_full_coil_set: bool = False,
    *,
    config: "PostProcessingConfig | None" = None,
) -> Dict[str, Any]:
    """Run complete post-processing pipeline.

    This function:
    1. Loads coils and plasma surface
    2. Computes B·n on plasma surface
    3. Generates Poincaré plot (if requested)
    4. Optionally computes QFM surface and runs VMEC equilibrium (if run_vmec=True)
    5. Optionally runs SIMPLE fast particle tracing (if run_simple=True and VMEC succeeded)
    6. Optionally runs FEM structural analysis (if run_structural=True)
    7. Optionally computes per-coil shape gradients (if compute_shape_gradient=True)

    Parameters
    ----------
    coils_json_path : Path
        Path to coils JSON file.
    output_dir : Path
        Directory where output files will be saved.
    case_yaml_path : Path, optional
        Path to case.yaml file.
    plasma_surfaces_dir : Path, optional
        Directory containing plasma surface files.
    run_vmec : bool, default=False
        Whether to run QFM surface computation and VMEC equilibrium calculation.
        This is computationally expensive and disabled by default.
    helicity_m : int, default=1
        Poloidal mode number for quasisymmetry.
    helicity_n : int, default=0
        Toroidal mode number for quasisymmetry.
    ns : int, default=50
        Number of radial surfaces for quasisymmetry evaluation.
    plot_boozer : bool, default=True
        Whether to generate Boozer surface plot (requires run_vmec=True).
    plot_poincare : bool, default=True
        Whether to generate Poincaré plot.
    nfieldlines : int, default=20
        Number of fieldlines to trace for Poincaré plot.
    mpi : Any, optional
        MPI partition for parallel execution. If None, one is created based on ngroups.
    run_simple : bool, default=False
        Whether to run SIMPLE fast particle tracing after VMEC (requires simple.x executable
        and run_vmec=True). Disabled by default.
    simple_executable_path : Path, optional
        Path to simple.x executable. If None, searches in common locations.
    run_vmec_original : bool, default=False
        Whether to also run VMEC on the original plasma surface for comparison.
        This doubles the VMEC computation time but provides comparison plots.
    plot_finite_build : bool, default=False
        Whether to generate finite-build coil geometry (rectangular cross-section
        swept along centerline) and export to VTK.
    finite_build_width : float, optional
        Cross-section width [m] for finite-build coils. Default 35 cm at
        reactor scale, automatically scaled by a0 with a 5 cm lower bound.
    finite_build_height : float, optional
        Cross-section height [m] for finite-build coils. Defaults to
        ``finite_build_width`` when not specified.
    run_structural : bool, default=False
        Whether to run FEM structural (linear-elasticity) analysis on the
        finite-build coil mesh.  Requires DOLFINx or scikit-fem.
    structural_E : float, optional
        Young's modulus [Pa] for structural analysis.
    structural_nu : float, optional
        Poisson ratio for structural analysis.
    compute_shape_gradient : bool, default=False
        Whether to compute per-coil shape gradients and save them as
        extra point data in a coils VTK file.
    config : PostProcessingConfig, optional
        When provided, all post-processing options are read from this
        dataclass and the individual keyword arguments above are ignored
        (except *coils_json_path*, *output_dir*, *case_yaml_path*,
        *plasma_surfaces_dir*, and *mpi* which remain positional/standalone).

    Notes
    -----
    MPI parallelization is automatically used when available:
    - Fieldline tracing uses all available MPI processes via comm_world
    - VMEC uses all available MPI processes via MpiPartition(ngroups=1)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing post-processing results:
        - 'qfm_surface': QFM surface object
        - 'quasisymmetry_average': Average quasisymmetry error
        - 'quasisymmetry_profile': Radial quasisymmetry profile
        - 'vmec': VMEC equilibrium object (if run_vmec=True)
        - 'simple_results': Dictionary with SIMPLE results (if run_simple=True and VMEC succeeded)
    """
    opts = _apply_post_processing_config(
        config,
        run_vmec=run_vmec,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ns=ns,
        plot_boozer=plot_boozer,
        plot_poincare=plot_poincare,
        nfieldlines=nfieldlines,
        run_simple=run_simple,
        simple_executable_path=simple_executable_path,
        run_vmec_original=run_vmec_original,
        plot_finite_build=plot_finite_build,
        finite_build_width=finite_build_width,
        finite_build_height=finite_build_height,
        run_structural=run_structural,
        structural_E=structural_E,
        structural_nu=structural_nu,
        compute_shape_gradient=compute_shape_gradient,
        export_structural_full_coil_set=export_structural_full_coil_set,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clear_timing_results()

    if mpi is None and MpiPartition is not None:
        mpi = MpiPartition(ngroups=1)

    is_mpi_parallel = is_mpi_enabled()

    if is_mpi_parallel:
        proc0_print(f"Running with MPI: {comm_world.size} processes")  # type: ignore
        proc0_print("Note: Only VMEC and fieldline tracing use MPI parallelization")

    with timed_section("load_coils_and_surface"):
        bfield, surface = load_coils_and_surface(
            coils_json_path,
            case_yaml_path=case_yaml_path,
            plasma_surfaces_dir=plasma_surfaces_dir,
        )

    return _run_post_processing_steps(
        bfield,
        surface,
        output_dir,
        opts,
        case_yaml_path,
        coils_json_path,
        plasma_surfaces_dir,
        mpi,
        is_mpi_parallel,
    )
