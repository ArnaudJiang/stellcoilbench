"""VMEC equilibrium analysis for post-processing.

Runs VMEC equilibrium reconstruction, quasisymmetry evaluation,
iota/QS profile plotting, and optional SIMPLE particle tracing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

from ..mpi_utils import is_proc0, proc0_print, proc0_try, proc0_warning
from ..utils import timed_section

if TYPE_CHECKING:
    from simsopt.field import BiotSavart
    from simsopt.geo import SurfaceRZFourier
    from simsopt.mhd.vmec import Vmec
    from simsopt.util.mpi import MpiPartition

try:
    from simsopt.util.mpi import MpiPartition as _MpiPartition
except (ImportError, RuntimeError):
    _MpiPartition = None  # type: ignore
from simsopt.mhd.vmec import Vmec  # type: ignore
from simsopt.mhd import QuasisymmetryRatioResidual  # type: ignore
from ..utils import suppress_output


def run_vmec_equilibrium(
    qfm_surface: "SurfaceRZFourier",
    vmec_input_path: Path | None = None,
    mpi: _MpiPartition | None = None,
    plasma_surfaces_dir: Path | None = None,
) -> Vmec | None:
    """Run VMEC to compute equilibrium from QFM surface.

    Parameters
    ----------
    qfm_surface : SurfaceRZFourier
        QFM surface to use as VMEC boundary.
    vmec_input_path : Path, optional
        Path to VMEC input file. If None or if the file is not a VMEC input file
        (e.g., .focus file), uses a reference VMEC input file as a template.
    mpi : Any, optional
        MPI partition for parallel execution. If None, creates a single-process partition.
    plasma_surfaces_dir : Path, optional
        Directory containing plasma surface files. Used to find a reference VMEC input file.

    Returns
    -------
    Vmec
        VMEC equilibrium object.
    """
    if mpi is None:
        if _MpiPartition is not None:
            mpi = _MpiPartition(ngroups=1)  # type: ignore

    is_vmec_input_file = False
    template_vmec_path = None

    if vmec_input_path is not None and vmec_input_path.exists():
        vmec_input_path_str = str(vmec_input_path).lower()
        is_vmec_input_file = (
            "input" in vmec_input_path_str or vmec_input_path_str.endswith(".input")
        )
        if is_vmec_input_file:
            template_vmec_path = vmec_input_path

    if not is_vmec_input_file:
        template_vmec_path = _find_vmec_template_input(
            vmec_input_path, plasma_surfaces_dir
        )

    equil = Vmec(str(template_vmec_path), mpi)
    equil.boundary = qfm_surface
    with suppress_output():
        equil.run()

    return equil


def compute_quasisymmetry(
    equil: Vmec,
    helicity_m: int = 1,
    helicity_n: int = 0,
    ns: int = 50,
) -> Tuple[float, np.ndarray]:
    """Compute quasisymmetry metrics from VMEC equilibrium.

    Parameters
    ----------
    equil : Vmec
        VMEC equilibrium object.
    helicity_m : int, default=1
        Poloidal mode number for quasisymmetry.
    helicity_n : int, default=0
        Toroidal mode number for quasisymmetry.
    ns : int, default=50
        Number of radial surfaces to evaluate.

    Returns
    -------
    Tuple[float, np.ndarray]
        Average quasisymmetry error and radial profile.
    """
    radii = np.arange(0, 1.01, 1.01 / ns)
    qs = QuasisymmetryRatioResidual(
        equil,
        radii,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
    )

    qs_profile = qs.profile()
    qs_average = float(np.mean(qs_profile))

    return qs_average, qs_profile


def _run_vmec_analysis(
    surface: SurfaceRZFourier,
    bfield: BiotSavart,
    output_dir: Path,
    case_yaml_path: Path | None,
    coils_json_path: Path,
    plasma_surfaces_dir: Path | None,
    mpi: MpiPartition | None,
    helicity_m: int,
    helicity_n: int,
    ns: int,
    plot_boozer: bool,
    run_simple: bool,
    simple_executable_path: Optional[Path],
    run_vmec_original: bool,
    is_mpi_parallel: bool,
    qfm_surface: SurfaceRZFourier | None = None,
) -> Dict[str, Any]:
    """Run VMEC equilibrium, QS analysis, and optional SIMPLE tracing.

    QFM surface should be pre-computed and passed via ``qfm_surface``.

    Parameters
    ----------
    surface : SurfaceRZFourier
        Plasma boundary surface.
    bfield : BiotSavart or MagneticFieldSum
        Magnetic field from coils.
    output_dir : Path
        Output directory for files.
    case_yaml_path : Path or None
        Path to case YAML.
    coils_json_path : Path
        Path to coils JSON file.
    plasma_surfaces_dir : Path or None
        Directory containing plasma surface files.
    mpi : MpiPartition or None
        MPI partition for VMEC.
    helicity_m, helicity_n : int
        Helicity parameters for quasisymmetry.
    ns : int
        Number of radial surfaces for QS evaluation.
    plot_boozer : bool
        Whether to generate Boozer surface plot.
    run_simple : bool
        Whether to run SIMPLE particle tracing.
    simple_executable_path : Path or None
        Path to simple.x executable.
    run_vmec_original : bool
        Whether to run VMEC on original surface for comparison.
    is_mpi_parallel : bool
        Whether MPI parallelism is active.
    qfm_surface : SurfaceRZFourier or None, optional
        Pre-computed QFM surface. If None, computed here.

    Returns
    -------
    Dict[str, Any]
        VMEC analysis results (QS average, profiles, loss fraction, etc.).
    """
    results: Dict[str, Any] = {}

    # Resolve VMEC input path
    vmec_input_path = _resolve_vmec_input_path(
        surface, case_yaml_path, coils_json_path, plasma_surfaces_dir
    )
    is_vmec_input = False
    if vmec_input_path is not None and vmec_input_path.exists():
        vmec_input_str = str(vmec_input_path).lower()
        is_vmec_input = "input" in vmec_input_str or vmec_input_str.endswith(".input")

    if not is_vmec_input:
        proc0_print("Note: Original surface file is not a VMEC input file.")
        proc0_print(
            "Using a template VMEC input file and replacing boundary with QFM surface."
        )

    # Optionally run VMEC for original surface (comparison)
    equil_original = None
    qs_profile_original = None
    radii_original = None
    if run_vmec_original:
        equil_original, qs_profile_original, radii_original = (
            _run_vmec_for_original_surface(
                surface,
                vmec_input_path,
                is_vmec_input,
                mpi,
                plasma_surfaces_dir,
                helicity_m,
                helicity_n,
                ns,
            )
        )

    # Run VMEC for QFM surface
    with timed_section("vmec_qfm_surface"):
        equil = run_vmec_equilibrium(
            qfm_surface,
            vmec_input_path=vmec_input_path if is_vmec_input else None,
            mpi=mpi,
            plasma_surfaces_dir=plasma_surfaces_dir,
        )
    results["vmec"] = equil

    if equil is None:
        proc0_warning("VMEC returned None — skipping post-VMEC analysis.")
        return results

    # Post-VMEC analysis on rank 0
    if is_proc0():
        _run_post_vmec_analysis(
            results,
            equil,
            output_dir,
            helicity_m,
            helicity_n,
            ns,
            plot_boozer,
            run_simple,
            simple_executable_path,
            equil_original,
            qs_profile_original,
            radii_original,
        )

    return results


def _find_vmec_template_input(
    vmec_input_path: Path | None,
    plasma_surfaces_dir: Path | None,
) -> Path:
    """Find a VMEC input file to use as template when the given path is not valid.

    Searches in vmec_input_path.parent (if provided), then plasma_surfaces_dir
    and standard search dirs. Looks for reference files first, then any file
    with "input" in the name.

    Parameters
    ----------
    vmec_input_path : Path or None
        Optional path (e.g. surface file); its parent is searched first.
    plasma_surfaces_dir : Path or None
        Plasma surfaces directory for get_surface_search_base_dirs.

    Returns
    -------
    Path
        Path to a VMEC input file.

    Raises
    ------
    ValueError
        If no VMEC input file can be found.
    """
    from ..path_utils import get_surface_search_base_dirs

    search_dirs: list[Path] = []
    if vmec_input_path is not None and vmec_input_path.parent.exists():
        search_dirs.append(vmec_input_path.parent)
    search_dirs.extend(
        get_surface_search_base_dirs(plasma_surfaces_dir=plasma_surfaces_dir)
    )

    reference_files = [
        "input.LandremanPaul2021_QA",
        "input.circular_tokamak",
        "input.HSX_QHS_mn1824_ns101",
        "input.cfqs_2b40",
    ]

    for search_dir in search_dirs:
        for ref_file in reference_files:
            potential_path = search_dir / ref_file
            if potential_path.exists():
                return potential_path

    for search_dir in search_dirs:
        if search_dir.exists():
            for file in search_dir.iterdir():
                if "input" in file.name.lower() and file.suffix == "":
                    return file

    raise ValueError(
        "Could not find a VMEC input file to use as template. "
        "VMEC requires an input file even when using a custom boundary surface."
    )


def _resolve_vmec_input_path(
    surface: SurfaceRZFourier,
    case_yaml_path: Path | None,
    coils_json_path: Path,
    plasma_surfaces_dir: Path | None,
) -> Path | None:
    """Resolve the VMEC input file path from surface or case YAML.

    Parameters
    ----------
    surface : SurfaceRZFourier
        Surface with optional filename attribute.
    case_yaml_path : Path or None
        Path to case YAML.
    coils_json_path : Path
        Path to coils JSON.
    plasma_surfaces_dir : Path or None
        Directory containing plasma surface files.

    Returns
    -------
    Path or None
        Resolved VMEC input path, or None if not found.
    """
    from ._surface_resolution import _resolve_surface_from_hints

    return _resolve_surface_from_hints(
        surface, case_yaml_path, plasma_surfaces_dir, coils_json_path
    )


def _run_vmec_for_original_surface(
    surface: SurfaceRZFourier,
    vmec_input_path: Path | None,
    is_vmec_input: bool,
    mpi: MpiPartition | None,
    plasma_surfaces_dir: Path | None,
    helicity_m: int,
    helicity_n: int,
    ns: int,
) -> Tuple[Vmec | None, np.ndarray | None, np.ndarray | None]:
    """Run VMEC on original surface for comparison.

    Returns
    -------
    tuple
        (equil_original, qs_profile_original, radii_original) or (None, None, None) on failure.
    """
    result: Tuple[Vmec | None, np.ndarray | None, np.ndarray | None] = (
        None,
        None,
        None,
    )
    with proc0_try(
        "Failed to compute original surface profiles: {e}",
        on_catch=lambda: proc0_print("Proceeding with QFM surface only."),
    ):
        proc0_print("Running VMEC for original plasma surface (for comparison)...")
        with timed_section("vmec_original_surface"):
            equil_original = run_vmec_equilibrium(
                surface,
                vmec_input_path=vmec_input_path if is_vmec_input else None,
                mpi=mpi,
                plasma_surfaces_dir=plasma_surfaces_dir,
            )

        qs_profile_original = None
        radii_original = None
        if is_proc0():
            with timed_section("quasisymmetry_original"):
                qs_average_original, qs_profile_original = compute_quasisymmetry(
                    equil_original,
                    helicity_m=helicity_m,
                    helicity_n=helicity_n,
                    ns=ns,
                )
            radii_original = np.arange(0, 1.01, 1.01 / ns)
            proc0_print(
                f"Original surface average quasisymmetry error: {qs_average_original:.2e}"
            )
        result = (equil_original, qs_profile_original, radii_original)
    return result


def _run_post_vmec_analysis(
    results: Dict[str, Any],
    equil: Vmec,
    output_dir: Path,
    helicity_m: int,
    helicity_n: int,
    ns: int,
    plot_boozer: bool,
    run_simple: bool,
    simple_executable_path: Path | None,
    equil_original: Vmec | None,
    qs_profile_original: Any,
    radii_original: Any,
) -> None:
    """Run post-VMEC analysis: QS computation, profile plots, Boozer, SIMPLE.

    Modifies ``results`` dict in-place. Runs only on rank 0.

    Parameters
    ----------
    results : Dict[str, Any]
        Results dict to update with QS metrics and SIMPLE results.
    equil : Vmec
        QFM-surface VMEC equilibrium.
    output_dir : Path
        Output directory for plots.
    helicity_m, helicity_n : int
        Helicity for QS computation.
    ns : int
        Number of radial surfaces.
    plot_boozer : bool
        Generate Boozer surface plot.
    run_simple : bool
        Run SIMPLE fast particle tracing.
    simple_executable_path : Path or None
        Path to simple.x.
    equil_original : Vmec or None
        Original-surface equilibrium for comparison plots.
    qs_profile_original, radii_original : array or None
        Original-surface QS profile and radii for overlay.
    """
    from ._boozer_plots import (
        plot_boozer_surface,
        plot_iota_profile,
        plot_quasisymmetry_profile,
    )
    from ._simple import run_simple_particle_tracing

    proc0_print("Computing quasisymmetry metrics...")
    with timed_section("quasisymmetry_qfm"):
        qs_average, qs_profile = compute_quasisymmetry(
            equil,
            helicity_m=helicity_m,
            helicity_n=helicity_n,
            ns=ns,
        )
    results["quasisymmetry_average"] = float(qs_average)
    results["quasisymmetry_profile"] = qs_profile.tolist()
    proc0_print(f"Average quasisymmetry error: {qs_average:.2e}")

    sign = -1 if helicity_n == -1 else 1
    proc0_print("Generating iota profile plot vs flux coordinate...")
    with timed_section("plot_iota_profile"):
        plot_iota_profile(
            equil,
            output_dir / "iota_profile.png",
            sign=sign,
            equil_original=equil_original,
            dpi=100,
        )

    proc0_print("Generating quasisymmetry profile plot vs flux coordinate...")
    radii = np.arange(0, 1.01, 1.01 / ns)
    with timed_section("plot_quasisymmetry_profile"):
        plot_quasisymmetry_profile(
            qs_profile,
            radii,
            output_dir / "quasisymmetry_profile.png",
            qs_profile_original=qs_profile_original,
            radii_original=radii_original,
            dpi=100,
        )

    if plot_boozer:
        proc0_print(
            "Generating Boozer surface plot (2x2 grid at s = 0, 0.25, 0.5, 1.0)..."
        )
        with timed_section("plot_boozer_surface_total"):
            plot_boozer_surface(equil, output_dir / "boozer_surface.png", dpi=100)

    if run_simple:
        with proc0_try(
            "SIMPLE particle tracing failed: {e}",
            on_catch=lambda: proc0_print("  Continuing without SIMPLE results."),
        ):
            proc0_print("Running SIMPLE fast particle tracing...")
            with timed_section("simple_particle_tracing"):
                simple_results = run_simple_particle_tracing(
                    equil,
                    output_dir,
                    simple_executable_path=simple_executable_path,
                )
            if simple_results:
                results["simple_results"] = simple_results
                if "loss_fraction" in simple_results:
                    results["loss_fraction"] = simple_results["loss_fraction"]
                    proc0_print(
                        f"  Particle loss fraction: {simple_results['loss_fraction']:.6e}"
                    )
