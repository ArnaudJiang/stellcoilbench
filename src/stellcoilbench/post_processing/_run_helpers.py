"""Helper functions for run_post_processing pipeline decomposition."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from ..mpi_utils import is_proc0, proc0_print, proc0_try
from ..utils import print_timing_summary

if TYPE_CHECKING:
    from ..config_scheme import PostProcessingConfig


def _apply_post_processing_config(
    config: "PostProcessingConfig | None",
    **defaults: Any,
) -> Dict[str, Any]:
    """Apply PostProcessingConfig overrides to defaults. Returns effective kwargs dict."""
    if config is None:
        return dict(defaults)
    return {
        **defaults,
        "run_vmec": config.run_vmec,
        "helicity_m": config.helicity_m,
        "helicity_n": config.helicity_n,
        "ns": config.ns,
        "plot_boozer": config.plot_boozer,
        "plot_poincare": config.plot_poincare,
        "nfieldlines": config.nfieldlines,
        "run_simple": config.run_simple,
        "simple_executable_path": config.simple_executable_path,
        "run_vmec_original": config.run_vmec_original,
        "plot_finite_build": config.plot_finite_build,
        "finite_build_width": config.finite_build_width,
        "finite_build_height": config.finite_build_height,
        "run_structural": config.run_structural,
        "export_structural_full_coil_set": config.export_structural_full_coil_set,
        "structural_E": config.structural_E,
        "structural_nu": config.structural_nu,
        "compute_shape_gradient": config.compute_shape_gradient,
    }


def _run_vmec_pipeline(
    surface: Any,
    bfield: Any,
    output_dir: Path,
    case_yaml_path: Path | None,
    coils_json_path: Path,
    plasma_surfaces_dir: Path | None,
    mpi: Any,
    helicity_m: int,
    helicity_n: int,
    ns: int,
    plot_boozer: bool,
    run_simple: bool,
    simple_executable_path: Path | None,
    run_vmec_original: bool,
    is_mpi_parallel: bool,
) -> Dict[str, Any]:
    """Run QFM + VMEC equilibrium + post-analysis. Returns dict to merge into results."""
    from ._qfm import compute_qfm_surface
    from ._vmec import _run_vmec_analysis
    from ..utils import timed_section

    results: Dict[str, Any] = {}
    qfm_surface = None

    if is_proc0():
        proc0_print("Computing QFM surface...")
        with timed_section("compute_qfm_surface"):
            qfm_surface = compute_qfm_surface(surface, bfield)
        results["qfm_surface"] = qfm_surface

        proc0_print("Saving QFM surface as VTK file...")
        with timed_section("save_qfm_vtk"):
            qfm_vtk_path = output_dir / "qfm_surface"
            with proc0_try("Failed to save QFM surface as VTK: {e}"):
                qfm_surface.to_vtk(str(qfm_vtk_path))
                proc0_print(f"Saved QFM surface to {qfm_vtk_path}.vts")
                results["qfm_vtk_path"] = str(qfm_vtk_path)

    if is_mpi_parallel:
        from ..mpi_utils import comm_world

        qfm_temp_path = output_dir / "_qfm_surface_temp.json"
        if is_proc0():
            from simsopt._core import save as _simsopt_save

            _simsopt_save(qfm_surface, str(qfm_temp_path))
        comm_world.Barrier()  # type: ignore
        if not is_proc0():
            from simsopt._core import load as _simsopt_load

            qfm_surface = _simsopt_load(str(qfm_temp_path))

    with proc0_try(
        "VMEC calculation failed: {e}",
        OSError,
        RuntimeError,
        ValueError,
        ImportError,
        TypeError,
        on_catch=lambda: proc0_print("Skipping VMEC-dependent post-processing."),
    ):
        vmec_results = _run_vmec_analysis(
            surface,
            bfield,
            output_dir,
            case_yaml_path,
            coils_json_path,
            plasma_surfaces_dir,
            mpi,
            helicity_m,
            helicity_n,
            ns,
            plot_boozer,
            run_simple,
            simple_executable_path,
            run_vmec_original,
            is_mpi_parallel,
            qfm_surface=qfm_surface,
        )
        results.update(vmec_results)

    return results


def _run_post_processing_steps(
    bfield: Any,
    surface: Any,
    output_dir: Path,
    opts: Dict[str, Any],
    case_yaml_path: Path | None,
    coils_json_path: Path,
    plasma_surfaces_dir: Path | None,
    mpi: Any,
    is_mpi_parallel: bool,
) -> Dict[str, Any]:
    """Execute the post-processing pipeline steps (B·n, Poincaré, VMEC, etc.).

    Called by run_post_processing after loading coils and surface. Merges
    results from each step into a single dict.

    Parameters
    ----------
    bfield, surface : Any
        Loaded magnetic field and plasma surface.
    output_dir : Path
        Output directory.
    opts : dict
        Effective options from _apply_post_processing_config.
    case_yaml_path, coils_json_path, plasma_surfaces_dir : Path or None
        Path hints for surface resolution.
    mpi : Any
        MPI partition.
    is_mpi_parallel : bool
        Whether MPI is enabled.

    Returns
    -------
    dict
        Accumulated results from all steps.
    """
    from ..utils import timed_section
    from ._bdotn import compute_bdotn_on_surface
    from ._poincare import _run_poincare_analysis
    from ._results_io import _save_post_processing_results
    from ._runners import run_optional_steps
    from ..mpi_utils import comm_world

    results: Dict[str, Any] = {}

    if opts.get("plot_poincare"):
        poincare_out = _run_poincare_analysis(
            bfield,
            surface,
            output_dir,
            case_yaml_path,
            plasma_surfaces_dir,
            opts.get("nfieldlines", 20),
            is_mpi_parallel,
        )
        results.update(poincare_out)

    BdotN = 0.0
    BdotN_over_B = 0.0
    if is_proc0():
        with timed_section("compute_BdotN"):
            bdotn_metrics = compute_bdotn_on_surface(bfield, surface)
            BdotN = bdotn_metrics["BdotN"]
            BdotN_over_B = bdotn_metrics["BdotN_over_B"]
    results["BdotN"] = BdotN
    results["BdotN_over_B"] = BdotN_over_B

    run_optional_steps(
        bfield,
        surface,
        output_dir,
        opts,
        results,
        is_proc0=is_proc0(),
    )

    if is_mpi_parallel:
        BdotN = comm_world.bcast(BdotN, root=0)  # type: ignore
        BdotN_over_B = comm_world.bcast(BdotN_over_B, root=0)  # type: ignore

    if opts.get("run_vmec"):
        vmec_results = _run_vmec_pipeline(
            surface,
            bfield,
            output_dir,
            case_yaml_path,
            coils_json_path,
            plasma_surfaces_dir,
            mpi,
            opts.get("helicity_m", 1),
            opts.get("helicity_n", 0),
            opts.get("ns", 50),
            opts.get("plot_boozer", True),
            opts.get("run_simple", False),
            opts.get("simple_executable_path"),
            opts.get("run_vmec_original", False),
            is_mpi_parallel,
        )
        results.update(vmec_results)

    if is_proc0():
        _save_post_processing_results(results, output_dir)
        _print_post_processing_summary(results)

    return results


def _print_post_processing_summary(results: Dict[str, Any]) -> None:
    """Print a combined summary of post-processing metrics and timing.

    Displays key scalar metrics (B-dot-n, quasisymmetry, loss fraction,
    structural) followed by a timing breakdown of all pipeline steps.

    Parameters
    ----------
    results : Dict[str, Any]
        Accumulated results from the post-processing pipeline.
    """
    proc0_print("\n" + "=" * 60)
    proc0_print("POST-PROCESSING RESULTS")
    proc0_print("=" * 60)

    bdotn = results.get("BdotN")
    bdotn_b = results.get("BdotN_over_B")
    if bdotn is not None:
        proc0_print(f"  {'B·n (avg)':32s} {bdotn:.4e}")
    if bdotn_b is not None:
        proc0_print(f"  {'B·n / |B| (avg)':32s} {bdotn_b:.4e}")

    qs = results.get("quasisymmetry_average")
    if qs is not None:
        proc0_print(f"  {'Quasisymmetry error (avg)':32s} {qs:.4e}")

    simple_res = results.get("simple_results", {})
    lf = simple_res.get("loss_fraction")
    if lf is not None:
        proc0_print(f"  {'Particle loss fraction':32s} {lf:.4f}")

    sm = results.get("structural_metrics", {})
    if sm.get("skipped"):
        proc0_print(
            f"  {'Structural analysis':32s} skipped ({sm.get('reason', 'no mesh')})"
        )
    else:
        vm = sm.get("max_von_mises_stress_Pa")
        if vm is not None:
            proc0_print(f"  {'Max Von Mises stress':32s} {vm:.4e} Pa")
        md = sm.get("max_displacement_m")
        if md is not None:
            proc0_print(f"  {'Max displacement':32s} {md:.4e} m")

    print_timing_summary()
