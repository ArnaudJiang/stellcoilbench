"""Post-optimization processing dispatch.

Resolves case YAML, locates coils JSON, and delegates to
``post_processing.run_post_processing`` after coil optimization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from ..constants import DEFAULT_VMEC_NS
from ..mpi_utils import proc0_print, proc0_try, proc0_warning
from ..path_utils import (
    coils_json_path_from_dir,
    find_plasma_surfaces_dir,
    resolve_all,
)

if TYPE_CHECKING:
    from simsopt.geo import SurfaceRZFourier
    from simsopt.util.mpi import MpiPartition

from ..config_scheme import PostProcessingConfig

logger = logging.getLogger(__name__)


def _run_post_processing_after_optimization(
    out_dir: Path,
    s: "SurfaceRZFourier",
    case_path: Path | str | None,
    pp_flags: PostProcessingConfig,
    *,
    mpi: "MpiPartition | None" = None,
    coils_json_path: Path | None = None,
) -> Dict[str, Any]:
    """Run post-processing (QFM, Poincaré, VMEC, quasisymmetry) after coil optimization.

    Resolves case.yaml path via *case_path*, *out_dir*, surface filename, and
    ``cases/`` directory search.  Runs ``run_post_processing`` if coils JSON
    exists.  Returns empty dict on failure or if coils not found.

    Parameters
    ----------
    out_dir : Path
        Output directory (contains biot_savart_optimized.json or coils.json).
    s : SurfaceRZFourier
        Plasma surface (for filename-based case.yaml search).
    case_path : Path | str | None
        Case path hint (file or directory).
    pp_flags : PostProcessingConfig
        Bundled post-processing flags (run_vmec, plot_poincare, nfieldlines, etc.).
    mpi : MpiPartition | None, optional
        MPI partition passed to ``run_post_processing``.  When *None* the call
        omits the ``mpi`` keyword.
    coils_json_path : Path | None, optional
        Explicit path to coils JSON.  When *None* the function probes for
        ``biot_savart_optimized.json`` / ``coils.json`` in *out_dir*.

    Returns
    -------
    Dict[str, Any]
        Post-processing results (quasisymmetry_average, loss_fraction, etc.) or {}.
    """
    import traceback

    result: Dict[str, Any] = {}

    def _on_post_processing_catch() -> None:
        traceback.print_exc()

    with proc0_try(
        "Post-processing failed: {e}",
        OSError,
        RuntimeError,
        ValueError,
        on_catch=_on_post_processing_catch,
    ):
        from ..post_processing import run_post_processing
        from ._config_parsing import _detect_helicity_from_case

        surface_filename = (
            str(s.filename) if hasattr(s, "filename") and s.filename else None
        )
        resolved = resolve_all(
            Path(out_dir),
            case_hint=case_path,
            surface_filename=surface_filename,
        )
        case_yaml_path = resolved.case_yaml

        if coils_json_path is None:
            coils_json_path = coils_json_path_from_dir(Path(out_dir))

        if coils_json_path is None or not coils_json_path.exists():
            proc0_warning(
                f"Skipping post-processing (coils_json not found: {coils_json_path})"
            )
            return {}

        proc0_print("\nRunning post-processing (QFM, Poincaré plots, profiles)...")

        helicity_n = _detect_helicity_from_case(case_yaml_path)

        plasma_surfaces_dir = find_plasma_surfaces_dir(Path(out_dir))

        pp_kwargs = pp_flags.to_run_post_processing_kwargs(
            helicity_n=helicity_n,
            ns=DEFAULT_VMEC_NS,
            coils_json_path=coils_json_path,
            output_dir=out_dir,
            case_yaml_path=case_yaml_path
            if (case_yaml_path is not None and case_yaml_path.exists())
            else None,
            plasma_surfaces_dir=plasma_surfaces_dir,
        )
        if mpi is not None:
            pp_kwargs["mpi"] = mpi

        post_processing_results = run_post_processing(**pp_kwargs)
        proc0_print("Post-processing complete!")
        if "quasisymmetry_average" in post_processing_results:
            proc0_print(
                f"  Average quasisymmetry error: {post_processing_results['quasisymmetry_average']:.2e}"
            )
        return post_processing_results

    return result
