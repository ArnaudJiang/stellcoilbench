"""Config and case parsing helpers for coil optimization.

Extracted from optimization.py to reduce coupling: threshold keys,
post-processing param merging, helicity detection, coil kwarg filtering,
case path resolution, and optimization config preparation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..config_scheme import CaseConfig, PostProcessingConfig
from ..path_utils import (
    get_surface_filename,
    get_surface_search_base_dirs,
    get_target_B_from_surface,
    load_yaml,
    resolve_all,
    resolve_surface_path,
)
from ..path_utils import load_surface_with_range

from ._virtual_casing import _setup_virtual_casing

logger = logging.getLogger(__name__)

# Default coil objective terms for modular coils (case overrides)
DEFAULT_COIL_OBJECTIVE_TERMS = {
    "total_length": "l2_threshold",
    "coil_curvature": "lp_threshold",
    "coil_mean_squared_curvature": "l2_threshold",
    "linking_number": "",
    "coil_arclength_variation": "l2_threshold",
}

_THRESHOLD_KEYS = (
    "length_threshold",
    "length_threshold_device",
    "cc_threshold",
    "cs_threshold",
    "cc_threshold_device",
    "cs_threshold_device",
    "curvature_threshold",
    "curvature_threshold_device",
    "torsion_threshold",
    "torsion_threshold_device",
    "arclength_variation_threshold",
    "arclength_variation_threshold_device",
    "length_variance_threshold",
    "length_variance_threshold_device",
    "msc_threshold",
    "msc_threshold_device",
    "force_threshold",
    "force_threshold_device",
    "torque_threshold",
    "torque_threshold_device",
    "flux_threshold",
    "finite_build_width",
)


def _detect_helicity_from_case(case_yaml_path: Path | None) -> int:
    """Detect helicity_n from case YAML surface name.

    Returns -1 for quasi-helical (QH/QASH) surfaces, 0 otherwise.

    Parameters
    ----------
    case_yaml_path : Path | None
        Path to case YAML file. If None or non-existent, returns 0.

    Returns
    -------
    int
        Helicity parameter: -1 for QH, 0 for QA or unknown.
    """
    if case_yaml_path is None or not case_yaml_path.exists():
        return 0
    try:
        case_data = load_yaml(path=case_yaml_path)
        surface_name = get_surface_filename(case_data).lower()
        if "qh" in surface_name or "qash" in surface_name:
            return -1
    except (OSError, ValueError, KeyError) as exc:
        logger.debug("Failed to detect helicity from case YAML: %s", exc)
    return 0


def _merge_flag(
    name: str,
    cli_val: Any,
    pp_params: Dict[str, Any],
    default: Any,
    *,
    or_from_case: bool = True,
) -> Any:
    """Merge a single flag: use cli_val if non-default, else pp_params."""
    if or_from_case and cli_val == default:
        return pp_params.get(name, default)
    if not or_from_case and cli_val != default:
        return cli_val
    if or_from_case:
        return cli_val or pp_params.get(name, default)
    return pp_params.get(name, default)


def _merge_post_processing_params(
    pp_params: Dict[str, Any],
    cli_flags: PostProcessingConfig,
) -> PostProcessingConfig:
    """Merge post-processing settings from case.yaml with CLI flags.

    CLI flags take precedence when they differ from the default ``False``/``None``
    value; otherwise the case.yaml settings are used.

    Parameters
    ----------
    pp_params : dict
        ``post_processing_params`` section from case.yaml.
    cli_flags : PostProcessingConfig
        Flag values from CLI / caller.

    Returns
    -------
    PostProcessingConfig
        Merged post-processing configuration.
    """
    run_vmec = _merge_flag("run_vmec", cli_flags.run_vmec, pp_params, False)
    run_simple = _merge_flag("run_simple", cli_flags.run_simple, pp_params, False)
    plot_finite_build = _merge_flag(
        "plot_finite_build", cli_flags.plot_finite_build, pp_params, False
    )
    run_structural = _merge_flag(
        "run_structural", cli_flags.run_structural, pp_params, False
    )
    export_structural_full_coil_set = _merge_flag(
        "export_structural_full_coil_set",
        cli_flags.export_structural_full_coil_set,
        pp_params,
        False,
    )
    compute_shape_gradient = _merge_flag(
        "compute_shape_gradient",
        cli_flags.compute_shape_gradient,
        pp_params,
        False,
    )
    plot_poincare = (
        pp_params.get("plot_poincare", True)
        if cli_flags.plot_poincare
        else bool(pp_params.get("plot_poincare", False))
    )
    plot_boozer = pp_params.get("plot_boozer", True)
    finite_build_width = cli_flags.finite_build_width or pp_params.get(
        "finite_build_width"
    )
    finite_build_height = cli_flags.finite_build_height or pp_params.get(
        "finite_build_height"
    )
    structural_E = cli_flags.structural_E or pp_params.get("structural_E")
    structural_nu = cli_flags.structural_nu or pp_params.get("structural_nu")

    return PostProcessingConfig(
        run_vmec=run_vmec,
        run_simple=run_simple,
        plot_poincare=plot_poincare,
        plot_boozer=plot_boozer,
        nfieldlines=cli_flags.nfieldlines,
        plot_finite_build=plot_finite_build,
        finite_build_width=finite_build_width,
        finite_build_height=finite_build_height,
        run_structural=run_structural,
        export_structural_full_coil_set=export_structural_full_coil_set,
        structural_E=structural_E,
        structural_nu=structural_nu,
        compute_shape_gradient=compute_shape_gradient,
    )


def _extract_threshold_kwargs(
    coil_objective_terms: Dict[str, Any],
    case_yaml_path_abs: Optional[Path],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Separate threshold/weight kwargs from objective term options.

    Also reads ``dof_perturbation`` and ``random_seed`` from the raw YAML if
    present. These are not objective terms, but they must travel with the
    threshold kwargs because the coil initialization path consumes them.

    Parameters
    ----------
    coil_objective_terms : dict
        Combined objective terms dict (after merging defaults with case config).
    case_yaml_path_abs : Path or None
        Resolved absolute path to the case YAML file.

    Returns
    -------
    threshold_kwargs : dict
        Threshold and auxiliary kwargs to pass to optimize_coils_loop.
    filtered_terms : dict
        Objective terms with threshold keys removed.
    """
    threshold_kwargs: Dict[str, Any] = {}
    if coil_objective_terms:
        for key in _THRESHOLD_KEYS:
            if key in coil_objective_terms:
                threshold_kwargs[key] = coil_objective_terms[key]
        filtered_terms = {
            k: v for k, v in coil_objective_terms.items() if k not in _THRESHOLD_KEYS
        }
    else:
        filtered_terms = coil_objective_terms

    if case_yaml_path_abs and case_yaml_path_abs.exists():
        raw_config = load_yaml(path=case_yaml_path_abs)
        if isinstance(raw_config, dict) and "dof_perturbation" in raw_config:
            threshold_kwargs["dof_perturbation"] = raw_config["dof_perturbation"]
        if isinstance(raw_config, dict) and "random_seed" in raw_config:
            threshold_kwargs["random_seed"] = raw_config["random_seed"]

    return threshold_kwargs, filtered_terms


def _filter_coil_kwargs(
    coil_params: dict[str, Any],
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    """Return *coil_params* with the named keys removed.

    Parameters
    ----------
    coil_params : dict
        Raw coil params.
    exclude : set[str] or None
        Keys to drop.

    Returns
    -------
    dict[str, Any]
        Coil params with excluded keys removed.
    """
    if exclude is None:
        exclude = {"coil_type", "ncoils", "order", "target_B"}
    return {k: v for k, v in coil_params.items() if k not in exclude}


def _resolve_case_yaml_abs_path(case_path: Path) -> Path | None:
    """Resolve *case_path* to an absolute case YAML path.

    Uses path_utils.resolve_all with search dir derived from hint
    (file -> parent; directory -> dir itself).

    Parameters
    ----------
    case_path : Path
        Path to case directory or file.

    Returns
    -------
    Path | None
        Absolute path to case.yaml, or ``None`` if not found.
    """
    p = Path(case_path)
    search_dir = p.parent if p.is_file() else p
    resolved = resolve_all(search_dir, case_hint=case_path, surface_filename=None)
    return resolved.case_yaml


def _prepare_optimization_config(
    case_cfg: CaseConfig,
    case_path: Path,
    case_yaml_path_abs: Path | None,
    coils_out_path: Path,
    output_dir: Path | None,
    surface_resolution: int,
) -> dict[str, Any]:
    """Parse case config and resolve surface, output directory, virtual casing.

    Consolidates coil/optimizer/surface parameter extraction, surface file
    resolution & loading, output directory setup, target-B calculation, and
    virtual-casing setup.

    Parameters
    ----------
    case_cfg : CaseConfig
        Loaded case configuration.
    case_path : Path
        Path to the case directory or YAML file.
    case_yaml_path_abs : Path | None
        Resolved absolute path to case.yaml (may be ``None``).
    coils_out_path : Path
        Destination path for saved coils.
    output_dir : Path | None
        Explicit output directory, or ``None`` to derive from *coils_out_path*.
    surface_resolution : int
        Resolution (nphi = ntheta) for the plasma surface.

    Returns
    -------
    dict[str, Any]
        Keys: ``coil_params``, ``optimizer_params``, ``surface_params``,
        ``coil_objective_terms``, ``threshold_kwargs``, ``surface_file``,
        ``surface``, ``output_dir``, ``target_B``, ``vc_target``,
        ``vc_target_plot``.
    """
    coil_params = dict(case_cfg.coils_params)
    optimizer_params = dict(case_cfg.optimizer_params)
    surface_params = dict(case_cfg.surface_params)

    coil_objective_terms = dict(DEFAULT_COIL_OBJECTIVE_TERMS)
    if case_cfg.coil_objective_terms:
        coil_objective_terms.update(case_cfg.coil_objective_terms)

    threshold_kwargs, coil_objective_terms = _extract_threshold_kwargs(
        coil_objective_terms,
        case_yaml_path_abs,
    )

    # --- surface file resolution ---
    surface_file = surface_params["surface"]
    if not Path(surface_file).is_absolute():
        base_dirs = get_surface_search_base_dirs(case_path=case_path)
        resolved = resolve_surface_path(surface_file, base_dirs)
        if resolved is not None:
            surface_file = str(resolved)
        else:
            raise FileNotFoundError(
                f"Surface file not found: {surface_file}. Searched in: {base_dirs}"
            )

    surface = load_surface_with_range(
        surface_file,
        surface_range=surface_params.get("range", "half period"),
        nphi=surface_resolution,
        ntheta=surface_resolution,
    )

    # --- output directory ---
    if output_dir is None:
        output_dir = coils_out_path.parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(surface_file).is_absolute():
        surface_file = str(Path(surface_file).resolve())

    try:
        surface.filename = surface_file  # type: ignore[attr-defined]
        surface.range = surface_params.get("range", "half period")  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        logger.debug("Failed to set surface filename/range attributes: %s", exc)

    # --- target B and virtual casing ---
    target_B_override = surface_params.get("target_B")
    target_B = (
        float(target_B_override)
        if target_B_override is not None
        else get_target_B_from_surface(surface_file)
    )
    coil_params["target_B"] = target_B

    vc_target, vc_target_plot = _setup_virtual_casing(
        surface_file,
        surface_params,
        surface_resolution,
    )

    return {
        "coil_params": coil_params,
        "optimizer_params": optimizer_params,
        "surface_params": surface_params,
        "coil_objective_terms": coil_objective_terms,
        "threshold_kwargs": threshold_kwargs,
        "surface_file": surface_file,
        "surface": surface,
        "output_dir": output_dir,
        "target_B": target_B,
        "vc_target": vc_target,
        "vc_target_plot": vc_target_plot,
    }
