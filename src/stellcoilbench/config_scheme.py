"""
Configuration dataclasses for StellCoilBench case definitions.

This module defines the schema for case YAML files and submission metadata,
providing typed structures for validation and runtime use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict

from .constants import ARIES_CS_B0  # noqa: F401 – used in REACTOR_REFERENCE
from .constants import ARIES_CS_MINOR_RADIUS  # noqa: F401 – re-exported
from .constants import WP_YOUNGS_MODULUS_PA  # noqa: F401 – re-exported
from .constants import WP_POISSON_RATIO  # noqa: F401 – re-exported

REACTOR_REFERENCE = {
    "B_field": ARIES_CS_B0,
    "description": "ARIES-CS reactor-scale reference",
}


class SurfaceParams(TypedDict, total=False):
    """Typed dictionary for surface parameter configuration."""

    surface: str
    range: str
    virtual_casing: bool


class CoilsParams(TypedDict, total=False):
    """Typed dictionary for coil parameter configuration."""

    coil_type: str
    ncoils: int
    order: int
    target_B: float
    coil_width: float
    vv_extension: float
    inboard_radius: float


class OptimizerParams(TypedDict, total=False):
    """Typed dictionary for optimizer parameter configuration."""

    algorithm: str
    max_iterations: int
    verbose: bool
    algorithm_options: dict
    max_iter_subopt: int


@dataclass
class CaseConfig:
    """
    Configuration for a single benchmark case, usually parsed from case.yaml.

    Attributes
    ----------
    description : str
        Human-readable description of the benchmark case.
    surface_params : SurfaceParams
        Plasma surface parameters (e.g., surface filename, range, virtual_casing).
    coils_params : CoilsParams
        Coil configuration (ncoils, order, coil_type, etc.).
    optimizer_params : OptimizerParams
        Optimizer settings (max_iterations, algorithm, tolerances).
    coil_objective_terms : dict[str, Any] | None
        Optional coil regularization terms (length, curvature, distances).
    fourier_continuation : dict[str, Any] | None
        Optional Fourier continuation orders for progressive refinement.
    post_processing_params : dict[str, Any] | None
        Optional post-processing options (VMEC, Poincaré, etc.).
    """

    description: str
    surface_params: SurfaceParams
    coils_params: CoilsParams
    optimizer_params: OptimizerParams
    scoring: dict[str, Any] | None = None
    coil_objective_terms: dict[str, Any] | None = None
    fourier_continuation: dict[str, Any] | None = None
    post_processing_params: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseConfig:
        """
        Construct a CaseConfig from a parsed YAML/JSON dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Raw configuration dict (e.g., from yaml.safe_load).

        Returns
        -------
        CaseConfig
            Validated configuration instance.
        """
        return cls(
            description=data.get("description", ""),
            surface_params=data.get("surface_params", {}),
            coils_params=data.get("coils_params", {}),
            optimizer_params=data.get("optimizer_params", {}),
            scoring=data.get("scoring"),
            coil_objective_terms=data.get("coil_objective_terms"),
            fourier_continuation=data.get("fourier_continuation"),
            post_processing_params=data.get("post_processing_params")
            or data.get("post_processing"),
        )


@dataclass
class SubmissionMetadata:
    """
    Descriptive information about a submission or method implementation.

    Attributes
    ----------
    method_version : str
        Version string for reproducibility.
    contact : str
        Contact identifier (e.g., GitHub username, email).
    hardware : str
        Hardware description (e.g., CPU model, GPU type).
    """

    method_version: str
    contact: str
    hardware: str


@dataclass
class PostProcessingConfig:
    """Configuration for the post-processing pipeline.

    Groups the parameters accepted by ``run_post_processing`` into a
    structured object for cleaner function signatures and easier
    forwarding from higher-level orchestration code.

    Attributes
    ----------
    run_vmec : bool
        Whether to run VMEC equilibrium reconstruction.
    helicity_m : int
        Poloidal helicity index for quasisymmetry evaluation.
    helicity_n : int
        Toroidal helicity index for quasisymmetry evaluation.
    ns : int
        Number of VMEC radial surfaces.
    plot_boozer : bool
        Whether to generate Boozer-surface plots.
    plot_poincare : bool
        Whether to generate Poincaré section plots.
    nfieldlines : int
        Number of field lines for Poincaré tracing.
    run_simple : bool
        Whether to run SIMPLE particle tracing.
    simple_executable_path : Optional[Path]
        Path to the SIMPLE executable (``None`` uses system default).
    run_vmec_original : bool
        Whether to run VMEC on the original (pre-optimization) coils.
    plot_finite_build : bool
        Whether to generate finite-build cross-section plots.
    finite_build_width : Optional[float]
        Winding-pack width for finite-build visualization (metres).
    finite_build_height : Optional[float]
        Winding-pack height for finite-build visualization (metres).
    run_structural : bool
        Whether to run FEM structural (linear-elasticity) analysis on finite-build
        coil geometry.  Requires DOLFINx and a tetrahedral mesh (Gmsh).
    export_structural_full_coil_set : bool
        When True and run_structural is True, also export structural_results_full.vtk
        with the full coil set (unique coils plus toroidal/stellarator symmetry copies).
    structural_E : Optional[float]
        Young's modulus [Pa] for the winding-pack material.  ``None`` uses the
        default from :obj:`constants.WP_YOUNGS_MODULUS_PA`.
    structural_nu : Optional[float]
        Poisson ratio for the winding-pack material.  ``None`` uses the
        default from :obj:`constants.WP_POISSON_RATIO`.
    compute_shape_gradient : bool
        Whether to compute per-coil shape gradients and save to VTK.
    """

    run_vmec: bool = False
    helicity_m: int = 1
    helicity_n: int = 0
    ns: int = 50
    plot_boozer: bool = True
    plot_poincare: bool = True
    nfieldlines: int = 20
    run_simple: bool = False
    simple_executable_path: Optional[Path] = None
    run_vmec_original: bool = False
    plot_finite_build: bool = False
    finite_build_width: Optional[float] = None
    finite_build_height: Optional[float] = None
    run_structural: bool = False
    export_structural_full_coil_set: bool = False
    structural_E: Optional[float] = None
    structural_nu: Optional[float] = None
    compute_shape_gradient: bool = False

    @classmethod
    def from_cli_options(cls, **kwargs: Any) -> "PostProcessingConfig":
        """Create from raw Typer CLI option values.

        Accepts keyword arguments matching PostProcessingConfig field names;
        unknown keys are ignored. Use after apply_all_post_processing_flags
        when building config from submit-case or post-process CLI.

        Parameters
        ----------
        **kwargs
            CLI option values (e.g. run_vmec, run_simple, plot_poincare,
            finite_build_width, structural_E). Only keys that match
            PostProcessingConfig fields are used.

        Returns
        -------
        PostProcessingConfig
            A configuration instance with provided values; defaults for omitted keys.
        """
        valid = {
            "run_vmec",
            "helicity_m",
            "helicity_n",
            "ns",
            "plot_boozer",
            "plot_poincare",
            "nfieldlines",
            "run_simple",
            "simple_executable_path",
            "run_vmec_original",
            "plot_finite_build",
            "finite_build_width",
            "finite_build_height",
            "run_structural",
            "export_structural_full_coil_set",
            "structural_E",
            "structural_nu",
            "compute_shape_gradient",
        }
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**filtered)

    @classmethod
    def from_case_config(
        cls, case_cfg: "CaseConfig", **overrides: Any
    ) -> "PostProcessingConfig":
        """Create from a CaseConfig's ``post_processing_params``, with optional overrides.

        Parameters
        ----------
        case_cfg : CaseConfig
            The case configuration whose ``post_processing_params`` dict
            supplies default values.
        **overrides
            Any keyword argument accepted by ``PostProcessingConfig``;
            these take precedence over values in ``post_processing_params``.

        Returns
        -------
        PostProcessingConfig
            A fully-resolved configuration instance.
        """
        pp_params = case_cfg.post_processing_params or {}
        kwargs: dict[str, Any] = {
            "run_vmec": pp_params.get("run_vmec", False),
            "run_simple": pp_params.get("run_simple", False),
            "plot_poincare": pp_params.get("plot_poincare", True),
            "plot_boozer": pp_params.get("plot_boozer", True),
            "plot_finite_build": pp_params.get("plot_finite_build", False),
            "finite_build_width": pp_params.get("finite_build_width"),
            "finite_build_height": pp_params.get("finite_build_height"),
            "run_structural": pp_params.get("run_structural", False),
            "export_structural_full_coil_set": pp_params.get(
                "export_structural_full_coil_set", False
            ),
            "structural_E": pp_params.get("structural_E"),
            "structural_nu": pp_params.get("structural_nu"),
            "nfieldlines": pp_params.get("nfieldlines", 20),
            "compute_shape_gradient": pp_params.get("compute_shape_gradient", False),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def to_run_post_processing_kwargs(
        self,
        *,
        helicity_n: int | None = None,
        ns: int | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Build kwargs dict for run_post_processing.

        Parameters
        ----------
        helicity_n : int | None
            Override helicity_n (e.g. from case surface detection). Uses
            self.helicity_n if None.
        ns : int | None
            Override VMEC radial surfaces. Uses self.ns if None.
        **extra : Any
            Additional kwargs (coils_json_path, output_dir, case_yaml_path,
            plasma_surfaces_dir, mpi) to merge into the result.

        Returns
        -------
        dict[str, Any]
            Kwargs suitable for run_post_processing.
        """
        out: dict[str, Any] = {
            "run_vmec": self.run_vmec,
            "helicity_m": self.helicity_m,
            "helicity_n": self.helicity_n if helicity_n is None else helicity_n,
            "ns": self.ns if ns is None else ns,
            "plot_boozer": self.plot_boozer,
            "plot_poincare": self.plot_poincare,
            "nfieldlines": self.nfieldlines,
            "run_simple": self.run_simple,
            "plot_finite_build": self.plot_finite_build,
            "finite_build_width": self.finite_build_width,
            "finite_build_height": self.finite_build_height,
            "run_structural": self.run_structural,
            "export_structural_full_coil_set": self.export_structural_full_coil_set,
            "structural_E": self.structural_E,
            "structural_nu": self.structural_nu,
            "compute_shape_gradient": self.compute_shape_gradient,
        }
        out.update(extra)
        return out
