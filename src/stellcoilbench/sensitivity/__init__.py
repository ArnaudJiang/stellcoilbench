"""
Coil sensitivity analysis via stochastic perturbation.

Uses simsopt's CurvePerturbed / GaussianSampler to quantify how robust
optimized coil solutions are to geometric perturbations (e.g. manufacturing
tolerances, positioning errors).  The principal output is **sigma*** -- the
perturbation amplitude (standard deviation in metres) at which the squared-flux
metric f_B degrades by no more than a user-chosen factor (default 2x) at the
chosen percentile (default 95th).

Typical usage
-------------
>>> from stellcoilbench.sensitivity import run_sensitivity_analysis
>>> results = run_sensitivity_analysis(
...     coils_json_path=Path("coils.json"),
...     case_yaml_path=Path("case.yaml"),
... )
>>> print(f"Critical sigma*: {results['critical_sigma_m'] * 1e3:.2f} mm")
"""

from __future__ import annotations

from ._core import (
    BisectionStep,
    SensitivityResult,
    compute_fb_perturbed,
    find_critical_sigma,
    run_sensitivity_analysis,
)
from ._sensitivity_io import export_perturbed_coils_vtk
from ._sensitivity_samplers import (
    _build_unit_samplers,
    _coil_arc_length,
    _make_full_torus_surface,
    _repair_sampler_L,
)
from ._plotting import plot_sensitivity

__all__ = [
    "BisectionStep",
    "SensitivityResult",
    "run_sensitivity_analysis",
    "compute_fb_perturbed",
    "find_critical_sigma",
    "plot_sensitivity",
    "export_perturbed_coils_vtk",
    "_coil_arc_length",
    "_make_full_torus_surface",
    "_repair_sampler_L",
    "_build_unit_samplers",
]
