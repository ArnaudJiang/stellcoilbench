"""I/O utilities for coil sensitivity analysis: VTK export and JSON serialization.

This module provides file-writing helpers for sensitivity analysis results:
- _save_sensitivity_results_json: Serializes SensitivityResult to JSON.
- export_perturbed_coils_vtk: Exports Monte-Carlo perturbed coil sets and
  plasma surfaces with B·n coloration to VTK for visual comparison with
  the nominal solution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.random import Generator, PCG64DXSM, SeedSequence

from ._sensitivity_samplers import _build_unit_samplers, _make_full_torus_surface

if TYPE_CHECKING:
    from simsopt.field import BiotSavart
    from simsopt.geo import SurfaceRZFourier

logger = logging.getLogger(__name__)


def _save_sensitivity_results_json(result: Any, output_path: Path) -> None:
    """Serialize a SensitivityResult to JSON and write to file.

    Writes the result of run_sensitivity_analysis to a JSON file for
    persistence and sharing. The output includes critical_sigma_m,
    bisection history, sweep data, and metadata.

    Parameters
    ----------
    result : SensitivityResult
        Result dataclass with a ``to_dict()`` method. Typically the
        return value of :func:`run_sensitivity_analysis`.
    output_path : Path
        Path for the output JSON file. Parent directories are created
        if they do not exist.

    Notes
    -----
    Called internally by :func:`run_sensitivity_analysis` to write
    ``sensitivity_results.json`` to the output directory.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)


def export_perturbed_coils_vtk(
    bs: BiotSavart,
    surface: SurfaceRZFourier,
    sigma: float,
    correlation_length_m: float,
    output_dir: Path,
    n_vtk_samples: int = 3,
    seed: int = 42,
) -> list[Path]:
    r"""Export perturbed coil sets and surface B·n to VTK.

    Uses perturbation amplitude :math:`\sigma` (typically :math:`\sigma^*`)
    to generate :math:`n` perturbed coil sets via the GP sampler.
    For each sample writes ``coils_perturbed_XX.vtu`` and
    ``surface_perturbed_XX.vts`` (full-torus plasma surface coloured by
    B.n from the perturbed field, matching ``surface_optimized.vts``).

    Parameters
    ----------
    bs : simsopt.field.BiotSavart
        Nominal BiotSavart field.
    surface : simsopt.geo.SurfaceRZFourier
        Plasma surface (may be half-period; a full-torus copy is built
        internally for VTK output).
    sigma : float
        Perturbation amplitude (metres) -- typically sigma*.
    correlation_length_m : float
        Correlation length (metres).
    output_dir : Path
        Directory to write VTK files into.
    n_vtk_samples : int
        Number of perturbed coil sets to export (default 3).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[Path]
        Paths to the written VTK files (coils + surfaces interleaved).
    """
    from simsopt.field import BiotSavart as BS, Coil, coils_to_vtk
    from simsopt.geo import CurvePerturbed, PerturbationSample

    from ..utils import suppress_output

    coils = bs.coils
    samplers = _build_unit_samplers(coils, correlation_length_m, n_derivs=1)

    s_plot = _make_full_torus_surface(surface)
    nphi = s_plot.quadpoints_phi.size
    ntheta = s_plot.quadpoints_theta.size

    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = SeedSequence(seed).spawn(n_vtk_samples)
    vtk_paths: list[Path] = []

    for i in range(n_vtk_samples):
        rg = Generator(PCG64DXSM(seeds[i]))
        perturbed_coils: list[Coil] = []
        for coil, sampler in zip(coils, samplers):
            pert = PerturbationSample(sampler, randomgen=rg)
            for k in range(len(pert._sample)):
                pert._sample[k] = pert._sample[k] * sigma
            perturbed_coils.append(Coil(CurvePerturbed(coil.curve, pert), coil.current))

        coil_stem = output_dir / f"coils_perturbed_{i:02d}"
        try:
            with suppress_output():
                coils_to_vtk(perturbed_coils, str(coil_stem), close=True)
            vtk_paths.append(Path(f"{coil_stem}.vtu"))
            logger.info("Wrote perturbed coil VTK: %s", coil_stem)
        except Exception as exc:
            logger.warning("Failed to write perturbed coil VTK %d: %s", i, exc)

        surf_stem = output_dir / f"surface_perturbed_{i:02d}"
        try:
            bs_pert = BS(perturbed_coils)
            bs_pert.set_points(s_plot.gamma().reshape((-1, 3)))
            B = bs_pert.B().reshape((nphi, ntheta, 3))
            normal = s_plot.unitnormal()
            BdotN = np.sum(B * normal, axis=2)[:, :, None]
            absB = bs_pert.AbsB().reshape((nphi, ntheta, 1))
            point_data = {
                "B_N/|B|": BdotN / absB,
                "B_N": BdotN,
                "modB": absB,
            }
            s_plot.to_vtk(str(surf_stem), extra_data=point_data)
            vtk_paths.append(Path(f"{surf_stem}.vts"))
            logger.info("Wrote perturbed surface VTK: %s", surf_stem)
        except Exception as exc:
            logger.warning("Failed to write perturbed surface VTK %d: %s", i, exc)

    return vtk_paths
