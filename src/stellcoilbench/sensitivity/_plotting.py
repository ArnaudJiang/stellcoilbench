"""
Matplotlib-based plotting for sensitivity analysis.

Generates f_B ratio vs sigma plots and saves them to PDF.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ._core import SensitivityResult

logger = logging.getLogger(__name__)


def plot_sensitivity(
    result: SensitivityResult,
    output_path: Path,
) -> Path:
    r"""Generate a sensitivity plot (:math:`f_B` ratio vs :math:`\sigma`) and save.

    Plots median, percentile (e.g. 95th), and mean of
    :math:`f_B^{\mathrm{pert}}/f_B^{\mathrm{nom}}` vs perturbation
    amplitude :math:`\sigma`, with :math:`\sigma^*` and the factor
    threshold marked.

    Parameters
    ----------
    result : SensitivityResult
        Completed sensitivity analysis result containing sweep data.
    output_path : Path
        Destination file (e.g. ``sensitivity_plot.pdf``).

    Returns
    -------
    Path
        The written file path.
    """
    from .._mpl import ensure_mpl_agg, get_plt

    ensure_mpl_agg()
    plt = get_plt()
    if plt is None:
        raise ImportError("matplotlib is required for save_sensitivity_plot")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    sigmas_mm = [s * 1e3 for s in result.sweep_sigmas]
    ax.plot(sigmas_mm, result.sweep_p50_ratios, "-o", label="Median", linewidth=1.5)
    ax.plot(
        sigmas_mm,
        result.sweep_p95_ratios,
        "-s",
        label=f"{result.percentile:.0f}th percentile",
        linewidth=1.5,
    )
    ax.plot(
        sigmas_mm,
        result.sweep_mean_ratios,
        "--^",
        label="Mean",
        linewidth=1.2,
        alpha=0.7,
    )

    ax.axhline(
        result.factor,
        color="red",
        linestyle=":",
        linewidth=1.0,
        label=f"Factor = {result.factor}",
    )
    ax.axvline(
        result.critical_sigma_m * 1e3,
        color="green",
        linestyle="--",
        linewidth=1.0,
        label=f"$\\sigma^*$ = {result.critical_sigma_m * 1e3:.2f} mm",
    )

    ax.set_xlabel("Perturbation $\\sigma$ (mm)")
    ax.set_ylabel("$f_B^{\\mathrm{pert}} \\,/\\, f_B^{\\mathrm{nom}}$")
    ax.set_title("Coil Sensitivity Analysis")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0.8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Sensitivity plot saved to %s", output_path)
    return output_path
