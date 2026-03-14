"""SIMPLE output parsing and parameter extraction.

Parses SIMPLE output files (confined_fraction.dat, times_lost.dat), extracts
parameters from kwargs, and generates loss fraction plots.
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..mpi_utils import proc0_print, proc0_try

from ._simple_input import DEFAULT_NPOIPER2


def _extract_simple_params(
    kwargs: Dict[str, Any],
    vmec_output_file: Path,
) -> Tuple[Dict[str, Any], set, str]:
    """Extract SIMPLE configuration parameters from kwargs with defaults.

    Pops all recognized SIMPLE parameters from *kwargs* (so that only
    unknown keys remain afterwards) and returns them as a flat dict together
    with the set of keys that were explicitly provided by the caller and
    the resolved *netcdffile* path string.

    Parameters
    ----------
    kwargs : Dict[str, Any]
        Mutable keyword-argument dict from the caller.  Recognised keys are
        removed (popped) from it in-place.
    vmec_output_file : Path
        Path to the VMEC wout file, used as the default for *netcdffile*.

    Returns
    -------
    Tuple[Dict[str, Any], set, str]
        ``(params, provided_params, netcdffile)`` where *params* is a dict of
        all SIMPLE parameters with their defaults applied, *provided_params*
        is the set of parameter names that were explicitly supplied, and
        *netcdffile* is the resolved netCDF file path string.
    """
    provided_params: set = set(kwargs.keys())

    params: Dict[str, Any] = {
        "notrace_passing": kwargs.pop("notrace_passing", 0),
        "nper": kwargs.pop("nper", 1000),
        "npoiper": kwargs.pop("npoiper", 100),
        "ntimstep": kwargs.pop("ntimstep", 10000),
        "ntestpart": kwargs.pop("ntestpart", 1024),
        "trace_time": kwargs.pop("trace_time", 0.2),
        "num_surf": kwargs.pop("num_surf", 1),
        "sbeg": kwargs.pop("sbeg", 0.25),
        "phibeg": kwargs.pop("phibeg", 0.0),
        "thetabeg": kwargs.pop("thetabeg", 0.0),
        "contr_pp": kwargs.pop("contr_pp", -1.0),
        "facE_al": kwargs.pop("facE_al", 1.0),
        "npoiper2": kwargs.pop("npoiper2", DEFAULT_NPOIPER2),
        "n_e": kwargs.pop("n_e", 2),
        "n_d": kwargs.pop("n_d", 4),
        "ns_s": kwargs.pop("ns_s", 5),
        "ns_tp": kwargs.pop("ns_tp", 5),
        "multharm": kwargs.pop("multharm", 5),
        "isw_field_type": kwargs.pop("isw_field_type", 2),
        "generate_start_only": kwargs.pop("generate_start_only", False),
        "startmode": kwargs.pop("startmode", 1),
        "grid_density": kwargs.pop("grid_density", 0.0),
        "special_ants_file": kwargs.pop("special_ants_file", False),
        "integmode": kwargs.pop("integmode", 3),
        "relerr": kwargs.pop("relerr", 1e-13),
        "tcut": kwargs.pop("tcut", -1.0),
        "debug": kwargs.pop("debug", False),
        "class_plot": kwargs.pop("class_plot", False),
        "cut_in_per": kwargs.pop("cut_in_per", 0.0),
        "fast_class": kwargs.pop("fast_class", False),
        "vmec_B_scale": kwargs.pop("vmec_B_scale", 1.0),
        "vmec_RZ_scale": kwargs.pop("vmec_RZ_scale", 1.0),
        "swcoll": kwargs.pop("swcoll", False),
        "deterministic": kwargs.pop("deterministic", False),
        "old_axis_healing": kwargs.pop("old_axis_healing", True),
        "old_axis_healing_boundary": kwargs.pop("old_axis_healing_boundary", True),
        "am1": kwargs.pop("am1", 2.0),
        "am2": kwargs.pop("am2", 3.0),
        "Z1": kwargs.pop("Z1", 1.0),
        "Z2": kwargs.pop("Z2", 1.0),
        "densi1": kwargs.pop("densi1", 0.5e14),
        "densi2": kwargs.pop("densi2", 0.5e14),
        "tempi1": kwargs.pop("tempi1", 1.0e4),
        "tempi2": kwargs.pop("tempi2", 1.0e4),
        "tempe": kwargs.pop("tempe", 1.0e4),
        "batch_size": kwargs.pop("batch_size", 2000000000),
        "ran_seed": kwargs.pop("ran_seed", 12345),
        "reuse_batch": kwargs.pop("reuse_batch", False),
        "output_orbits_macrostep": kwargs.pop("output_orbits_macrostep", False),
        "output_error": kwargs.pop("output_error", False),
        "macrostep_time_grid": kwargs.pop("macrostep_time_grid", "linear"),
    }

    netcdffile_raw: Optional[str] = kwargs.pop("netcdffile", None)
    if netcdffile_raw is None:
        netcdffile = str(vmec_output_file.resolve())
    else:
        netcdffile = str(Path(netcdffile_raw).resolve())

    if kwargs:
        warnings.warn(
            f"Unknown SIMPLE parameters ignored: {list(kwargs.keys())}", UserWarning
        )

    return params, provided_params, netcdffile


def _parse_simple_output(output_dir: Path) -> Dict[str, Any]:
    """Parse SIMPLE output files and generate the loss-fraction plot.

    Parameters
    ----------
    output_dir : Path
        Directory containing ``confined_fraction.dat`` and ``times_lost.dat``.

    Returns
    -------
    dict
        Results with keys such as ``loss_fraction``, ``confined_fraction``, etc.
    """
    confined_fraction_file = output_dir / "confined_fraction.dat"
    times_lost_file = output_dir / "times_lost.dat"

    results: Dict[str, Any] = {
        "simple_output_dir": str(output_dir),
        "confined_fraction_file": str(confined_fraction_file)
        if confined_fraction_file.exists()
        else None,
        "times_lost_file": str(times_lost_file) if times_lost_file.exists() else None,
    }

    if confined_fraction_file.exists():
        with proc0_try(
            "Could not parse confined_fraction.dat: {e}",
            OSError,
            ValueError,
            IndexError,
        ):
            data = np.loadtxt(str(confined_fraction_file))
            if len(data.shape) == 2 and data.shape[1] >= 3:
                time_arr = data[:, 0]
                confined_passing = data[:, 1]
                confined_trapped = data[:, 2]
                total_confined = confined_passing + confined_trapped
                loss_fraction = 1.0 - total_confined

                final_row = data[-1]
                final_time = final_row[0]
                final_confined_passing = final_row[1]
                final_confined_trapped = final_row[2]
                final_total_confined = final_confined_passing + final_confined_trapped
                final_loss_fraction = 1.0 - final_total_confined

                results["loss_fraction"] = float(final_loss_fraction)
                results["confined_fraction"] = float(final_total_confined)
                results["confined_passing"] = float(final_confined_passing)
                results["confined_trapped"] = float(final_confined_trapped)
                results["final_time"] = float(final_time)

                proc0_print(f"  Final loss fraction: {final_loss_fraction:.6e}")
                proc0_print(f"  Final confined fraction: {final_total_confined:.6e}")

                with proc0_try("Could not generate loss fraction plot: {e}"):
                    plot_path = output_dir / "simple_loss_fraction.png"
                    _plot_simple_loss_fraction(time_arr, loss_fraction, plot_path)
                    results["loss_fraction_plot"] = str(plot_path)
                    proc0_print(f"  Generated loss fraction plot: {plot_path}")

    return results


def _plot_simple_loss_fraction(
    time: np.ndarray, loss_fraction: np.ndarray, output_path: Path
) -> None:
    """Plot loss fraction versus time from SIMPLE output.

    Parameters
    ----------
    time : np.ndarray
        Time array from confined_fraction.dat
    loss_fraction : np.ndarray
        Loss fraction array (1.0 - total_confined)
    output_path : Path
        Where to save the plot PNG file
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time, loss_fraction, label="Loss Fraction", color="red", linewidth=2)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Loss Fraction", fontsize=12)
    plt.title("Particle Loss Fraction Over Time", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close("all")
    gc.collect()
