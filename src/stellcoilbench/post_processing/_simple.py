"""SIMPLE particle tracing functions for post-processing.

This submodule contains all functions related to running the SIMPLE
fast-particle tracing code (``simple.x``), including input file
generation, subprocess execution, output parsing, and ARIES-CS
reactor scaling.

Implementation is split across:
- :mod:`._simple_input` — input file generation
- :mod:`._simple_run` — subprocess execution
- :mod:`._simple_parse` — output parsing and parameter extraction
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

from ..mpi_utils import proc0_print, proc0_try, proc0_warning

from ._simple_input import _build_simple_input_content
from ._simple_parse import _extract_simple_params, _parse_simple_output
from ._simple_run import _run_simple_subprocess

if TYPE_CHECKING:
    from simsopt.mhd.vmec import Vmec


def _find_simple_executable(
    simple_executable_path: Optional[Path] = None,
) -> Optional[Path]:
    """Locate the SIMPLE ``simple.x`` executable.

    When *simple_executable_path* is ``None`` the function searches the
    project root directory (four levels up from this source file).  If the
    executable cannot be found, a warning is printed and ``None`` is
    returned so that the caller can skip particle tracing gracefully.

    Parameters
    ----------
    simple_executable_path : Path, optional
        Explicit path to ``simple.x``.  When provided the function only
        checks that the file exists.

    Returns
    -------
    Optional[Path]
        Absolute path to the ``simple.x`` executable, or ``None`` if not
        found.
    """
    if simple_executable_path is None:
        env_path = os.environ.get("SIMPLE_EXECUTABLE")
        if env_path:
            simple_executable_path = Path(env_path)
        else:
            try:
                project_root = Path(__file__).resolve().parent.parent.parent.parent
            except NameError:
                project_root = Path.cwd()

            simple_executable_path = project_root / "simple.x"

        if not simple_executable_path.exists() or not simple_executable_path.is_file():
            proc0_warning("simple.x executable not found. Skipping particle tracing.")
            proc0_print(f"  Expected location: {simple_executable_path}")
            proc0_print("  Tip: Copy simple.x to the project root directory.")
            return None

    if not simple_executable_path.exists():
        proc0_warning(
            f"simple.x executable not found at {simple_executable_path}. Skipping particle tracing."
        )
        return None

    return simple_executable_path.resolve()


def _read_vmec_for_scaling(
    netcdffile: str,
    vmec_B_scale: float,
    vmec_RZ_scale: float,
    provided_params: set,
) -> Tuple[float, float]:
    """Read VMEC wout file and compute ARIES-CS reactor scaling factors.

    If the device minor radius is below the ARIES-CS reference value, the
    geometry and B-field scaling factors are updated automatically (unless
    the caller explicitly provided them).

    Parameters
    ----------
    netcdffile : str
        Path to the VMEC wout netCDF file.
    vmec_B_scale : float
        Current B-field scaling factor (may be overridden).
    vmec_RZ_scale : float
        Current geometry scaling factor (may be overridden).
    provided_params : set
        Set of parameter names explicitly supplied by the caller.  If
        ``"vmec_B_scale"`` or ``"vmec_RZ_scale"`` are in this set the
        corresponding auto-scaling is skipped.

    Returns
    -------
    Tuple[float, float]
        ``(vmec_B_scale, vmec_RZ_scale)`` — potentially updated scaling
        factors.
    """
    from ..config_scheme import ARIES_CS_B0, ARIES_CS_MINOR_RADIUS

    with proc0_try(
        "Could not read VMEC file for auto-scaling: {e}",
        OSError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        on_catch=lambda: proc0_print(
            "  Using default scaling (vmec_RZ_scale=1.0, vmec_B_scale=1.0)"
        ),
    ):
        from scipy.io import netcdf_file

        with netcdf_file(netcdffile, "r", mmap=False) as f:
            r0 = float(f.variables["raxis_cc"].data[0])
            aminor = float(f.variables["Aminor_p"].data)
            bmnc = f.variables["bmnc"].data
            xm = f.variables["xm"].data
            xn = f.variables["xn"].data
            idx_00 = np.where((xm == 0) & (xn == 0))[0][0]
            b0 = abs(float(bmnc[-1, idx_00]))

        if aminor < ARIES_CS_MINOR_RADIUS:
            proc0_print(
                f"Scaling to ARIES-CS reactor (a={ARIES_CS_MINOR_RADIUS}m, B0={ARIES_CS_B0}T)"
            )

            if "vmec_RZ_scale" not in provided_params:
                vmec_RZ_scale = ARIES_CS_MINOR_RADIUS / aminor
                scaled_r0 = r0 * vmec_RZ_scale
                proc0_print(
                    f"  Geometry: a={aminor:.3f}m -> {ARIES_CS_MINOR_RADIUS}m (vmec_RZ_scale={vmec_RZ_scale:.2f})"
                )
                proc0_print(f"            R0={r0:.2f}m -> {scaled_r0:.2f}m")

            if "vmec_B_scale" not in provided_params and b0 > 0:
                vmec_B_scale = ARIES_CS_B0 / b0
                proc0_print(
                    f"  B-field: B0={b0:.2f}T -> {ARIES_CS_B0}T (vmec_B_scale={vmec_B_scale:.2f})"
                )
        else:
            proc0_print(
                f"Device at reactor scale: a={aminor:.2f}m, R0={r0:.2f}m, B0={b0:.2f}T (no scaling applied)"
            )

    return vmec_B_scale, vmec_RZ_scale


def run_simple_particle_tracing(
    vmec_equil: Vmec,
    output_dir: Path,
    simple_executable_path: Optional[Path] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run SIMPLE fast particle tracing using simple.x executable.

    Creates simple.in and runs simple.x on a VMEC equilibrium. All kwargs are
    optional with defaults from :func:`._simple_parse._extract_simple_params`. For full parameter
    documentation see https://github.com/itpplasma/SIMPLE.

    Parameters
    ----------
    vmec_equil : Vmec
        VMEC equilibrium with output_file pointing to wout.
    output_dir : Path
        Directory for simple.in and output files.
    simple_executable_path : Path, optional
        Path to simple.x. If None, uses project root or SIMPLE_EXECUTABLE env.

    **kwargs : Any
        SIMPLE config parameters (trace_time, sbeg, ntestpart, etc.).
        See https://github.com/itpplasma/SIMPLE for full list.

    Returns
    -------
    Dict[str, Any]
        Keys: simple_output_dir, confined_fraction_file, times_lost_file,
        loss_fraction, confined_fraction, confined_passing, confined_trapped, final_time.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Note: SIMPLE has a limitation with nfp=1 configurations (both tokamaks and nfp=1 stellarators)
    # The error "Spline not supported for a Phi period of 1" occurs for any nfp=1 configuration.
    # We don't preemptively skip here - let SIMPLE try and handle the error gracefully if it occurs.

    vmec_output_file = Path(vmec_equil.output_file)
    if not vmec_output_file.exists():
        raise FileNotFoundError(f"VMEC output file not found: {vmec_output_file}")

    simple_params, provided_params, netcdffile = _extract_simple_params(
        kwargs,
        vmec_output_file,
    )

    # Auto-scale device to ARIES-CS reactor parameters for SIMPLE particle tracing
    # SIMPLE traces 3.5 MeV alpha particles, which require reactor-scale devices
    # with proper field strength for meaningful confinement results.
    # Reference: ARIES-CS compact stellarator reactor design
    # (Ku et al., Fusion Science and Technology, 2008)
    vmec_B_scale, vmec_RZ_scale = _read_vmec_for_scaling(
        netcdffile,
        simple_params["vmec_B_scale"],
        simple_params["vmec_RZ_scale"],
        provided_params,
    )
    simple_params["vmec_B_scale"] = vmec_B_scale
    simple_params["vmec_RZ_scale"] = vmec_RZ_scale

    resolved_path = _find_simple_executable(simple_executable_path)
    if resolved_path is None:
        return {}
    simple_executable_path = resolved_path

    simple_in_content = _build_simple_input_content(
        simple_params,
        provided_params,
        netcdffile,
    )
    simple_in_path = output_dir / "simple.in"
    simple_in_path.write_text(simple_in_content)
    proc0_print(f"Created simple.in file: {simple_in_path}")

    proc0_print("Running simple.x particle tracing...")
    proc0_print(f"  Executable: {simple_executable_path}")
    proc0_print(f"  VMEC file: {netcdffile}")
    proc0_print(f"  Working directory: {output_dir}")

    from ..constants import SIMPLE_SUBPROCESS_TIMEOUT

    result = _run_simple_subprocess(
        simple_executable_path,
        output_dir,
        timeout=SIMPLE_SUBPROCESS_TIMEOUT,
    )
    if result is None:
        return {}

    return _parse_simple_output(output_dir)
