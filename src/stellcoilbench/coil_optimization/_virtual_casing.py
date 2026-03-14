"""Virtual casing calculation setup.

Wraps simsopt's VirtualCasing.from_vmec to compute B_external_normal
at both the optimization resolution and at 2× resolution for plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..constants import VC_SRC_NPHI, VC_SRC_NTHETA
from ..mpi_utils import proc0_print

try:
    from simsopt.mhd.virtual_casing import VirtualCasing

    VIRTUAL_CASING_AVAILABLE = True
except ImportError:
    VIRTUAL_CASING_AVAILABLE = False
    VirtualCasing = None  # type: ignore


def _setup_virtual_casing(
    surface_file: str,
    surface_params: Dict[str, Any],
    surface_resolution: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run virtual casing calculation if enabled.

    Parameters
    ----------
    surface_file : str
        Resolved path to the plasma surface file.
    surface_params : dict
        Surface parameters from case config.
    surface_resolution : int
        Resolution used for the optimization surface.

    Returns
    -------
    vc_target : np.ndarray or None
        B_external_normal at optimization resolution.
    vc_target_plot : np.ndarray or None
        B_external_normal at 2× resolution for plotting.

    Raises
    ------
    ImportError
        If virtual casing is enabled but the package is not installed.
    ValueError
        If no VMEC wout file can be found.
    """
    use_virtual_casing = surface_params.get("virtual_casing", False)
    if not use_virtual_casing:
        return None, None

    if not VIRTUAL_CASING_AVAILABLE:
        raise ImportError(
            "Virtual casing is enabled but the virtual_casing package is not installed. "
            "Install it with: pip install git+https://github.com/hiddenSymmetries/virtual-casing"
        )

    surface_file_lower = surface_file.lower()
    vmec_file = None
    if "wout" in surface_file_lower:
        vmec_file = surface_file
    else:
        surface_path = Path(surface_file)
        potential_wout_files = [
            surface_path.parent / f"wout_{surface_path.stem}.nc",
            surface_path.parent / f"wout_{surface_path.stem.replace('input.', '')}.nc",
            Path("plasma_surfaces") / f"wout_{surface_path.stem}.nc",
            Path("plasma_surfaces")
            / f"wout_{surface_path.stem.replace('input.', '')}.nc",
        ]
        for wout_path in potential_wout_files:
            if wout_path.exists():
                vmec_file = str(wout_path)
                break

    if vmec_file is None:
        raise ValueError(
            f"Virtual casing is enabled but no VMEC wout file found for surface: {surface_file}. "
            "Virtual casing requires a VMEC wout file. Either provide a wout file directly "
            "or ensure a corresponding wout_*.nc file exists."
        )

    vc_src_nphi = VC_SRC_NPHI
    vc_src_ntheta = VC_SRC_NTHETA

    proc0_print("Running virtual casing calculation...")
    proc0_print(f"  VMEC file: {vmec_file}")
    proc0_print(f"  Source resolution: {vc_src_nphi} x {vc_src_ntheta}")
    proc0_print(
        f"  Target resolution (matches surface): {surface_resolution} x {surface_resolution}"
    )

    vc = VirtualCasing.from_vmec(
        vmec_file,
        src_nphi=vc_src_nphi,
        src_ntheta=vc_src_ntheta,
        trgt_nphi=surface_resolution,
        trgt_ntheta=surface_resolution,
    )
    vc_target = vc.B_external_normal.copy()
    proc0_print(
        f"  Virtual casing calculation complete. B_external_normal shape: {vc_target.shape}"
    )
    del vc
    import gc

    gc.collect()
    proc0_print(f"  Surface resolution: {surface_resolution} x {surface_resolution}")

    plot_resolution = 2 * surface_resolution
    proc0_print(
        f"  Computing virtual casing for full surface plotting (resolution: {plot_resolution} x {plot_resolution})..."
    )
    vc_plot = VirtualCasing.from_vmec(
        vmec_file,
        src_nphi=vc_src_nphi,
        src_ntheta=vc_src_ntheta,
        trgt_nphi=plot_resolution,
        trgt_ntheta=plot_resolution,
    )
    vc_target_plot = vc_plot.B_external_normal
    proc0_print(
        f"  Virtual casing for plotting complete. B_external_normal shape: {vc_target_plot.shape}"
    )

    return vc_target, vc_target_plot
