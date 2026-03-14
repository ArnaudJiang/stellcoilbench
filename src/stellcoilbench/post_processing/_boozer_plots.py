"""Boozer surface plots and iota/quasisymmetry profiles from VMEC equilibrium.

Extracts flux-surface data from VMEC wout files, runs booz_xform to obtain
Boozer angles, and plots iota, quasisymmetry error, and related profiles.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from .._mpl import MATPLOTLIB_AVAILABLE, ensure_mpl_agg, get_plt
from .._optional_imports import optional_import
from ..mpi_utils import proc0_print
from ..utils import suppress_output, timed_section

booz_xform_mod = optional_import("booz_xform", "", fallback=None)

if not MATPLOTLIB_AVAILABLE:
    raise ImportError("matplotlib is required for post_processing")
ensure_mpl_agg()
plt = get_plt()

if TYPE_CHECKING:
    from simsopt.mhd.vmec import Vmec


def plot_boozer_surface(
    equil: Vmec,
    output_path: Path,
    dpi: int = 100,
) -> None:
    r"""Plot Boozer surfaces from VMEC equilibrium.

    Creates a 2×2 grid showing Boozer surfaces at :math:`s \in \{0, 0.25,
    0.5, 1.0\}`, where :math:`s` is the normalised flux coordinate.

    Parameters
    ----------
    equil : Vmec
        VMEC equilibrium object.
    output_path : Path
        Where to save the plot.
    dpi : int, default=100
        Resolution for saved figure.
    """
    if booz_xform_mod is None:
        raise ImportError(
            "booz_xform is required for Boozer surface plots. "
            "Install with: pip install booz-xform"
        )
    bx = booz_xform_mod

    with timed_section("booz_xform_read_wout"):
        b2 = bx.Booz_xform()
        b2.read_wout(equil.output_file)

    # booz_xform uses 0-based indices for compute_surfs (>= 0 and < ns_in)
    ns_in_raw = getattr(b2, "ns_in", None)
    if ns_in_raw is None or not isinstance(ns_in_raw, (int, np.integer)):
        ns_in = max(1, len(equil.wout.iotas) - 1)  # type: ignore[union-attr]
    else:
        ns_in = int(ns_in_raw)
    max_js_0based = max(0, ns_in - 1)

    # Sample 4 evenly spaced surfaces (0-based input indices)
    if max_js_0based == 0:
        js_indices = [0, 0, 0, 0]
    else:
        js_indices = np.linspace(0, max_js_0based, 4, dtype=int).tolist()

    # Only compute booz_xform on the specific surfaces we need to plot
    # This is much faster than computing all surfaces
    b2.compute_surfs = js_indices
    proc0_print(
        f"Computing Boozer transform on surfaces: {js_indices} (of {ns_in} total)"
    )

    with timed_section("booz_xform_run"):
        with suppress_output():
            b2.run()

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    plt.rcParams["font.family"] = "serif"
    plt.rc("font", size=18)  # Increased base font size

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()

    # Plot each surface with error handling
    # surfplot(b2, js=i): js is 0-based index among *output* surfaces (0..3)
    for i, input_idx in enumerate(js_indices):
        ax = axes_flat[i]
        plt.sca(ax)
        output_idx = i
        try:
            bx.surfplot(b2, js=output_idx, fill=False)
        except (IndexError, ValueError) as e:
            # If index error, try progressively smaller output indices
            error_str = str(e).lower()
            if "out of bounds" in error_str or "index" in error_str:
                for fallback_output in range(i - 1, -1, -1):
                    try:
                        bx.surfplot(b2, js=fallback_output, fill=False)
                        output_idx = fallback_output
                        input_idx = js_indices[
                            min(fallback_output, len(js_indices) - 1)
                        ]
                        break
                    except (IndexError, ValueError, RuntimeError):
                        continue
                else:
                    bx.surfplot(b2, js=0, fill=False)
                    output_idx = 0
                    input_idx = js_indices[0]
            else:
                raise

        # Increase font sizes for axes labels and tick labels
        ax.xaxis.label.set_fontsize(18)
        ax.yaxis.label.set_fontsize(18)
        ax.tick_params(labelsize=16)

        # Compute s value from flux: use b2.s_in when available, else linear mapping
        try:
            s_arr = getattr(b2, "s_in", None)
            if s_arr is not None and input_idx < len(s_arr):
                s_val = float(s_arr[input_idx])
            else:
                s_val = input_idx / max(1, max_js_0based)
        except (TypeError, ValueError, IndexError):
            s_val = input_idx / max(1, max_js_0based)
        ax.set_title(f"s = {s_val:.2f}", fontsize=20)

    # Increase colorbar font sizes after all plots are created
    # Find colorbars by checking axes that aren't in our main subplot axes
    main_axes_set = set(axes_flat)
    for ax_fig in fig.axes:
        if ax_fig not in main_axes_set:
            # This is likely a colorbar axis
            try:
                ax_fig.tick_params(labelsize=16)
                # Update labels if they exist
                label = ax_fig.get_ylabel()
                if label:
                    ax_fig.set_ylabel(label, fontsize=18)
                label = ax_fig.get_xlabel()
                if label:
                    ax_fig.set_xlabel(label, fontsize=18)
            except (AttributeError, TypeError, ValueError):
                pass

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close("all")  # Close all figures to ensure cleanup

    # Explicit garbage collection to free memory
    gc.collect()


def plot_iota_profile(
    equil: Vmec,
    output_path: Path,
    sign: int = 1,
    equil_original: Vmec | None = None,
    dpi: int = 100,
) -> None:
    r"""Plot rotational transform :math:`\iota` profile from VMEC equilibrium.

    Plots :math:`\iota(s)` vs normalised toroidal flux :math:`s`.

    Parameters
    ----------
    equil : Vmec
        VMEC equilibrium object (self-consistent solution).
    output_path : Path
        Where to save the plot.
    sign : int, default=1
        Sign to apply to iota (1 or -1).
    equil_original : Vmec, optional
        Original VMEC equilibrium object for comparison.
    dpi : int, default=100
        Resolution for saved figure.
    """
    # Access iota profile from VMEC output
    iotas = equil.wout.iotas[1:]  # type: ignore
    psi_s = np.linspace(0, len(iotas) * equil.ds, len(iotas))

    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.rcParams["font.family"] = "serif"
    plt.rc("font", size=15)

    # Plot original surface if provided
    if equil_original is not None:
        iotas_orig = equil_original.wout.iotas[1:]  # type: ignore
        psi_s_orig = np.linspace(
            0, len(iotas_orig) * equil_original.ds, len(iotas_orig)
        )
        plt.plot(
            psi_s_orig, sign * iotas_orig, "b-", label="Original surface", linewidth=2
        )

    # Plot self-consistent solution
    plt.plot(psi_s, sign * iotas, "rx", label="Self-consistent (QFM)", markersize=8)

    if equil_original is not None:
        plt.legend()

    plt.ylabel(r"rotational transform $\iota$")
    plt.xlabel("Normalized toroidal flux s")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_quasisymmetry_profile(
    qs_profile: np.ndarray,
    radii: np.ndarray,
    output_path: Path,
    qs_profile_original: Optional[np.ndarray] = None,
    radii_original: Optional[np.ndarray] = None,
    dpi: int = 100,
) -> None:
    r"""Plot quasisymmetry error profile vs normalised flux :math:`s`.

    Quasisymmetry error measures deviation of :math:`|\mathbf{B}|` from
    constancy on flux surfaces.

    Parameters
    ----------
    qs_profile : np.ndarray
        Quasisymmetry error at each radius (self-consistent solution).
    radii : np.ndarray
        Normalized toroidal flux radii.
    output_path : Path
        Where to save the plot.
    qs_profile_original : np.ndarray, optional
        Original quasisymmetry error profile for comparison.
    radii_original : np.ndarray, optional
        Normalized toroidal flux radii for original profile.
    dpi : int, default=100
        Resolution for saved figure.
    """
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = "serif"
    plt.rc("font", size=15)

    # Plot original surface if provided
    if qs_profile_original is not None and radii_original is not None:
        plt.semilogy(
            radii_original,
            qs_profile_original,
            "b-",
            label="Original surface",
            linewidth=2,
        )

    # Plot self-consistent solution
    plt.semilogy(radii, qs_profile, "rx", label="Self-consistent (QFM)", markersize=8)

    if qs_profile_original is not None:
        plt.legend()

    plt.xlabel("Normalized toroidal flux")
    plt.ylabel("Two-term quasisymmetry error")
    plt.grid()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
