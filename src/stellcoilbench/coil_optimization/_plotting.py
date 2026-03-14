"""
Plotting utilities for coil optimization.

Contains 3D B·n error visualization, VTK surface data, plotting surface creation,
and verbose iteration output formatting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .._mpl import MATPLOTLIB_AVAILABLE, get_plt
from ..mpi_utils import proc0_print, proc0_warning
from ..path_utils import get_surface_search_base_dirs, resolve_surface_path
from ..path_utils import load_surface_with_range

from simsopt.geo import SurfaceRZFourier

from ._optimization_types import OptimizationLoopContext

logger = logging.getLogger(__name__)

SAFE_ABS_B_FLOOR = 1e-10


def _compute_surface_vtk_data(
    bs: Any,
    s_plot: SurfaceRZFourier,
    qphi: int,
    qtheta: int,
    include_bn: bool = False,
) -> Dict[str, Any]:
    """Compute VTK point data for a surface from a BiotSavart field.

    Uses shared B·n computation from :mod:`stellcoilbench.post_processing._bdotn`.

    Parameters
    ----------
    bs : BiotSavart
        Magnetic field object.
    s_plot : SurfaceRZFourier
        Plotting surface (full torus).
    qphi, qtheta : int
        Grid dimensions (must match s_plot quadpoints).
    include_bn : bool, default=False
        If True, include raw B_N in addition to B_N/|B| and modB.

    Returns
    -------
    Dict[str, Any]
        Point data dict suitable for ``s_plot.to_vtk(..., extra_data=pointData)``.
    """
    from ..post_processing._bdotn import compute_bdotn_point_data

    bdotn_data = compute_bdotn_point_data(bs, s_plot)
    BdotN = bdotn_data["BdotN"][:, :, None]
    absB = bdotn_data["absB"][:, :, None]
    pointData: Dict[str, Any] = {
        "B_N/|B|": BdotN / absB,
        "modB": absB,
    }
    if include_bn:
        pointData["B_N"] = BdotN
    return pointData


def _create_plotting_surface(
    s: SurfaceRZFourier,
    surface_resolution: int,
    kwargs: Dict[str, Any],
) -> Tuple[SurfaceRZFourier, int, int]:
    """Create a full-torus plotting surface from the optimization surface.

    Delegates to :func:`stellcoilbench.post_processing.load_surface_with_range`
    when the surface has a filename or when surface_file/surface_path is in
    kwargs; falls back to coefficient copying otherwise.

    Parameters
    ----------
    s : SurfaceRZFourier
        Optimization surface (may be half-period for stellarator symmetry).
    surface_resolution : int
        Base resolution (nphi = ntheta = plot_upsample_factor * surface_resolution).
    kwargs : Dict[str, Any]
        May contain plot_upsample_factor (default: 2), surface_file, or
        surface_path for fallback when s.filename is not set.

    Returns
    -------
    tuple
        (s_plot, qphi, qtheta) - plotting surface and grid dimensions.
    """
    plot_upsample = kwargs.get("plot_upsample_factor", 2)
    qphi = plot_upsample * surface_resolution
    qtheta = plot_upsample * surface_resolution

    surface_path_to_try: str | Path | None = None
    if hasattr(s, "filename") and s.filename:
        surface_path_to_try = s.filename
    else:
        surface_path_to_try = kwargs.get("surface_file") or kwargs.get("surface_path")
        if (
            surface_path_to_try is not None
            and not Path(surface_path_to_try).is_absolute()
        ):
            base_dirs = get_surface_search_base_dirs()
            resolved = resolve_surface_path(str(surface_path_to_try), base_dirs)
            surface_path_to_try = (
                resolved if resolved is not None else surface_path_to_try
            )

    s_plot = None
    if surface_path_to_try is not None:
        try:
            s_plot = load_surface_with_range(
                surface_path_to_try,
                surface_range="full torus",
                nphi=qphi,
                ntheta=qtheta,
            )
        except (UnicodeDecodeError, ValueError, OSError) as exc:
            logger.debug("Failed to load s_plot from surface file: %s", exc)
            s_plot = None

    if s_plot is None:
        quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
        quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)
        s_plot = SurfaceRZFourier(
            nfp=s.nfp,
            stellsym=s.stellsym,
            mpol=s.mpol,
            ntor=s.ntor,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )

    for m in range(s.mpol + 1):
        for n in range(-s.ntor, s.ntor + 1):
            if s.get_rc(m, n) != 0:
                s_plot.set_rc(m, n, s.get_rc(m, n))
            if s.get_zs(m, n) != 0:
                s_plot.set_zs(m, n, s.get_zs(m, n))
    return s_plot, qphi, qtheta


def _segments_from_mask(points: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Split *points* into contiguous segments where *mask* is True.

    Parameters
    ----------
    points : np.ndarray
        Array of 3-D points, shape ``(N, 3)``.
    mask : np.ndarray
        Boolean array of length ``N``.

    Returns
    -------
    list[np.ndarray]
        Contiguous sub-arrays of *points* where *mask* is True.
    """
    segments: list[np.ndarray] = []
    start = 0
    for i in range(1, len(points)):
        if mask[i] != mask[i - 1]:
            if mask[i - 1]:
                segments.append(points[start:i])
            start = i
    if mask[-1]:
        segments.append(points[start:])
    return segments


def _plot_bn_error_3d(
    surface,
    bs,
    coils,
    out_dir: Path,
    filename: str = "bn_error_3d_plot.png",
    title: str = "B_N/|B| Error on Plasma Surface with Optimized Coils",
    plot_upsample: int = 1,
    vc_target: np.ndarray | None = None,
) -> None:
    """
    Generate a 3D plot showing B_N/|B| error on the plasma surface.

    Renders the plasma surface colored by |B_N/|B|| (or B_N error vs vc_target
    if virtual casing is used). Coils are overlaid. Requires matplotlib.

    Parameters
    ----------
    surface : SurfaceRZFourier
        Plasma surface for plotting (should be full torus).
    bs : BiotSavart
        BiotSavart object containing the magnetic field from coils.
    coils : list
        List of coil objects to plot.
    out_dir : Path
        Directory where the PNG will be saved.
    filename : str, optional
        Output filename (default: bn_error_3d_plot.png).
    title : str, optional
        Plot title.
    plot_upsample : int, optional
        Factor to upsample surface quadrature for smoother plot (default: 1).
    vc_target : np.ndarray | None, optional
        Virtual casing target B_N; if provided, error = |B_N - vc_target|.
    """
    if not MATPLOTLIB_AVAILABLE:
        proc0_warning("matplotlib not available, skipping 3D plot generation")
        return

    plt = get_plt()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib import cm
    from matplotlib.colors import Normalize

    # Upsample surface for smoother plotting when possible
    plot_surface = surface
    if isinstance(surface, SurfaceRZFourier) and plot_upsample > 1:
        try:
            qphi = max(16, int(len(surface.quadpoints_phi) * plot_upsample))
            qtheta = max(16, int(len(surface.quadpoints_theta) * plot_upsample))
            quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
            quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)
            plot_surface = SurfaceRZFourier(
                nfp=surface.nfp,
                stellsym=surface.stellsym,
                mpol=surface.mpol,
                ntor=surface.ntor,
                quadpoints_phi=quadpoints_phi,
                quadpoints_theta=quadpoints_theta,
            )
            for m in range(surface.mpol + 1):
                for n in range(-surface.ntor, surface.ntor + 1):
                    rc_val = surface.get_rc(m, n)
                    zs_val = surface.get_zs(m, n)
                    if rc_val != 0:
                        plot_surface.set_rc(m, n, rc_val)
                    if zs_val != 0:
                        plot_surface.set_zs(m, n, zs_val)
        except (ValueError, RuntimeError, AttributeError) as _:
            plot_surface = surface

    # Get surface points - grid should be square (nphi == ntheta)
    surface_points = plot_surface.gamma().reshape(-1, 3)
    npoints = surface_points.shape[0]
    nphi_plot = int(np.sqrt(npoints))
    ntheta_plot = nphi_plot

    # Reshape surface points to grid
    x_surf = surface_points[:, 0].reshape((nphi_plot, ntheta_plot))
    y_surf = surface_points[:, 1].reshape((nphi_plot, ntheta_plot))
    z_surf = surface_points[:, 2].reshape((nphi_plot, ntheta_plot))

    # Calculate B_N/|B| on surface using shared B·n computation
    from ..post_processing._bdotn import compute_bdotn_point_data

    bdotn_data = compute_bdotn_point_data(bs, plot_surface)
    BdotN_coils = bdotn_data["BdotN"]
    abs_B = bdotn_data["absB"]

    # If virtual casing target is provided, subtract it from the coil B_N
    if vc_target is not None:
        BdotN_error = np.abs(BdotN_coils - vc_target)
    else:
        BdotN_error = np.abs(BdotN_coils)

    # Avoid division by zero
    abs_B = np.where(abs_B > SAFE_ABS_B_FLOOR, abs_B, SAFE_ABS_B_FLOOR)
    bn_over_b = BdotN_error / abs_B

    # Create figure with 3D subplot
    fig = plt.figure(figsize=(12, 9), dpi=100)  # type: ignore
    ax = fig.add_subplot(111, projection="3d")  # type: ignore

    # Plot surface with B_N/|B| as colormap (opaque to avoid artifacts)
    norm = Normalize(vmin=0, vmax=bn_over_b.max() if bn_over_b.max() > 0 else 1)  # type: ignore
    facecolors = cm.viridis(norm(bn_over_b))  # type: ignore[attr-defined]
    ax.plot_surface(  # type: ignore[attr-defined]
        x_surf,
        y_surf,
        z_surf,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=True,
        rstride=1,
        cstride=1,
        zorder=1,
    )

    # Plot coils colored by current magnitude with simple front/back layering
    currents = [abs(c.current.get_value()) for c in coils]
    current_norm = Normalize(  # type: ignore[call-overload]
        vmin=min(currents) if currents else 0.0,
        vmax=max(currents) if currents else 1.0,
    )
    current_cmap = cm.plasma  # type: ignore

    center = np.array([x_surf.mean(), y_surf.mean(), z_surf.mean()])
    azim = np.deg2rad(ax.azim)  # type: ignore[attr-defined]
    elev = np.deg2rad(ax.elev)  # type: ignore[attr-defined]
    view_vec = np.array(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ]
    )

    front_segments: list[tuple[np.ndarray, tuple[float, float, float]]] = []

    for coil in coils:
        coil_points = coil.curve.gamma()
        current_val = abs(coil.current.get_value())
        color_rgba = current_cmap(current_norm(current_val))
        if len(color_rgba) == 4:
            color = tuple(color_rgba[:3])
        else:
            color = color_rgba
        closed = np.vstack([coil_points, coil_points[0]])
        depth = (closed - center) @ view_vec
        front_mask = depth >= 0
        back_mask = ~front_mask

        for seg in _segments_from_mask(closed, back_mask):
            ax.plot(
                seg[:, 0],
                seg[:, 1],
                seg[:, 2],
                color=color,
                linewidth=2.2,
                solid_capstyle="round",
                zorder=0,
            )

        for seg in _segments_from_mask(closed, front_mask):
            front_segments.append((seg, color))

    # Set labels and title
    ax.set_xlabel("X (m)", fontsize=12)  # type: ignore
    ax.set_ylabel("Y (m)", fontsize=12)  # type: ignore
    ax.set_zlabel("Z (m)", fontsize=12)  # type: ignore
    ax.set_title(title, fontsize=13, pad=16)  # type: ignore

    # Add surface colorbar
    mappable = cm.ScalarMappable(cmap=cm.viridis, norm=norm)  # type: ignore
    mappable.set_array(bn_over_b)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, pad=0.1)  # type: ignore
    cbar.set_label("|B_N|/|B|", fontsize=12, rotation=270, labelpad=20)

    # Add coil current colorbar on the left side
    coil_mappable = cm.ScalarMappable(cmap=current_cmap, norm=current_norm)  # type: ignore
    coil_mappable.set_array(currents)
    coil_cbar = plt.colorbar(  # type: ignore
        coil_mappable,
        ax=ax,
        shrink=0.6,
        aspect=20,
        pad=0.08,
        location="left",
    )
    coil_cbar.set_label("|I| (A)", fontsize=12, rotation=90, labelpad=18)

    # Draw front coil segments after the surface for better depth cues
    for seg, color in front_segments:
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            seg[:, 2],
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=3,
        )

    # Set equal aspect ratio
    max_range = (
        np.array(
            [
                x_surf.max() - x_surf.min(),
                y_surf.max() - y_surf.min(),
                z_surf.max() - z_surf.min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (x_surf.max() + x_surf.min()) * 0.5
    mid_y = (y_surf.max() + y_surf.min()) * 0.5
    mid_z = (z_surf.max() + z_surf.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)  # type: ignore
    ax.set_ylim(mid_y - max_range, mid_y + max_range)  # type: ignore
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  # type: ignore

    # Clean up axes for a sleeker look
    ax.grid(True)  # type: ignore
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):  # type: ignore
        axis.pane.fill = False  # type: ignore
        axis.pane.set_edgecolor("w")  # type: ignore

    # Save as PNG for smaller file size
    plot_path = out_dir / filename
    plt.savefig(plot_path, format="png", dpi=100, bbox_inches="tight")  # type: ignore
    plt.close(fig)  # type: ignore


_MAX_COILS_TO_PRINT: int = 4
"""Number of coils to show in the optimization setup summary."""


def _print_optimization_setup_summary(ctx: OptimizationLoopContext) -> None:
    """Print a diagnostic summary of the optimization setup."""
    ncoils_tf_total = len(ctx.coils)
    proc0_print(
        f"Coils: {ctx.ncoils} unique, {ncoils_tf_total} total (before optimization)"
    )

    proc0_print("\n" + "=" * 60)
    proc0_print("OPTIMIZATION SETUP SUMMARY")
    proc0_print("=" * 60)
    proc0_print(f"  Algorithm:          {ctx.algorithm}")
    proc0_print(f"  Max iterations:     {ctx.max_iterations}")
    proc0_print(
        f"  Surface:            R0={ctx.major_radius:.4f} m, a={ctx.th['minor_radius']:.4f} m"
    )
    proc0_print(f"  a0 (ARIES/a):       {ctx.th['a0']:.2f}")
    proc0_print(f"  Coil width:         {ctx.coil_width:.6f} m")
    proc0_print(f"  Ncoils (unique):    {ctx.ncoils}")
    proc0_print(f"  Total current:      {ctx.total_current:.2f} A")
    proc0_print(f"  Target B:           {ctx.target_B:.4f} T")

    proc0_print("\n  Thresholds (device scale):")
    proc0_print(f"    flux_threshold:        {ctx.flux_threshold:.2e}")
    proc0_print(
        f"    finite_build_width:    {ctx.th.get('finite_build_width', 'N/A')} m"
    )
    proc0_print(f"    cc_threshold:          {ctx.cc_threshold:.2g} m")
    proc0_print(f"    cs_threshold:          {ctx.cs_threshold:.2g} m")
    proc0_print(f"    length_threshold:      {ctx.length_threshold:.2g} m")
    proc0_print(f"    curvature_threshold:   {ctx.curvature_threshold:.2g} 1/m")
    torsion_threshold = ctx.th.get("torsion_threshold")
    if (
        torsion_threshold is not None
        and ctx.effective_obj_terms
        and "coil_torsion" in ctx.effective_obj_terms
    ):
        proc0_print(f"    torsion_threshold:      {torsion_threshold:.2g} 1/m")
    proc0_print(f"    msc_threshold:         {ctx.msc_threshold:.2g} 1/m²")
    proc0_print(f"    force_threshold:       {ctx.force_threshold:.2g} N/m")
    proc0_print(f"    torque_threshold:      {ctx.torque_threshold:.2g} N·m")
    proc0_print(
        f"    arclength_var_thresh:  {ctx.arclength_variation_threshold:.2g} m²"
    )
    if ctx.structural_obj is not None and ctx.effective_obj_terms:
        stress_thresh = float(
            ctx.effective_obj_terms.get("structural_stress_threshold", 0.0)
        )
        proc0_print(f"    structural_stress_thresh: {stress_thresh:.2e} Pa")

    proc0_print(f"\n  Constraint list ({len(ctx.c_list)} entries):")
    for ci, name_thresh in enumerate(ctx.constraint_names_and_thresholds):
        cname, cthresh = name_thresh
        scale = ctx.constraint_scaling.get(ci + 1, "N/A")
        if cthresh is None:
            thresh_str = "None"
        elif isinstance(cthresh, float):
            thresh_str = f"{cthresh:.2g}"
        else:
            thresh_str = str(cthresh)
        scale_str = f"{scale:.2g}" if isinstance(scale, float) else str(scale)
        proc0_print(
            f"    [{ci + 1}] {cname:20s}  threshold={thresh_str}  scaling={scale_str}"
        )
    proc0_print(f"    CC distance idx:  {ctx.cc_distance_index}")
    proc0_print(f"    CS distance idx:  {ctx.cs_distance_index}")

    proc0_print("\n  Initial coil geometry:")
    for i, c in enumerate(ctx.coils[: min(ctx.ncoils, _MAX_COILS_TO_PRINT)]):
        g = np.asarray(c.curve.gamma()).reshape(-1, 3)
        r = np.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2)
        proc0_print(
            f"    Coil {i}: R=[{r.min():.2e}, {r.max():.2e}] m, "
            f"Z=[{g[:, 2].min():.2e}, {g[:, 2].max():.2e}] m, "
            f"I={c.current.get_value():.2e} A"
        )

    proc0_print("\n  Initial objective values:")
    proc0_print(f"    SquaredFlux (Jf):      {ctx.Jf.J():.2e}")
    proc0_print(
        f"    CC distance (Jccdist): {ctx.Jccdist.J():.2e}  "
        f"(shortest={ctx.Jccdist.shortest_distance():.2e} m)"
    )
    proc0_print(
        f"    CS distance (Jcsdist): {ctx.Jcsdist.J():.2e}  "
        f"(shortest={ctx.Jcsdist.shortest_distance():.2e} m)"
    )
    if ctx.Jlink is not None:
        try:
            proc0_print(f"    LinkingNumber:         {ctx.Jlink.J():.2e}")
        except Exception as e:
            logger.debug("Skipping Jlink in context summary: %s", e)
    if ctx.Jforce is not None:
        try:
            proc0_print(f"    Force (Jforce):        {ctx.Jforce.J():.2e}")
        except Exception as e:
            logger.debug("Skipping Jforce in context summary: %s", e)
    for i, jl in enumerate(ctx.Jls[: min(ctx.ncoils, _MAX_COILS_TO_PRINT)]):
        proc0_print(f"    Length[{i}]:             {jl.J():.2e} m")
    if ctx.structural_obj is not None:
        try:
            proc0_print(
                f"    Structural stress (σ_vm): {ctx.structural_obj.J():.2e} GPa"
            )
        except Exception as e:
            logger.debug("Skipping structural_obj in context summary: %s", e)
    proc0_print("=" * 60 + "\n")
