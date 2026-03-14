"""Recompute coils_linked_to_surface from stored coil/surface data."""

from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from pathlib import Path

from ..path_utils import get_surface_search_base_dirs, resolve_surface_path

logger = logging.getLogger(__name__)


def _recompute_coils_linked_to_surface(
    submission_path: Path,
    surface_name: str,
    repo_root: Path,
) -> bool | None:
    """Recompute *coils_linked_to_surface* from stored coil/surface data.

    Uses the per-phi-slice R check (matching the fix in coil_optimization.py)
    to determine whether each coil topologically encircles the plasma at its
    local toroidal position. Loads BiotSavart from the submission (zip or
    directory) and the plasma surface from ``repo_root/plasma_surfaces/``.

    Parameters
    ----------
    submission_path : Path
        Path to a submission directory or zip file containing
        ``biot_savart_optimized.json``.
    surface_name : str
        Plasma surface identifier (e.g. ``LandremanPaul2021_QA``).
    repo_root : Path
        Repository root; plasma surfaces are sought under
        ``repo_root / "plasma_surfaces"``.

    Returns
    -------
    bool | None
        True if all coils encircle the plasma; False if any coil does not;
        None if data cannot be loaded (missing files, simsopt not installed).
    """
    try:
        from simsopt._core import load as simsopt_load  # type: ignore
        import numpy as np
        from stellcoilbench.post_processing import load_surface_with_range
    except ImportError:  # pragma: no cover
        return None

    # ---- locate BiotSavart JSON inside the submission ----
    bs_json_bytes: bytes | None = None
    try:
        if submission_path.suffix == ".zip":
            with zipfile.ZipFile(submission_path, "r") as zf:
                bs_files = [
                    n for n in zf.namelist() if n.endswith("biot_savart_optimized.json")
                ]
                if not bs_files:
                    return None
                # Prefer highest order (order_16 > order_8 > order_4 > root)
                bs_files.sort(reverse=True)
                bs_json_bytes = zf.read(bs_files[0])
        else:
            # Regular directory – look for biot_savart in parent
            submission_dir = submission_path.parent
            candidates = sorted(
                submission_dir.rglob("biot_savart_optimized.json"),
                key=lambda p: str(p),
                reverse=True,
            )
            if candidates:
                bs_json_bytes = candidates[0].read_bytes()
    except OSError:  # pragma: no cover
        return None

    if bs_json_bytes is None:
        return None

    # ---- load BiotSavart ----
    tmpfile: str | None = None
    try:
        fd, tmpfile = tempfile.mkstemp(suffix=".json")
        os.write(fd, bs_json_bytes)
        os.close(fd)
        bs = simsopt_load(tmpfile)
    except (OSError, RuntimeError, ValueError):  # pragma: no cover
        if tmpfile is not None:
            try:
                os.unlink(tmpfile)
            except OSError as exc:
                logger.debug("Could not remove temp file %s: %s", tmpfile, exc)
        return None
    finally:
        if tmpfile is not None:
            try:
                os.unlink(tmpfile)
            except OSError as exc:  # pragma: no cover
                logger.debug("Could not remove temp file %s: %s", tmpfile, exc)

    # ---- locate and load the plasma surface ----
    base_dirs = get_surface_search_base_dirs(
        plasma_surfaces_dir=repo_root / "plasma_surfaces"
    )
    surface_file = None
    for name in [f"input.{surface_name}", surface_name]:
        surface_file = resolve_surface_path(name, base_dirs)
        if surface_file is not None:
            break
    if surface_file is None:
        return None

    try:
        s = load_surface_with_range(
            surface_file, surface_range="full torus", nphi=64, ntheta=64
        )
    except (OSError, RuntimeError, ValueError):  # pragma: no cover
        return None

    # ---- per-phi-slice linking check ----
    try:
        surface_gamma = s.gamma()
        R_surface = np.sqrt(surface_gamma[:, :, 0] ** 2 + surface_gamma[:, :, 1] ** 2)
        R_min_per_phi = np.min(R_surface, axis=1)
        R_max_per_phi = np.max(R_surface, axis=1)
        phi_surface_slices = np.arctan2(surface_gamma[:, 0, 1], surface_gamma[:, 0, 0])

        coils = bs.coils
        for coil in coils:
            gamma = coil.curve.gamma()
            R_coil = np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)
            phi_coil = np.arctan2(gamma[:, 1], gamma[:, 0])
            dphi = phi_coil[:, None] - phi_surface_slices[None, :]
            dphi = np.abs(np.arctan2(np.sin(dphi), np.cos(dphi)))
            nearest = np.argmin(dphi, axis=1)
            local_R_min = R_min_per_phi[nearest]
            local_R_max = R_max_per_phi[nearest]
            if not (np.any(R_coil < local_R_min) and np.any(R_coil > local_R_max)):
                return False
        return True
    except (RuntimeError, ValueError, AttributeError):  # pragma: no cover
        return None
