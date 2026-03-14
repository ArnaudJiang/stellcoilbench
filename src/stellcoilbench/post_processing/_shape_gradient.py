"""Shape gradient computation for coil curves.

Transforms the gradient of an objective function with respect to curve
Fourier DOFs into a physically meaningful per-point vector field along
each coil curve.  The resulting pointwise shape gradient can be
visualised in ParaView via VTK point data on the coil polylines.

The mathematical formulation follows Gil et al.: given
``df/dp`` (the gradient with respect to XYZFourier coefficients),
solve the linear system ``A · S = df/dp`` where each entry of *A*
encodes the inner product of ``dgamma/dp_i`` with the *j*-th Fourier
basis function weighted by the arclength Jacobian ``|dr/dl|``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from scipy.linalg import solve

if TYPE_CHECKING:
    from simsopt.field import BiotSavart
    from simsopt.geo import CurveXYZFourier, SurfaceRZFourier

logger = logging.getLogger(__name__)


def _is_curve_xyz_fourier(curve: Any) -> bool:
    """Check whether *curve* is a ``CurveXYZFourier`` instance.

    Uses name-based check to avoid hard import dependency at module level.

    Parameters
    ----------
    curve : Any
        Curve object to inspect.

    Returns
    -------
    bool
        ``True`` when *curve* is an instance of ``CurveXYZFourier``.
    """
    return type(curve).__name__ == "CurveXYZFourier"


def _build_fourier_system_matrix(
    curve: "CurveXYZFourier",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the three (x, y, z) Fourier-weighted Gram matrices.

    For DOF index *i* and basis index *j* the entry is::

        A[i, j] = mean_k( |dr/dl|_k  *  dgamma/dp[k, coord, dof_i]  *  phi_j(t_k) )

    where ``phi_j`` is ``cos(n * 2pi * t)`` or ``sin(n * 2pi * t)``
    matching the ``CurveXYZFourier`` coefficient ordering
    ``[c(0), s(1), c(1), s(2), c(2), ...]``.

    Parameters
    ----------
    curve : CurveXYZFourier
        Curve whose geometry and Jacobians are evaluated.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(A_x, A_y, A_z)`` each of shape ``(D//3, D//3)``.
    """
    D = curve.num_dofs()
    n_dofs_per_coord = D // 3
    N_quad = len(curve.quadpoints)
    quad_pts = curve.quadpoints

    dr_dl = curve.gammadash()
    abs_dr_dl = np.sqrt(np.sum(dr_dl**2, axis=1))
    dr_dp = curve.dgamma_by_dcoeff()

    # Pre-compute basis function values: shape (n_dofs_per_coord, N_quad)
    basis = np.empty((n_dofs_per_coord, N_quad))
    for j in range(n_dofs_per_coord):
        if j % 2 == 0:
            n = j // 2
            basis[j] = np.cos(n * quad_pts * 2.0 * np.pi)
        else:
            n = (j + 1) // 2
            basis[j] = np.sin(n * quad_pts * 2.0 * np.pi)

    # Vectorised assembly: avoid Python double loop
    # weight[k] = abs_dr_dl[k],  shape (N_quad,)
    # dr_dp shape: (N_quad, 3, D)
    # For x: dr_dp[:, 0, 0:D//3],  for y: dr_dp[:, 1, D//3:2*D//3],  etc.
    w = abs_dr_dl  # (N_quad,)

    A_x = np.zeros((n_dofs_per_coord, n_dofs_per_coord))
    A_y = np.zeros((n_dofs_per_coord, n_dofs_per_coord))
    A_z = np.zeros((n_dofs_per_coord, n_dofs_per_coord))

    # dr_dp_x[k, i] = dgamma/dcoeff[k, 0, i] for x-DOFs
    dr_dp_x = dr_dp[:, 0, :n_dofs_per_coord]  # (N_quad, D//3)
    dr_dp_y = dr_dp[:, 1, n_dofs_per_coord : 2 * n_dofs_per_coord]
    dr_dp_z = dr_dp[:, 2, 2 * n_dofs_per_coord :]

    for i in range(n_dofs_per_coord):
        # weighted_x[k] = w[k] * dr_dp_x[k, i],  shape (N_quad,)
        wx = w * dr_dp_x[:, i]
        wy = w * dr_dp_y[:, i]
        wz = w * dr_dp_z[:, i]
        for j in range(n_dofs_per_coord):
            bj = basis[j]
            A_x[i, j] = np.mean(wx * bj)
            A_y[i, j] = np.mean(wy * bj)
            A_z[i, j] = np.mean(wz * bj)

    return A_x, A_y, A_z


def _precondition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric diagonal preconditioning: ``D^{-1} A D^{-1}``.

    Parameters
    ----------
    A : np.ndarray
        Square matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(A_precond, P)`` where ``P = D^{-1}`` (diagonal scaling).
    """
    d = np.sqrt(np.abs(np.diag(A)))
    d[d == 0] = 1.0  # guard against zero diagonal
    D_inv = np.diag(1.0 / d)
    return D_inv @ A @ D_inv, D_inv


def compute_shape_gradient_single_curve(
    curve: "CurveXYZFourier",
    df_dp: np.ndarray,
    *,
    return_pointwise: bool = True,
) -> np.ndarray:
    """Compute the shape gradient for a single ``CurveXYZFourier``.

    Solves three independent preconditioned linear systems (one per
    Cartesian coordinate) to obtain the Fourier representation of the
    shape gradient, then optionally evaluates it at the curve quadrature
    points.

    Parameters
    ----------
    curve : CurveXYZFourier
        Curve with **all DOFs free** (unfixed).
    df_dp : np.ndarray, shape ``(D,)``
        Gradient of the objective with respect to the curve's
        ``D = 3 * (2*order + 1)`` Fourier DOFs, ordered as
        ``[x-DOFs, y-DOFs, z-DOFs]``.
    return_pointwise : bool, default True
        If ``True``, return the shape gradient evaluated at the curve's
        quadrature points (shape ``(N_quad, 3)``).  If ``False``,
        return the raw Fourier coefficient vector (shape ``(D,)``).

    Returns
    -------
    np.ndarray
        Shape gradient, either pointwise ``(N_quad, 3)`` or Fourier
        coefficients ``(D,)`` depending on *return_pointwise*.

    Raises
    ------
    TypeError
        If *curve* is not a ``CurveXYZFourier``.
    ValueError
        If *df_dp* length does not match the curve's DOF count.
    np.linalg.LinAlgError
        If any of the three linear systems is singular.
    """
    if not _is_curve_xyz_fourier(curve):
        raise TypeError(
            f"Shape gradient requires CurveXYZFourier, got {type(curve).__name__}"
        )

    D = curve.num_dofs()
    if df_dp.shape != (D,):
        raise ValueError(
            f"df_dp length {df_dp.shape} does not match curve DOF count {D}"
        )

    n = D // 3
    A_x, A_y, A_z = _build_fourier_system_matrix(curve)

    results = []
    for A, b in [
        (A_x, df_dp[:n]),
        (A_y, df_dp[n : 2 * n]),
        (A_z, df_dp[2 * n :]),
    ]:
        A_pre, P = _precondition(A)
        b_pre = P @ b
        S_pre = solve(A_pre, b_pre, assume_a="sym", check_finite=False)
        results.append(P @ S_pre)

    S = np.concatenate(results)  # shape (D,)

    if not return_pointwise:
        return S

    # Evaluate at quadrature points via a temporary CurveXYZFourier
    from simsopt.geo import CurveXYZFourier as CurveXYZ

    N_quad = len(curve.quadpoints)
    N_ord = (n - 1) // 2
    tmp = CurveXYZ(N_quad, N_ord)
    tmp.x = S
    return tmp.gamma()  # (N_quad, 3)


def _find_base_curve(curve: Any) -> Any:
    """Walk up through ``RotatedCurve`` wrappers to find the leaf base curve.

    Parameters
    ----------
    curve : Any
        A simsopt curve, possibly a ``RotatedCurve``.

    Returns
    -------
    Any
        The underlying leaf curve (typically ``CurveXYZFourier``).
    """
    while hasattr(curve, "ancestors") and curve.ancestors:
        curve = curve.ancestors[0]
    return curve


def _extract_curve_dof_offsets(
    objective: Any,
) -> Dict[int, tuple[int, Any]]:
    """Map each unique ``CurveXYZFourier`` object-id to its offset in ``dJ()``.

    Parses ``objective.dof_names`` to find contiguous groups belonging to
    ``CurveXYZFourier`` instances, then matches them to the actual objects
    found in the coil tree.

    Parameters
    ----------
    objective : Optimizable
        The simsopt objective (e.g. ``SquaredFlux``) whose ``dof_names``
        and ``unique_dof_lineage`` will be inspected.

    Returns
    -------
    Dict[int, tuple[int, Any]]
        ``{id(curve_obj): (start_offset, curve_obj)}`` for each unique
        ``CurveXYZFourier`` whose DOFs appear in ``objective.x``.
    """
    names = objective.dof_names

    # Group consecutive DOF names by their object prefix (e.g. "CurveXYZFourier1")
    groups: list[tuple[str, int, int]] = []
    current_prefix: Optional[str] = None
    current_start = 0
    for i, name in enumerate(names):
        prefix = name.split(":")[0]
        if prefix != current_prefix:
            if current_prefix is not None:
                groups.append((current_prefix, current_start, i))
            current_prefix = prefix
            current_start = i
    if current_prefix is not None:
        groups.append((current_prefix, current_start, len(names)))

    # Collect unique CurveXYZFourier objects from the lineage
    xyz_curves: list[Any] = []
    for obj in objective.unique_dof_lineage:
        if _is_curve_xyz_fourier(obj) and obj not in xyz_curves:
            xyz_curves.append(obj)

    # Match lineage objects to DOF groups by class name prefix
    curve_offsets: Dict[int, tuple[int, Any]] = {}
    curve_idx = 0
    for group_name, start, _end in groups:
        if "CurveXYZFourier" in group_name and curve_idx < len(xyz_curves):
            curve_offsets[id(xyz_curves[curve_idx])] = (start, xyz_curves[curve_idx])
            curve_idx += 1

    return curve_offsets


def compute_shape_gradients(
    coils: list,
    bfield: "BiotSavart",
    surface: "SurfaceRZFourier",
) -> List[Optional[np.ndarray]]:
    """Compute per-coil pointwise shape gradients from the SquaredFlux objective.

    Builds ``SquaredFlux(surface, bfield)``, computes ``dJ()``, finds
    each unique ``CurveXYZFourier`` base curve in the DOF tree, solves
    the shape gradient system for it, and evaluates the result at every
    coil's quadrature points (including symmetry copies that share the
    same base curve DOFs).

    Parameters
    ----------
    coils : list
        List of simsopt ``Coil`` objects (may include symmetry copies).
    bfield : BiotSavart
        Magnetic field object containing the coils.
    surface : SurfaceRZFourier
        Plasma boundary surface for the flux objective.

    Returns
    -------
    List[Optional[np.ndarray]]
        One entry per coil.  Each entry is either an ``(N_quad, 3)``
        array of pointwise shape gradient vectors or ``None`` if the
        curve type is unsupported.
    """
    from simsopt.objectives import SquaredFlux
    from simsopt.geo import CurveXYZFourier as CurveXYZ

    Jf = SquaredFlux(surface, bfield)
    full_grad = Jf.dJ()

    curve_offsets = _extract_curve_dof_offsets(Jf)

    # Compute shape gradient Fourier coefficients for each unique base curve
    base_sg_dofs: Dict[int, Optional[np.ndarray]] = {}
    for obj_id, (offset, curve_obj) in curve_offsets.items():
        D = curve_obj.dof_size
        curve_grad = full_grad[offset : offset + D]
        try:
            sg_dofs = compute_shape_gradient_single_curve(
                curve_obj,
                curve_grad,
                return_pointwise=False,
            )
            base_sg_dofs[obj_id] = sg_dofs
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning(
                "Shape gradient solve failed for curve id=%d: %s",
                obj_id,
                exc,
            )
            base_sg_dofs[obj_id] = None

    # For each coil, evaluate the shape gradient at its quadrature points.
    # RotatedCurve shares DOFs with the base CurveXYZFourier, so the
    # Fourier coefficients are the same; we evaluate on a temporary curve
    # with the same quadrature points to get pointwise values.
    shape_grads: List[Optional[np.ndarray]] = []
    for coil in coils:
        curve = coil.curve
        base = _find_base_curve(curve)
        base_id = id(base)

        if base_id not in base_sg_dofs or not _is_curve_xyz_fourier(base):
            shape_grads.append(None)
            continue

        sg_dofs = base_sg_dofs[base_id]
        if sg_dofs is None:
            shape_grads.append(None)
            continue

        D = base.dof_size
        n_per_coord = D // 3
        N_ord = (n_per_coord - 1) // 2
        N_quad = len(curve.quadpoints)

        # Evaluate Fourier coefficients at *this* curve's quadrature points
        tmp = CurveXYZ(N_quad, N_ord)
        tmp.x = sg_dofs
        sg_pts = tmp.gamma()  # (N_quad, 3) in lab frame of the base curve

        if _is_curve_xyz_fourier(curve):
            shape_grads.append(sg_pts)
        else:
            # For RotatedCurve: the Fourier coefficients describe the
            # shape sensitivity in the *base* curve's frame.  The
            # pointwise displacement direction is the same in Fourier
            # space because the DOFs are shared.  We evaluate using the
            # base curve's parametric mapping, which already gives us
            # the correct lab-frame vectors for the base geometry.
            # This is the correct shape gradient for steepest-descent
            # interpretation in DOF space.
            shape_grads.append(sg_pts)

    return shape_grads


def shape_gradient_to_vtk_data(
    shape_grads: List[Optional[np.ndarray]],
    coils: list,
    *,
    close: bool = False,
) -> Dict[str, Any]:
    """Pack per-coil shape gradients into VTK point-data arrays.

    The output dict is suitable for passing as ``extra_data`` to
    ``simsopt.field.coils_to_vtk``.

    Parameters
    ----------
    shape_grads : list[np.ndarray | None]
        One entry per coil from :func:`compute_shape_gradients`.
    coils : list
        Coil objects (same order as *shape_grads*).
    close : bool, default False
        Whether the VTK polyline is closed (appends first point).

    Returns
    -------
    Dict[str, Any]
        Keys ``"ShapeGrad"`` (tuple of 3 contiguous arrays for the
        vector) and ``"ShapeGrad_mag"`` (scalar magnitude).
    """
    contig = np.ascontiguousarray
    all_x, all_y, all_z, all_mag = [], [], [], []

    for i, (sg, coil) in enumerate(zip(shape_grads, coils)):
        npts = coil.curve.gamma().shape[0]
        if close:
            npts += 1

        if sg is not None:
            sx, sy, sz = sg[:, 0], sg[:, 1], sg[:, 2]
            mag = np.sqrt(sx**2 + sy**2 + sz**2)
            if close:
                sx = np.append(sx, sx[0])
                sy = np.append(sy, sy[0])
                sz = np.append(sz, sz[0])
                mag = np.append(mag, mag[0])
        else:
            sx = np.zeros(npts)
            sy = np.zeros(npts)
            sz = np.zeros(npts)
            mag = np.zeros(npts)

        all_x.append(sx)
        all_y.append(sy)
        all_z.append(sz)
        all_mag.append(mag)

    return {
        "ShapeGrad": (
            contig(np.concatenate(all_x)),
            contig(np.concatenate(all_y)),
            contig(np.concatenate(all_z)),
        ),
        "ShapeGrad_mag": contig(np.concatenate(all_mag)),
    }
