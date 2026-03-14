"""Structural stress objective wrappers and helpers for coil optimization.

Provides guard and short-circuit wrappers around StructuralStressObjective
and builder for the FEM objective.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..mpi_utils import comm_world, is_mpi_enabled, proc0_print, proc0_warning

from ._objective_wrappers import _ObjectiveWrapperBase


class _StructuralStressGuardWrapper(_ObjectiveWrapperBase):
    r"""Wraps StructuralStressObjective to skip FEM when coil-coil distance is too small.

    Guard condition: when :math:`d_{\mathrm{cc}} < \mathtt{safety\_frac}
    \times \mathtt{cc\_threshold}`, J×B and Von Mises stress :math:`\sigma_{\mathrm{vm}}`
    blow up. This wrapper skips the expensive FEM evaluation and returns a finite
    penalty with zero gradient, steering the optimizer away from singular regions.
    """

    _EXCLUDE_ATTRS = _ObjectiveWrapperBase._EXCLUDE_ATTRS | frozenset(
        {
            "Jccdist",
            "cc_threshold",
            "safety_frac",
            "_guarded_penalty_gpa",
        }
    )

    def __init__(
        self,
        objective: Any,
        Jccdist: Any,
        cc_threshold: float,
        *,
        safety_frac: float = 1.0,
        guarded_penalty_gpa: float | None = None,
    ) -> None:
        super().__init__(objective)
        self.Jccdist = Jccdist
        self.cc_threshold = cc_threshold
        self.safety_frac = safety_frac
        self._guarded_penalty_gpa = guarded_penalty_gpa

    def _is_guarded(self) -> bool:
        d_cc = float(self.Jccdist.shortest_distance())
        return d_cc < self.safety_frac * self.cc_threshold

    def J(self) -> float:
        if self._is_guarded():
            return (
                self._guarded_penalty_gpa
                if self._guarded_penalty_gpa is not None
                else 10.0
            )
        return self.objective.J()

    def dJ(self, **kwargs: Any) -> Any:
        from simsopt._core.derivative import Derivative

        if self._is_guarded():
            if is_mpi_enabled() and comm_world.size > 1:
                ctrl = np.array([2, 0], dtype=np.int64)  # tag=2 skip
                comm_world.Bcast(ctrl, root=0)
            return Derivative({})
        return self.objective.dJ(**kwargs)


# Default tolerance: when weight*penalty is below this, skip dJ() (gradient negligible)
_DEFAULT_SHORT_CIRCUIT_TOLERANCE: float = 1e-4


class _StructuralStressShortCircuitWrapper(_ObjectiveWrapperBase):
    r"""Wraps StructuralStressObjective to short-circuit dJ() when contribution is negligible.

    When the weighted contribution :math:`w \cdot \max(0, J - t)^2` is below a
    tolerance (default 1e-4), the gradient contribution is negligible. This
    wrapper skips the expensive FD gradient (~65 FEM solves) in that case,
    returning ``Derivative({})`` immediately.

    With microscopic weight (e.g. 1e-10), dJ is skipped even if the raw penalty
    is large. With stress below threshold, penalty=0 so contribution=0.
    """

    _EXCLUDE_ATTRS = _ObjectiveWrapperBase._EXCLUDE_ATTRS | frozenset(
        {"threshold", "weight", "short_circuit_tolerance"}
    )

    def __init__(
        self,
        objective: Any,
        threshold: float,
        *,
        weight: float = 1.0,
        short_circuit_tolerance: float = _DEFAULT_SHORT_CIRCUIT_TOLERANCE,
    ) -> None:
        super().__init__(objective)
        self.threshold = threshold
        self.weight = weight
        self.short_circuit_tolerance = short_circuit_tolerance

    def J(self) -> float:
        """Forward to underlying objective."""
        return self.objective.J()

    def dJ(self, **kwargs: Any) -> Any:
        """Return zero gradient when weight * penalty < tolerance."""
        from simsopt._core.derivative import Derivative

        J_val = self.objective.J()
        penalty = max(0.0, J_val - self.threshold) ** 2
        contribution = self.weight * penalty
        if contribution < self.short_circuit_tolerance:
            if is_mpi_enabled() and comm_world.size > 1:
                ctrl = np.array([2, 0], dtype=np.int64)  # tag=2 skip
                comm_world.Bcast(ctrl, root=0)
            return Derivative({})
        return self.objective.dJ(**kwargs)


def _build_structural_stress_objective(
    coils: list,
    bs: Any,
    ncoils: int,
    coil_objective_terms: Dict[str, Any],
    thresholds: Dict[str, Any],
    *,
    out_dir: Path | None = None,
) -> Any | None:
    r"""Lazily construct a :class:`StructuralStressObjective` if scikit-fem and gmsh are available.

    The objective evaluates Von Mises stress :math:`\sigma_{\mathrm{vm}}` via FEM
    (J×B body force, linear elasticity). Guard with :class:`_StructuralStressGuardWrapper`
    when :math:`d_{\mathrm{cc}}` can become small.

    Parameters
    ----------
    coils : list
        Full simsopt ``Coil`` objects (all coils including symmetry copies).
    bs : BiotSavart
        Magnetic field evaluator.
    ncoils : int
        Number of unique base coils. FEM meshing uses coils[:ncoils]; body
        force uses all coils.
    coil_objective_terms : dict
        Must contain ``"structural_stress"``; may also contain
        ``structural_mesh_resolution_coarse``, ``structural_E``,
        ``structural_nu``, ``structural_eval_interval``,
        ``structural_stress_metric``.
    thresholds : dict
        Full thresholds dict (not used here but kept for future extension).

    Returns
    -------
    StructuralStressObjective | None
        The objective, or ``None`` if dependencies are missing.
    """
    try:
        from ._structural_objective import StructuralStressObjective
    except ImportError:
        proc0_print(
            "[structural_stress] scikit-fem or gmsh not available; "
            "skipping structural stress objective"
        )
        return None

    mesh_resolution_coarse = float(
        coil_objective_terms.get("structural_mesh_resolution_coarse", 0.16)
    )
    mesh_resolution_fine_raw = coil_objective_terms.get(
        "structural_mesh_resolution_fine"
    )
    refine_stress_ratio = float(
        coil_objective_terms.get("structural_refine_stress_ratio", 0.5)
    )
    E = float(coil_objective_terms.get("structural_E", 100e9))
    nu = float(coil_objective_terms.get("structural_nu", 0.3))
    eval_interval = int(coil_objective_terms.get("structural_eval_interval", 1))
    stress_metric = str(
        coil_objective_terms.get("structural_stress_metric", "mean_von_mises")
    )
    fb_width = float(thresholds.get("finite_build_width", 0.05))
    width = height = fb_width
    fd_step = float(coil_objective_terms.get("structural_fd_step", 1e-5))
    use_cached_K = bool(coil_objective_terms.get("structural_use_cached_K", False))

    # Adaptive mesh: coarse -> fine when stress >= refine_stress_ratio * threshold
    use_adaptive = mesh_resolution_fine_raw is not None
    if use_adaptive:
        mesh_resolution_fine = (
            float(mesh_resolution_fine_raw)
            if mesh_resolution_fine_raw is not None
            else mesh_resolution_coarse / 2
        )
        initial_res = mesh_resolution_coarse
        proc0_print(
            f"[structural_stress] Adaptive mesh: coarse={mesh_resolution_coarse} m, "
            f"fine={mesh_resolution_fine} m, refine_ratio={refine_stress_ratio}"
        )
    else:
        mesh_resolution_fine = None
        initial_res = mesh_resolution_coarse
        proc0_print(
            f"[structural_stress] Single mesh resolution: {mesh_resolution_coarse} m"
        )

    if initial_res > width:
        proc0_warning(
            f"[structural_stress] mesh_resolution ({initial_res} m) exceeds "
            f"coil cross-section ({width} m); clamping to {width} m."
        )
        if use_adaptive:
            mesh_resolution_coarse = width
            initial_res = width
        else:
            mesh_resolution_coarse = width
            initial_res = width

    # Prefer DOLFINx when available; allow YAML override via structural_backend
    from stellcoilbench.structural_analysis import _DOLFINX_AVAILABLE

    backend_default = "dolfinx" if _DOLFINX_AVAILABLE else "skfem"
    structural_backend = str(
        coil_objective_terms.get("structural_backend", backend_default)
    ).lower()
    if structural_backend not in ("dolfinx", "skfem"):
        proc0_warning(
            f"[structural_stress] Invalid structural_backend={structural_backend!r}; "
            f"using {backend_default}"
        )
        structural_backend = backend_default
    if structural_backend == "dolfinx" and not _DOLFINX_AVAILABLE:
        proc0_warning(
            "[structural_stress] DOLFINx requested but not available; falling back to skfem"
        )
        structural_backend = "skfem"
    elif structural_backend == "skfem" and not _DOLFINX_AVAILABLE:
        pass  # skfem fallback when DOLFINx unavailable

    proc0_print(
        f"[structural_stress] Building FEM objective: "
        f"metric={stress_metric}, mesh_res={initial_res}, "
        f"eval_interval={eval_interval}, unique_coils={ncoils}, backend={structural_backend}"
    )
    quadrature_degree = int(coil_objective_terms.get("structural_quadrature_degree", 1))
    polynomial_degree = int(coil_objective_terms.get("structural_polynomial_degree", 2))
    kwargs: Dict[str, Any] = {
        "unique_coils": coils[:ncoils],
        "bs": bs,
        "all_coils": coils,
        "width": width,
        "height": height,
        "E": E,
        "nu": nu,
        "stress_metric": stress_metric,
        "fd_step": fd_step,
        "eval_interval": eval_interval,
        "use_cached_K": use_cached_K,
        "structural_backend": structural_backend,
        "quadrature_degree": quadrature_degree,
        "polynomial_degree": polynomial_degree,
    }
    if use_adaptive:
        kwargs["mesh_resolution_coarse"] = mesh_resolution_coarse
        kwargs["mesh_resolution_fine"] = mesh_resolution_fine
        kwargs["refine_stress_ratio"] = refine_stress_ratio
    else:
        kwargs["mesh_resolution"] = mesh_resolution_coarse
    return StructuralStressObjective(**kwargs)
