"""Unit tests for StructuralStressObjective.

Tests cover J(), dJ(), Taylor test, mesh caching, x property, eval_interval,
stress_metric options, timing, penalty wrapper integration, msh_path support,
and timing breakdown profiling.

Taylor test regime note:
- test_taylor_test: Exercises the BARE StructuralStressObjective (no wrapper).
  No threshold; the objective is always the raw stress. Verifies structural
  gradient specifically.
- Optimizer Taylor test (in _scipy_optimizer): Runs on the full JF
  (Weight * QuadraticPenalty(structural, threshold, "max") + ...). When
  structural stress J <= threshold, the penalty is zero and
  _StructuralStressShortCircuitWrapper returns Derivative({}) for dJ()—the
  structural gradient is never called. The optimizer Taylor test therefore
  does NOT verify the structural gradient when stress is below threshold.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

skfem = pytest.importorskip("skfem")
gmsh = pytest.importorskip("gmsh")
pytest.importorskip("stellcoilbench.coil_optimization._structural_objective")

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts" / "fem_benchmarks"


def _make_test_coils(
    ncoils: int = 2,
    order: int = 2,
    R0: float = 1.0,
    R1: float = 0.1,
) -> tuple[Any, Any, Any, Any]:
    """Create simple test coils with BiotSavart.

    Parameters
    ----------
    ncoils : int
        Number of base coils.
    order : int
        Fourier order of curves.
    R0 : float
        Major radius [m].
    R1 : float
        Minor radius [m].

    Returns
    -------
    coils : list
        Coil objects from coils_via_symmetries.
    bs : BiotSavart
        BiotSavart field from coils.
    base_curves : list
        Base curve objects.
    base_currents : list
        Base Current objects.
    """
    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    from simsopt.geo import create_equally_spaced_curves

    base_curves = create_equally_spaced_curves(
        ncoils, 1, stellsym=True, R0=R0, R1=R1, order=order
    )
    base_currents = [Current(1e5) for _ in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, 1, True)
    bs = BiotSavart(coils)
    return coils, bs, base_curves, base_currents


def _create_prebuilt_torus_mesh_for_coils(
    coils: Any,
    tmp_path: Path,
    *,
    width: float = 0.05,
    height: float = 0.05,
    mesh_size: float = 0.16,
    suffix: str = "",
) -> Path:
    """Create a pre-built torus mesh for structural tests to avoid addPipe failures.

    Uses _generate_torus_mesh_gmsh with geometry derived from the first coil.
    Tests use this when the rectangular pipe mesh fails (PLC / self-intersection).
    """
    from stellcoilbench.coil_optimization._structural_mesh import (
        _generate_torus_mesh_gmsh,
    )
    from stellcoilbench.structural_analysis._common import _compute_coil_frame

    frame = _compute_coil_frame(coils[0])
    gamma = frame["gamma"]
    xy = gamma[:, :2]
    R_major = float(np.mean(np.linalg.norm(xy, axis=1)))
    if R_major < 1e-6:
        R_major = 1.2
    centroid = np.mean(gamma, axis=0)
    r_minor = max(width, height) / 2
    msh_file = tmp_path / f"coil{suffix}.msh"
    temp_msh = _generate_torus_mesh_gmsh(
        float(centroid[0]),
        float(centroid[1]),
        float(centroid[2]),
        R_major,
        r_minor,
        mesh_size,
    )
    shutil.copy(temp_msh, msh_file)
    temp_msh.unlink(missing_ok=True)
    return msh_file


def test_surface_sweep_to_msh_produces_valid_mesh(tmp_path: Path) -> None:
    """Verify _surface_sweep_to_msh produces tetrahedral mesh when Gmsh succeeds.

    Uses a simple circular coil. Gmsh addThruSections and STL→volume fallback
    can fail on some geometries (PLC errors, parametrization); we assert only
    when one succeeds.
    """
    meshio = pytest.importorskip("meshio")
    n_pts = 24
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    gamma = np.column_stack([1.2 * np.cos(theta), 1.2 * np.sin(theta), np.zeros(n_pts)])
    gammadash = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros(n_pts)])
    gammadash = gammadash / np.linalg.norm(gammadash, axis=1, keepdims=True)

    from stellcoilbench.coil_optimization._structural_mesh import _surface_sweep_to_msh

    tmp_msh = _surface_sweep_to_msh(
        gamma, gammadash, width=0.05, height=0.05, mesh_size=0.12
    )
    if tmp_msh is None:
        pytest.skip("Gmsh sweep-to-volume failed (addThruSections and STL fallback)")
    try:
        m = meshio.read(str(tmp_msh))
        n_tets = 0
        for cb in m.cells:
            if cb.type in ("tetra", "tetra10"):
                n_tets = len(cb.data)
                break
        assert n_tets > 0
        assert m.points.shape[0] > 0
    finally:
        tmp_msh.unlink(missing_ok=True)


def _make_objective(
    coils: Any,
    bs: Any,
    *,
    ncoils: int | None = None,
    width: float = 0.05,
    height: float = 0.05,
    mesh_resolution: float = 0.16,
    mesh_resolution_coarse: float | None = None,
    mesh_resolution_fine: float | None = None,
    refine_stress_ratio: float = 0.5,
    stress_metric: str = "max_von_mises",
    fd_step: float = 1e-7,
    eval_interval: int = 1,
    use_cached_K: bool = False,
    msh_path: Path | None = None,
    structural_backend: str | None = None,
) -> Any:
    """Create StructuralStressObjective with given parameters.

    Parameters
    ----------
    coils : list
        Full coil list (all coils including symmetry copies).
    bs : BiotSavart
        BiotSavart field evaluator.
    ncoils : int, optional
        Number of unique base coils. If None, uses len(coils) (all coils unique).
    msh_path : Path, optional
        Pre-built mesh path. If provided, avoids rectangular pipe mesh generation.
    """
    from stellcoilbench.coil_optimization._structural_objective import (
        StructuralStressObjective,
    )

    n = ncoils if ncoils is not None else len(coils)
    kwargs: dict[str, Any] = {
        "unique_coils": coils[:n],
        "bs": bs,
        "all_coils": coils,
        "width": width,
        "height": height,
        "stress_metric": stress_metric,
        "fd_step": fd_step,
        "eval_interval": eval_interval,
        "use_cached_K": use_cached_K,
        "msh_path": msh_path,
    }
    if structural_backend is not None:
        kwargs["structural_backend"] = structural_backend
    if mesh_resolution_coarse is not None:
        kwargs["mesh_resolution_coarse"] = mesh_resolution_coarse
        kwargs["mesh_resolution_fine"] = mesh_resolution_fine
        kwargs["refine_stress_ratio"] = refine_stress_ratio
    else:
        kwargs["mesh_resolution"] = mesh_resolution
    return StructuralStressObjective(**kwargs)


class TestStructuralStressObjective:
    """Tests for StructuralStressObjective."""

    def test_j_returns_positive_scalar(self, tmp_path: Path) -> None:
        """J() should return a positive float for non-trivial coils."""
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(coils, bs, mesh_resolution=0.16, msh_path=msh)
        result = obj.J()
        assert isinstance(result, (float, np.floating))
        assert result >= 0.0
        assert np.isscalar(result)

    def test_j_returns_zero_for_zero_current(self, tmp_path: Path) -> None:
        """With zero current, Von Mises stress should be near zero."""
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.geo import create_equally_spaced_curves

        base_curves = create_equally_spaced_curves(
            2, 1, stellsym=True, R0=1.0, R1=0.1, order=2
        )
        base_currents = [Current(0.0) for _ in range(2)]
        coils = coils_via_symmetries(base_curves, base_currents, 1, True)
        bs = BiotSavart(coils)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(coils, bs, mesh_resolution=0.16, msh_path=msh)
        result = obj.J()
        assert isinstance(result, (float, np.floating))
        assert result >= 0.0
        # J() returns GPa; 1 MPa = 1e-3 GPa
        assert result < 1e-3, "Zero current should yield negligible stress"

    def test_bs_mutual_has_length_n_unique(self, tmp_path: Path) -> None:
        """_bs_mutual should have length n_unique, not n_total.

        With symmetry (4 coils from 2 unique), mutual BiotSavart objects are
        one per unique coil: bs_mutual[k] excludes indices j where j % n_unique == k.
        """
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        n_total = len(coils)
        n_unique = 2
        assert n_total == 4, "coils_via_symmetries(2,1,stellsym) yields 4 coils"
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(
            coils, bs, ncoils=n_unique, mesh_resolution=0.16, msh_path=msh
        )
        assert len(obj._bs_mutual) == n_unique
        assert len(obj._bs_mutual) < n_total

    def test_dj_shape_matches_dofs(self, tmp_path: Path) -> None:
        """dJ() should return array with same shape as x."""
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(coils, bs, mesh_resolution=0.16, msh_path=msh)
        x = obj.x
        dJ = obj.dJ()
        assert isinstance(dJ, np.ndarray)
        np.testing.assert_equal(np.asarray(dJ).shape, np.asarray(x).shape)

    def _run_structural_taylor_test(self, obj: Any, label: str = "") -> None:
        """Run Taylor test for structural dJ(): |J(x+εh)-J(x)-ε∇J·h| = O(ε²)."""
        np.random.seed(42)
        x0 = np.asarray(obj.x, dtype=float)
        h = np.random.randn(len(x0))
        h = h / np.linalg.norm(h)

        J0 = obj.J()
        dJ0 = np.asarray(obj.dJ(), dtype=float)

        epsilons = [1e-4, 1e-5, 1e-6]
        residuals = []
        for eps in epsilons:
            obj.x = x0 + eps * h
            Jp = obj.J()
            residual = abs(Jp - J0 - eps * np.dot(dJ0, h))
            residuals.append(residual)

        for i in range(len(residuals) - 1):
            assert residuals[i] > 0, (
                f"{label}Residual at eps={epsilons[i]}: {residuals[i]}"
            )
            ratio = residuals[i + 1] / residuals[i]
            expected_ratio = (epsilons[i + 1] / epsilons[i]) ** 2
            assert ratio < expected_ratio * 15.0, (
                f"{label}Taylor residual should decrease quadratically: "
                f"ratio={ratio:.4f} vs expected ~{expected_ratio:.4f} at eps={epsilons[i]:.1e}"
            )

    def test_taylor_test_fresh_K(self, tmp_path: Path) -> None:
        """Structural dJ() with use_cached_K=False passes Taylor test (accurate gradient)."""
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(
            coils, bs, mesh_resolution=0.16, use_cached_K=False, msh_path=msh
        )
        self._run_structural_taylor_test(obj, label="[fresh K] ")

    def test_taylor_test_cached_K(self, tmp_path: Path) -> None:
        """Structural dJ() with use_cached_K=True passes Taylor test (self-consistent).

        Uses skfem backend (K caching supported for skfem only).
        """
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(
            coils,
            bs,
            mesh_resolution=0.16,
            use_cached_K=True,
            msh_path=msh,
            structural_backend="skfem",
        )
        self._run_structural_taylor_test(obj, label="[cached K] ")

    def test_k_cache_gradient_accuracy(self, tmp_path: Path) -> None:
        """Compare gradient from dJ() (cached K) vs gradient with fresh K per perturbation.

        Reports max relative gradient error and cosine similarity. The cached K
        path reuses stiffness from baseline; fresh K assembles for each FD perturb.
        Documents K-cache accuracy for validation; Taylor test still passes with
        cached K, so the gradient is self-consistent for optimization.
        Uses skfem backend (K caching supported for skfem only).
        """
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(
            coils,
            bs,
            mesh_resolution=0.16,
            fd_step=1e-5,
            msh_path=msh,
            structural_backend="skfem",
        )

        grad_cached = obj._compute_gradient_impl(use_cached_K=True)
        grad_fresh = obj._compute_gradient_impl(use_cached_K=False)

        grad_norm = np.linalg.norm(grad_fresh) + 1e-14
        abs_err = np.linalg.norm(grad_cached - grad_fresh)
        rel_err_norm = float(abs_err / grad_norm)

        threshold = 1e-10
        mask = np.abs(grad_fresh) > threshold
        if np.any(mask):
            rel_err_where_sig = np.abs(grad_cached[mask] - grad_fresh[mask]) / (
                np.abs(grad_fresh[mask]) + 1e-14
            )
            max_rel_err_component = float(np.max(rel_err_where_sig))
        else:
            max_rel_err_component = 0.0

        dot = np.dot(grad_cached, grad_fresh)
        norms = np.linalg.norm(grad_cached) * np.linalg.norm(grad_fresh) + 1e-20
        cosine_sim = float(dot / norms)

        print(
            f"\n  [K-cache validation] rel_err_norm={rel_err_norm:.2e}, "
            f"max_rel_err_component={max_rel_err_component:.2e}, "
            f"cosine_sim={cosine_sim:.4f}"
        )

        # Both gradients must be finite (sanity check)
        assert np.all(np.isfinite(grad_cached)) and np.all(np.isfinite(grad_fresh))

    def test_mesh_caching(self, tmp_path: Path) -> None:
        """Second J() call should reuse cached mesh (not regenerate)."""
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(coils, bs, mesh_resolution=0.16, msh_path=msh)
        j1 = obj.J()
        j2 = obj.J()
        assert j1 == j2

    def test_stress_changes_when_coils_deform(self, tmp_path: Path) -> None:
        """Stress metric must change when coil geometry is perturbed.

        Regression test for constant von Mises during optimization.
        """
        coils, bs, base_curves, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(coils, bs, mesh_resolution=0.16, msh_path=msh)
        J1 = obj.J()
        # Perturb first coil's curve DOFs
        curve = base_curves[0]
        x0 = np.asarray(curve.x, dtype=float).copy()
        curve.x = x0 + 0.01
        J2 = obj.J()
        curve.x = x0
        assert J2 != J1, (
            f"Stress must change when coils deform: J1={J1:.6e}, J2={J2:.6e}"
        )
        tol = 1e-9
        assert abs(J2 - J1) > tol, (
            f"Stress change too small: |J2-J1|={abs(J2 - J1):.2e} <= {tol}"
        )

    def test_x_property_roundtrip(self, tmp_path: Path) -> None:
        """Setting x then getting it should return the same values."""
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(coils, bs, mesh_resolution=0.16, msh_path=msh)
        x_orig = np.asarray(obj.x, dtype=float).copy()
        obj.x = x_orig
        x_back = np.asarray(obj.x, dtype=float)
        np.testing.assert_allclose(x_back, x_orig)

    def test_eval_interval(self, tmp_path: Path) -> None:
        """With eval_interval > 1, J() should return cached value between evals."""
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(
            coils, bs, mesh_resolution=0.16, eval_interval=2, msh_path=msh
        )
        j1 = obj.J()
        j2 = obj.J()
        assert j1 == j2

    def test_dj_scaling_with_eval_interval(self, tmp_path: Path) -> None:
        """dJ() is independent of eval_interval; J() FEM evals scale down with it.

        dJ() uses _evaluate_stress_impl directly (not J()), so the gradient
        must be identical for any eval_interval. Separately, J() with
        eval_interval > 1 should perform fewer FEM solves when called
        repeatedly with unchanged x.
        """
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)

        # 1. dJ() invariance: gradient identical for eval_interval=1 vs 5
        obj1 = _make_objective(
            coils, bs, mesh_resolution=0.16, eval_interval=1, msh_path=msh
        )
        obj2 = _make_objective(
            coils, bs, mesh_resolution=0.16, eval_interval=5, msh_path=msh
        )
        np.testing.assert_allclose(obj1.full_x, obj2.full_x)
        dj1 = obj1.dJ()
        dj2 = obj2.dJ()
        np.testing.assert_allclose(dj1, dj2, rtol=1e-9, atol=1e-12)

        # 2. J() FEM scaling: with eval_interval=3, 6 J() calls (same x) -> 2 FEM evals
        fem_count: list[int] = [0]
        obj3 = _make_objective(
            coils, bs, mesh_resolution=0.16, eval_interval=3, msh_path=msh
        )
        original_eval = obj3._evaluate_stress

        def counted_eval() -> float:
            fem_count[0] += 1
            return original_eval()

        with patch.object(obj3, "_evaluate_stress", side_effect=counted_eval):
            for _ in range(6):
                obj3.J()
        assert fem_count[0] == 2, (
            f"eval_interval=3, 6 J() calls: expected 2 FEM evals, got {fem_count[0]}"
        )

    def test_eval_interval_after_refine_mesh(self, tmp_path: Path) -> None:
        """After refine_mesh() clears cache, J() must re-evaluate (no AssertionError).

        refine_mesh() sets _cached_J = None. The next J() may have
        eval_count % eval_interval != 0; without the fix we would skip eval
        and hit assert _cached_J is not None.
        """
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(
            coils,
            bs,
            mesh_resolution_coarse=0.16,
            mesh_resolution_fine=0.08,
            refine_stress_ratio=0.5,
            eval_interval=2,
            msh_path=msh,
        )
        obj.J()  # Prime: eval_count=1, _cached_J set
        obj.refine_mesh(0.08)  # Clears _cached_J
        # Next J(): eval_count % 2 != 0 would skip eval in buggy code; must force eval
        j_after = obj.J()
        assert np.isfinite(j_after) and j_after >= 0.0

    def test_adaptive_mesh_starts_coarse(self, tmp_path: Path) -> None:
        """With mesh_resolution_coarse, objective starts in adaptive mode with coarse mesh."""
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(
            coils,
            bs,
            mesh_resolution_coarse=0.16,
            mesh_resolution_fine=0.08,
            refine_stress_ratio=0.5,
            msh_path=msh,
        )
        assert obj._adaptive_mesh is True
        assert obj._refinement_done is False
        assert obj._mesh_resolution == 0.16
        assert obj._mesh_resolution_coarse == 0.16
        assert obj._mesh_resolution_fine == 0.08
        assert obj._refine_stress_ratio == 0.5

    def test_refine_mesh_updates_resolution_and_state(self, tmp_path: Path) -> None:
        """refine_mesh() updates resolution, clears cache, sets _refinement_done."""
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(
            coils,
            bs,
            mesh_resolution_coarse=0.16,
            mesh_resolution_fine=0.08,
            refine_stress_ratio=0.5,
            msh_path=msh,
        )
        obj._cached_J = 1.0  # Simulate cached value
        obj.refine_mesh(0.08)
        assert obj._mesh_resolution == 0.08
        assert obj._refinement_done is True
        assert obj._cached_J is None

    def test_stress_metrics(self, tmp_path: Path) -> None:
        """Different stress_metric options should all return valid values."""
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        for metric in ("max_von_mises", "mean_von_mises", "lp_von_mises"):
            obj = _make_objective(
                coils, bs, mesh_resolution=0.16, stress_metric=metric, msh_path=msh
            )
            result = obj.J()
            assert isinstance(result, (float, np.floating))
            assert result >= 0.0
            assert np.isfinite(result)

    def test_timing_under_200ms(self, tmp_path: Path) -> None:
        """Single J() evaluation at coarse resolution should be < 200ms."""
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(coils, bs, mesh_resolution=0.16, msh_path=msh)
        t0 = time.perf_counter()
        obj.J()
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.2, f"J() took {elapsed:.3f}s, expected < 0.2s"

    def test_dj_with_fixed_current(self, tmp_path: Path) -> None:
        """dJ() works correctly when one current is fixed (_make_base_currents)."""
        from simsopt.field import BiotSavart, coils_via_symmetries
        from simsopt.geo import create_equally_spaced_curves

        from stellcoilbench.coil_optimization._adaptive_search import (
            _make_base_currents,
        )

        ncoils, order, total_current = 4, 2, 1e5
        base_curves = create_equally_spaced_curves(
            ncoils, 2, stellsym=True, R0=1.0, R1=0.2, order=order
        )
        base_currents = _make_base_currents(total_current, ncoils)
        coils = coils_via_symmetries(base_curves, base_currents, 2, True)
        bs = BiotSavart(coils)

        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.25)
        obj = _make_objective(
            coils, bs, ncoils=ncoils, mesh_resolution=0.25, msh_path=msh
        )
        x = obj.x
        dJ = obj.dJ()
        assert isinstance(dJ, np.ndarray)
        np.testing.assert_equal(np.asarray(dJ).shape, np.asarray(x).shape)

        # Gradient via partials must have correct per-opt sizes for Derivative.__add__
        dJ_partials = obj.dJ(partials=True)
        for opt, arr in dJ_partials.data.items():
            if opt is obj:
                continue
            expected = opt.local_full_dof_size
            assert arr.shape == (expected,), (
                f"{opt.__class__.__name__}: expected size {expected}, got {arr.shape}"
            )

    def test_dj_restores_coils_on_stress_impl_exception(self, tmp_path: Path) -> None:
        """dJ() restores full_x when _evaluate_stress_impl raises during FD loop.

        Patches _evaluate_stress_impl to raise on the 3rd FD perturbation (call count 2).
        Verifies the try/finally in dJ() restores coils even when an exception occurs.
        """
        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        structural = _make_objective(coils, bs, mesh_resolution=0.16, msh_path=msh)
        x0 = np.asarray(structural.x, dtype=float).copy()

        call_count = [0]
        real_impl = structural._evaluate_stress_impl

        def mock_impl(*args: Any, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 3:
                raise RuntimeError(
                    "Simulated stress impl failure on 3rd FD perturbation"
                )
            return real_impl(*args, **kwargs)

        with patch.object(structural, "_evaluate_stress_impl", side_effect=mock_impl):
            with pytest.raises(RuntimeError, match="Simulated stress impl failure"):
                structural.dJ()

        np.testing.assert_allclose(structural.full_x, x0)

    def test_penalty_wrapper_integration(self, tmp_path: Path) -> None:
        """Objective should work with LinearPenalty and QuadraticPenalty."""
        from simsopt.objectives import QuadraticPenalty, Weight
        from stellcoilbench.coil_optimization.optimization import LinearPenalty

        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        msh = _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        obj = _make_objective(coils, bs, mesh_resolution=0.16, msh_path=msh)

        # Threshold in GPa; 1e12 Pa = 1000 GPa
        lp = LinearPenalty(obj, threshold=1000.0)
        lp_val = lp.J()
        assert isinstance(lp_val, (float, np.floating))
        assert lp_val >= 0.0

        qp = QuadraticPenalty(obj, 0.0, "max")
        qp_val = qp.J()
        assert isinstance(qp_val, (float, np.floating))
        assert qp_val >= 0.0

        # Full optimizer path: Weight * QuadraticPenalty
        wqp = Weight(2.0) * qp
        wqp_val = wqp.J()
        assert isinstance(wqp_val, (float, np.floating))
        assert wqp_val >= 0.0
        dJ = wqp.dJ()
        assert isinstance(dJ, np.ndarray)
        np.testing.assert_equal(dJ.shape, wqp.x.shape)

    def test_structural_penalty_margin_term_off_below_effective_thresh(
        self,
    ) -> None:
        """With structural_penalty_margin < 1, term turns off when J <= margin*thresh.

        Uses max(0, J - margin*thresh)²: penalty and dJ short-circuit at the same
        effective threshold. Verifies short-circuit wrapper skips dJ when J <= thresh.
        """
        from simsopt._core.derivative import Derivative

        from stellcoilbench.coil_optimization._structural_stress import (
            _StructuralStressShortCircuitWrapper,
        )

        margin = 0.9
        design_thresh_gpa = 0.1
        effective_thresh_gpa = margin * design_thresh_gpa

        class _MockStructuralObj:
            def __init__(self, j_val: float) -> None:
                self._j_val = j_val
                self._dJ_called = False

            def J(self) -> float:
                return self._j_val

            def dJ(self, **kwargs: Any) -> Derivative:
                self._dJ_called = True
                return Derivative({})

        mock_below = _MockStructuralObj(0.08)
        wrapper_below = _StructuralStressShortCircuitWrapper(
            mock_below, effective_thresh_gpa
        )
        deriv_below = wrapper_below.dJ()
        assert isinstance(deriv_below, Derivative)
        assert not mock_below._dJ_called, (
            "dJ() should be short-circuited when J <= effective_thresh"
        )

        # Use j_val well above the effective threshold so floating-point rounding
        # cannot cause (j - thresh)^2 to fall below the default short_circuit_tolerance
        # of 1e-4.  (0.10 - 0.09)^2 = 1e-4 in exact arithmetic but evaluates to
        # ~9.9999e-05 in floating-point, which is < 1e-4 and incorrectly triggers
        # the short-circuit.  Using 0.15 gives (0.15-0.09)^2 = 3.6e-3 >> 1e-4.
        mock_above = _MockStructuralObj(0.15)
        wrapper_above = _StructuralStressShortCircuitWrapper(
            mock_above, effective_thresh_gpa
        )
        wrapper_above.dJ()
        assert mock_above._dJ_called, "dJ() should be called when J > effective_thresh"

        np.testing.assert_allclose(effective_thresh_gpa, 0.09)
        assert max(0, 0.08 - effective_thresh_gpa) ** 2 < 1e-12
        np.testing.assert_allclose(max(0, 0.15 - effective_thresh_gpa) ** 2, 0.0036)

    def test_short_circuit_when_weight_microscopic(self) -> None:
        """With microscopic weight, dJ is short-circuited even when J > threshold."""
        from simsopt._core.derivative import Derivative

        from stellcoilbench.coil_optimization._structural_stress import (
            _StructuralStressShortCircuitWrapper,
        )

        class _MockStructuralObj:
            def __init__(self, j_val: float) -> None:
                self._j_val = j_val
                self._dJ_called = False

            def J(self) -> float:
                return self._j_val

            def dJ(self, **kwargs: object) -> Derivative:
                self._dJ_called = True
                return Derivative({})

        # J=0.1 GPa above threshold 0.09 -> penalty=0.0001
        # weight=1e-10 -> contribution=1e-14 < 1e-6 -> short-circuit
        mock_above = _MockStructuralObj(0.10)
        wrapper = _StructuralStressShortCircuitWrapper(
            mock_above, 0.09, weight=1e-10
        )
        deriv = wrapper.dJ()
        assert isinstance(deriv, Derivative)
        assert not mock_above._dJ_called, (
            "dJ() should be short-circuited when weight*penalty < tolerance"
        )


class TestStructuralStressGuardWrapperAndBuilder:
    """Tests for GuardWrapper and _build_structural_stress_objective."""

    def test_guard_wrapper_j_returns_penalty_when_guarded(self) -> None:
        """GuardWrapper J() returns penalty when coil-coil distance is too small."""
        from simsopt._core.derivative import Derivative

        from stellcoilbench.coil_optimization._structural_stress import (
            _StructuralStressGuardWrapper,
        )

        mock_obj = type(
            "MockObj",
            (),
            {"J": lambda self: 0.5, "dJ": lambda self, **kw: Derivative({})},
        )()
        mock_jccdist = type(
            "MockJccdist", (), {"shortest_distance": lambda self: 0.05}
        )()
        # d_cc=0.05 < safety_frac*cc_threshold=1.0*0.1 → guarded
        wrapper = _StructuralStressGuardWrapper(
            mock_obj, mock_jccdist, cc_threshold=0.1, safety_frac=1.0
        )
        j_val = wrapper.J()
        assert j_val == 10.0

    def test_guard_wrapper_dj_returns_derivative_when_guarded(self) -> None:
        """GuardWrapper dJ() returns Derivative({}) without calling underlying when guarded."""
        from simsopt._core.derivative import Derivative

        from stellcoilbench.coil_optimization._structural_stress import (
            _StructuralStressGuardWrapper,
        )

        dj_called: list[bool] = [False]

        def track_dj(self, **kwargs: Any) -> Any:
            dj_called[0] = True
            return Derivative({})

        mock_obj = type("MockObj", (), {"J": lambda self: 0.5, "dJ": track_dj})()
        mock_jccdist = type(
            "MockJccdist", (), {"shortest_distance": lambda self: 0.05}
        )()
        wrapper = _StructuralStressGuardWrapper(
            mock_obj, mock_jccdist, cc_threshold=0.1, safety_frac=1.0
        )
        deriv = wrapper.dJ()
        assert isinstance(deriv, Derivative)
        assert not dj_called[0]

    def test_build_structural_stress_objective_import_error(self) -> None:
        """When StructuralStressObjective import fails, builder returns None."""
        from stellcoilbench.coil_optimization._structural_stress import (
            _build_structural_stress_objective,
        )

        # Fake module where accessing StructuralStressObjective raises ImportError.
        class FakeModule:
            @property
            def StructuralStressObjective(self) -> None:
                raise ImportError("scikit-fem not available")

        with patch.dict(
            "sys.modules",
            {"stellcoilbench.coil_optimization._structural_objective": FakeModule()},
        ):
            result = _build_structural_stress_objective(
                coils=[],
                bs=None,
                ncoils=0,
                coil_objective_terms={"structural_stress": {}},
                thresholds={},
                out_dir=None,
            )
        assert result is None

    def test_build_structural_stress_invalid_backend(self, tmp_path: Path) -> None:
        """Builder with invalid structural_backend falls back to default without crashing."""
        from stellcoilbench.coil_optimization._structural_stress import (
            _build_structural_stress_objective,
        )

        coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
        _create_prebuilt_torus_mesh_for_coils(coils, tmp_path, mesh_size=0.16)
        coil_objective_terms = {
            "structural_stress": {},
            "structural_backend": "invalid",
        }
        thresholds = {"finite_build_width": 0.05}
        result = _build_structural_stress_objective(
            coils=coils,
            bs=bs,
            ncoils=2,
            coil_objective_terms=coil_objective_terms,
            thresholds=thresholds,
            out_dir=tmp_path,
        )
        assert result is not None
        # Should have fallen back to skfem or dolfinx and built successfully
        assert hasattr(result, "J")

    def test_msh_path_accepts_prebuilt_mesh(self, tmp_path: Path) -> None:
        """StructuralStressObjective with msh_path loads pre-built mesh and computes J()."""
        from stellcoilbench.coil_optimization._structural_mesh import (
            _generate_torus_mesh_gmsh,
        )
        from stellcoilbench.coil_optimization._structural_objective import (
            StructuralStressObjective,
        )
        from stellcoilbench.structural_analysis._common import _compute_coil_frame

        coils, bs, _, _ = _make_test_coils(ncoils=1, order=2)
        frame = _compute_coil_frame(coils[0])
        gamma = frame["gamma"]
        xy = gamma[:, :2]
        R_major = float(np.mean(np.linalg.norm(xy, axis=1)))
        if R_major < 1e-6:
            R_major = 1.2
        centroid = np.mean(gamma, axis=0)
        r_minor = 0.075  # matches width/height 0.15 / 2
        msh_file = tmp_path / "coil.msh"
        temp_msh = _generate_torus_mesh_gmsh(
            float(centroid[0]),
            float(centroid[1]),
            float(centroid[2]),
            R_major,
            r_minor,
            0.16,
        )
        shutil.copy(temp_msh, msh_file)
        temp_msh.unlink(missing_ok=True)

        obj = StructuralStressObjective(
            unique_coils=coils,
            bs=bs,
            all_coils=coils,
            width=0.15,
            height=0.15,
            msh_path=msh_file,
        )
        result = obj.J()
        assert isinstance(result, (float, np.floating))
        assert result >= 0.0
        assert np.isfinite(result)


@pytest.mark.benchmark
def test_timing_breakdown_vs_resolution(tmp_path: Path) -> None:
    """Profile phases A–G of StructuralStressObjective at 5 resolutions.

    Outputs a printed timing table and saves
    tests/artifacts/fem_benchmarks/structural_objective_timing_breakdown.png.
    """
    from stellcoilbench.structural_analysis._common import _compute_jcross_b
    from stellcoilbench.structural_analysis._skfem import (
        _compute_von_mises_skfem,
        _solve_elasticity_skfem,
    )

    coils, bs, _, _ = _make_test_coils(ncoils=2, order=2)
    width, height = 0.05, 0.05
    cross_section_area = width * height
    resolutions = [0.20, 0.16, 0.12, 0.10, 0.08]
    n_repeat = 2  # median over repeats for stability

    phase_names = [
        "A: Mesh init",
        "B: Mesh deform",
        "C: J×B body force",
        "D: Stiffness+solve",
        "E: Von Mises",
        "F: Full J()",
        "G: Full dJ()",
    ]
    records: list[dict[str, Any]] = []

    for res in resolutions:
        # Phase A: constructor (use pre-built mesh to avoid addPipe failures)
        msh = _create_prebuilt_torus_mesh_for_coils(
            coils, tmp_path, mesh_size=res, suffix=f"_{res}"
        )
        t0 = time.perf_counter()
        obj = _make_objective(
            coils,
            bs,
            width=width,
            height=height,
            mesh_resolution=res,
            fd_step=1e-5,
            msh_path=msh,
        )
        t_a = time.perf_counter() - t0
        n_nodes = obj._mesh.p.shape[1]
        n_tets = obj._mesh.t.shape[1]

        # Phase B: deform_mesh
        times_b = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            obj._deform_mesh()
            times_b.append(time.perf_counter() - t0)
        t_b = float(np.median(times_b))

        # Phase C: _compute_jcross_b at quadrature points
        e = skfem.ElementTetP1()
        elem = skfem.ElementVector(e)
        ib = skfem.Basis(obj._mesh, elem)
        gc = ib.global_coordinates()
        n_elem, n_qpts = gc.shape[1], gc.shape[2]
        q_coords = np.asarray(
            gc.transpose(1, 2, 0).reshape(-1, 3),
            dtype=np.float64,
            order="C",
        )
        times_c = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            _ = _compute_jcross_b(
                q_coords,
                obj._all_coils,
                obj._bs,
                cross_section_area,
                width=width,
                height=height,
                use_regularized=True,
            )
            times_c.append(time.perf_counter() - t0)
        t_c = float(np.median(times_c))

        # Precompute body force for Phase D (avoid recomputing J×B)
        body_force = _compute_jcross_b(
            q_coords,
            obj._all_coils,
            obj._bs,
            cross_section_area,
            width=width,
            height=height,
            use_regularized=True,
        )

        def body_force_fn(coords: np.ndarray) -> np.ndarray:
            return body_force

        # Phase D: stiffness assembly + solve (no J×B recompute)
        times_d = []
        body_arr = np.zeros((n_nodes, 3))
        for _ in range(n_repeat):
            obj._deform_mesh()
            t0 = time.perf_counter()
            _ = _solve_elasticity_skfem(
                obj._mesh,
                body_arr,
                obj._E,
                obj._nu,
                body_force_fn=body_force_fn,
                width=width,
                height=height,
            )
            times_d.append(time.perf_counter() - t0)
        t_d = float(np.median(times_d))

        # Phase E: Von Mises
        u_array = _solve_elasticity_skfem(
            obj._mesh,
            body_arr,
            obj._E,
            obj._nu,
            body_force_fn=body_force_fn,
            width=width,
            height=height,
        )
        times_e = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            _ = _compute_von_mises_skfem(obj._mesh, u_array, obj._E, obj._nu)
            times_e.append(time.perf_counter() - t0)
        t_e = float(np.median(times_e))

        # Phase F: Full J()
        times_f = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            _ = obj.J()
            times_f.append(time.perf_counter() - t0)
        t_f = float(np.median(times_f))

        # Phase G: Full dJ()
        t0 = time.perf_counter()
        _ = obj.dJ()
        t_g = time.perf_counter() - t0

        n_qpt = n_elem * n_qpts
        records.append(
            {
                "res": res,
                "nodes": n_nodes,
                "tets": n_tets,
                "n_qpt": n_qpt,
                "t_a": t_a,
                "t_b": t_b,
                "t_c": t_c,
                "t_d": t_d,
                "t_e": t_e,
                "t_f": t_f,
                "t_g": t_g,
            }
        )

    # Print table
    header = f"{'res':>6}  {'nodes':>6}  {'tets':>6}  {'n_qpt':>8}  " + "  ".join(
        f"{p[:12]:>12}" for p in phase_names
    )
    print(header)
    print("-" * len(header))
    for r in records:
        row = (
            f"{r['res']:.2f}  {r['nodes']:6d}  {r['tets']:6d}  {r['n_qpt']:8d}  "
            f"{r['t_a']:12.4f}  {r['t_b']:12.4f}  {r['t_c']:12.4f}  "
            f"{r['t_d']:12.4f}  {r['t_e']:12.4f}  {r['t_f']:12.4f}  "
            f"{r['t_g']:12.4f}"
        )
        print(row)

    # Figure
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = [1 / r["res"] for r in records]
    phases = ["t_a", "t_b", "t_c", "t_d", "t_e", "t_f", "t_g"]
    labels = [p.split(": ")[1] for p in phase_names]
    for i, (ph, lb) in enumerate(zip(phases, labels)):
        ax.semilogy(x, [r[ph] for r in records], "o-", label=lb)
    ax.set_xlabel("1/mesh_resolution")
    ax.set_ylabel("Walltime (s)")
    ax.set_title("StructuralStressObjective: timing breakdown vs resolution")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = ARTIFACTS_DIR / "structural_objective_timing_breakdown.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")

    assert records[0]["t_f"] < 1.0, "Coarsest J() should be under 1s"


@pytest.mark.benchmark
def test_qa_coil_cross_validation(qa_coils_and_bs: dict) -> None:
    """Cross-validate StructuralStressObjective.J() against run_structural_analysis.

    Uses real QA coils, same mesh file, width=height=0.15. Both pipelines use
    scikit-fem; results should agree within ~5%.
    """
    try:
        from tests.test_fem_benchmarks import _generate_coil_mesh
    except ModuleNotFoundError:
        pytest.skip(
            "tests.test_fem_benchmarks not available (module removed or not installed)"
        )

    from stellcoilbench.coil_optimization._structural_objective import (
        StructuralStressObjective,
    )
    from stellcoilbench.post_processing import get_unique_coils
    from stellcoilbench.structural_analysis import run_structural_analysis

    data = qa_coils_and_bs
    surface = data["surface"]
    coils = get_unique_coils(
        data["coils"],
        nfp=int(surface.nfp),
        stellsym=bool(surface.stellsym),
    )
    bs = data["bs"]
    output_dir = data["output_dir"] / "qa_cross_val"
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height = 0.15, 0.15
    mesh_res = 0.16
    msh_path, _ = _generate_coil_mesh(
        coils, output_dir, width, height, mesh_res, mesh_res
    )
    if msh_path is None:
        pytest.skip("Mesh generation failed (Gmsh mesh generation unavailable)")

    run_dir = output_dir / "structural_run"
    run_dir.mkdir(exist_ok=True)
    sa_result = run_structural_analysis(
        coils=coils,
        bs=bs,
        output_dir=run_dir,
        msh_path=msh_path,
        width=width,
        height=height,
    )
    ref_vm_pa = sa_result["max_von_mises_stress_Pa"]
    assert ref_vm_pa > 0, "Reference max Von Mises should be positive"
    assert np.isfinite(ref_vm_pa), "Reference max Von Mises should be finite"

    from stellcoilbench.coil_optimization._structural_objective import PA_TO_GPA

    ref_vm_gpa = ref_vm_pa * PA_TO_GPA

    obj = StructuralStressObjective(
        unique_coils=coils,
        bs=bs,
        all_coils=coils,
        width=width,
        height=height,
        msh_path=msh_path,
    )
    obj_vm = obj.evaluate_and_export_vtk(
        output_dir / "structural_stress_objective_results.vtk"
    )
    assert obj_vm > 0, "StructuralStressObjective J() should be positive"
    assert np.isfinite(obj_vm), "StructuralStressObjective J() should be finite"

    rel_err = abs(obj_vm - ref_vm_gpa) / ref_vm_gpa
    assert rel_err < 0.05, (
        f"StructuralStressObjective J()={obj_vm:.4e} GPa should agree with "
        f"run_structural_analysis max_von_mises={ref_vm_gpa:.4e} GPa within 5%, "
        f"got rel_err={rel_err:.2%}"
    )
