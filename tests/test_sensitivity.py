"""Tests for stellcoilbench.sensitivity module."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt.objectives import SquaredFlux


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_coils_and_surface():
    """Create a simple 4-coil tokamak-like setup with a circular surface."""
    ncoils = 4
    nfp = 1
    R0, R1 = 1.0, 0.5
    order = 5
    curves = create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=True,
        R0=R0,
        R1=R1,
        order=order,
    )
    coils = coils_via_symmetries(
        curves,
        [Current(1e5)] * ncoils,
        nfp,
        stellsym=True,
    )
    bs = BiotSavart(coils)

    s = SurfaceRZFourier(nfp=nfp, stellsym=True, mpol=3, ntor=3)
    s.set_rc(0, 0, R0)
    s.set_rc(1, 0, 0.1)
    s.set_zs(1, 0, 0.1)

    return bs, s


# ---------------------------------------------------------------------------
# Tests for compute_fb_perturbed
# ---------------------------------------------------------------------------


class TestComputeFbPerturbed:
    """Tests for the core Monte-Carlo evaluation function."""

    def test_returns_correct_shape(self, simple_coils_and_surface):
        from stellcoilbench.sensitivity import compute_fb_perturbed

        bs, s = simple_coils_and_surface
        n_samples = 5
        fb_vals = compute_fb_perturbed(
            bs,
            s,
            sigma=0.001,
            correlation_length_m=0.5,
            n_samples=n_samples,
            seed=42,
        )
        assert fb_vals.shape == (n_samples,)
        assert np.all(fb_vals > 0), "All f_B values must be positive"

    def test_zero_sigma_recovers_nominal(self, simple_coils_and_surface):
        from stellcoilbench.sensitivity import compute_fb_perturbed

        bs, s = simple_coils_and_surface
        nominal = SquaredFlux(s, bs).J()
        fb_vals = compute_fb_perturbed(
            bs,
            s,
            sigma=0.0,
            correlation_length_m=0.5,
            n_samples=3,
            seed=42,
        )
        np.testing.assert_allclose(fb_vals, nominal, rtol=1e-10)

    def test_larger_sigma_increases_fb(self, simple_coils_and_surface):
        from stellcoilbench.sensitivity import compute_fb_perturbed

        bs, s = simple_coils_and_surface
        fb_small = compute_fb_perturbed(
            bs,
            s,
            sigma=1e-4,
            correlation_length_m=0.5,
            n_samples=20,
            seed=42,
        )
        fb_large = compute_fb_perturbed(
            bs,
            s,
            sigma=5e-3,
            correlation_length_m=0.5,
            n_samples=20,
            seed=42,
        )
        assert np.mean(fb_large) > np.mean(fb_small)

    def test_reproducibility_with_same_seed(self, simple_coils_and_surface):
        from stellcoilbench.sensitivity import compute_fb_perturbed

        bs, s = simple_coils_and_surface
        kwargs = dict(
            sigma=0.001,
            correlation_length_m=0.5,
            n_samples=5,
            seed=123,
        )
        fb1 = compute_fb_perturbed(bs, s, **kwargs)
        fb2 = compute_fb_perturbed(bs, s, **kwargs)
        np.testing.assert_array_equal(fb1, fb2)

    def test_different_seeds_give_different_results(self, simple_coils_and_surface):
        from stellcoilbench.sensitivity import compute_fb_perturbed

        bs, s = simple_coils_and_surface
        fb1 = compute_fb_perturbed(
            bs,
            s,
            sigma=0.001,
            correlation_length_m=0.5,
            n_samples=5,
            seed=1,
        )
        fb2 = compute_fb_perturbed(
            bs,
            s,
            sigma=0.001,
            correlation_length_m=0.5,
            n_samples=5,
            seed=2,
        )
        assert not np.allclose(fb1, fb2)


# ---------------------------------------------------------------------------
# Tests for find_critical_sigma
# ---------------------------------------------------------------------------


class TestFindCriticalSigma:
    """Tests for the bisection search."""

    def test_finds_sigma_in_range(self, simple_coils_and_surface):
        from stellcoilbench.sensitivity import find_critical_sigma

        bs, s = simple_coils_and_surface
        nominal_fb = SquaredFlux(s, bs).J()

        sigma_star, history = find_critical_sigma(
            bs,
            s,
            nominal_fb=nominal_fb,
            correlation_length_m=0.5,
            n_samples=20,
            factor=2.0,
            percentile=95.0,
            sigma_min=1e-5,
            sigma_max=0.01,
            seed=42,
            bisection_tol=0.1,
            max_bisection_iter=10,
        )

        assert sigma_star >= 1e-5
        assert sigma_star <= 0.01
        assert len(history) > 0

    def test_history_records_all_steps(self, simple_coils_and_surface):
        from stellcoilbench.sensitivity import find_critical_sigma

        bs, s = simple_coils_and_surface
        nominal_fb = SquaredFlux(s, bs).J()

        _, history = find_critical_sigma(
            bs,
            s,
            nominal_fb=nominal_fb,
            correlation_length_m=0.5,
            n_samples=10,
            sigma_min=1e-5,
            sigma_max=0.005,
            seed=42,
            max_bisection_iter=5,
        )

        for step in history:
            assert step.sigma > 0
            assert step.percentile_ratio > 0
            assert step.n_samples == 10


# ---------------------------------------------------------------------------
# Tests for SensitivityResult
# ---------------------------------------------------------------------------


class TestSensitivityResult:
    """Tests for the result dataclass serialization."""

    def test_to_dict_roundtrip(self):
        from stellcoilbench.sensitivity import BisectionStep, SensitivityResult

        result = SensitivityResult(
            critical_sigma_m=0.003,
            nominal_fb=1e-8,
            factor=2.0,
            percentile=95.0,
            correlation_length_m=1.0,
            n_samples=100,
            seed=42,
            bisection_history=[
                BisectionStep(sigma=0.01, percentile_ratio=5.0, n_samples=100),
                BisectionStep(sigma=0.005, percentile_ratio=2.5, n_samples=100),
            ],
            sweep_sigmas=[0.001, 0.002, 0.003],
            sweep_p50_ratios=[1.1, 1.3, 1.6],
            sweep_p95_ratios=[1.2, 1.6, 2.0],
            sweep_mean_ratios=[1.1, 1.4, 1.7],
        )

        d = result.to_dict()
        assert d["critical_sigma_m"] == 0.003
        assert len(d["bisection_history"]) == 2
        assert len(d["final_sweep"]["sigmas"]) == 3

        serialized = json.dumps(d)
        reloaded = json.loads(serialized)
        assert reloaded["critical_sigma_m"] == 0.003


# ---------------------------------------------------------------------------
# Tests for coil arc length helper
# ---------------------------------------------------------------------------


class TestCoilArcLength:
    """Test the internal arc-length computation."""

    def test_arc_length_positive(self, simple_coils_and_surface):
        from stellcoilbench.sensitivity import _coil_arc_length

        bs, _ = simple_coils_and_surface
        for coil in bs.coils:
            L = _coil_arc_length(coil.curve)
            assert L > 0


# ---------------------------------------------------------------------------
# Tests for plot generation
# ---------------------------------------------------------------------------


class TestPlotSensitivity:
    """Test the plotting function."""

    def test_plot_creates_file(self, tmp_path):
        from stellcoilbench.sensitivity import (
            SensitivityResult,
            plot_sensitivity,
        )

        result = SensitivityResult(
            critical_sigma_m=0.003,
            nominal_fb=1e-8,
            factor=2.0,
            percentile=95.0,
            correlation_length_m=1.0,
            n_samples=50,
            seed=42,
            sweep_sigmas=[0.001, 0.002, 0.003, 0.004, 0.005],
            sweep_p50_ratios=[1.05, 1.2, 1.5, 1.9, 2.5],
            sweep_p95_ratios=[1.1, 1.5, 2.0, 3.0, 5.0],
            sweep_mean_ratios=[1.07, 1.3, 1.6, 2.2, 3.0],
        )

        out = tmp_path / "sensitivity_plot.pdf"
        ret = plot_sensitivity(result, out)
        assert ret == out
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tests for perturbed-coil VTK export
# ---------------------------------------------------------------------------


class TestExportPerturbedCoilsVtk:
    """Test VTK export of perturbed coil sets."""

    @pytest.mark.parametrize(
        "n_vtk_samples,expect_files",
        [(2, True), (0, False)],
        ids=["exports_coil_and_surface", "zero_samples_empty"],
    )
    def test_export_perturbed_coils_vtk(
        self, simple_coils_and_surface, tmp_path, n_vtk_samples, expect_files
    ):
        """export_perturbed_coils_vtk produces files when n_vtk_samples > 0, else returns empty."""
        from stellcoilbench.sensitivity import export_perturbed_coils_vtk

        bs, s = simple_coils_and_surface
        vtk_paths = export_perturbed_coils_vtk(
            bs,
            s,
            sigma=0.001,
            correlation_length_m=0.5,
            output_dir=tmp_path,
            n_vtk_samples=n_vtk_samples,
            seed=42,
        )
        if expect_files:
            coil_paths = [p for p in vtk_paths if "coils_perturbed" in str(p)]
            surf_paths = [p for p in vtk_paths if "surface_perturbed" in str(p)]
            assert len(coil_paths) == n_vtk_samples
            assert len(surf_paths) == n_vtk_samples
        else:
            assert vtk_paths == []


# ---------------------------------------------------------------------------
# Tests for full-torus surface helper
# ---------------------------------------------------------------------------


class TestMakeFullTorusSurface:
    """Test the helper that creates a full-torus plotting surface."""

    def test_full_torus_covers_all_phi(self):
        from stellcoilbench.sensitivity import _make_full_torus_surface

        s = SurfaceRZFourier(
            nfp=2,
            stellsym=True,
            mpol=2,
            ntor=2,
            quadpoints_phi=np.linspace(0, 1 / (2 * 2), 8, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 8),
        )
        s.set_rc(0, 0, 1.0)
        s.set_rc(1, 0, 0.1)
        s.set_zs(1, 0, 0.1)

        s_plot = _make_full_torus_surface(s, nphi=32, ntheta=32)

        assert s_plot.quadpoints_phi.size == 32
        assert s_plot.quadpoints_theta.size == 32
        assert s_plot.quadpoints_phi[-1] > 0.9, "phi should span close to full torus"
        np.testing.assert_allclose(s_plot.get_rc(0, 0), 1.0)
        np.testing.assert_allclose(s_plot.get_rc(1, 0), 0.1)
        np.testing.assert_allclose(s_plot.get_zs(1, 0), 0.1)


# ---------------------------------------------------------------------------
# Tests for sigma-scaling equivalence
# ---------------------------------------------------------------------------


class TestSigmaScaling:
    """Verify that unit-sigma samplers + scaling produce identical f_B values."""

    def test_scaled_vs_direct_fb(self, simple_coils_and_surface):
        from stellcoilbench.sensitivity import (
            _build_unit_samplers,
            compute_fb_perturbed,
        )

        bs, s = simple_coils_and_surface
        sigma = 0.002
        kwargs = dict(
            sigma=sigma,
            correlation_length_m=0.5,
            n_samples=5,
            seed=99,
        )

        fb_direct = compute_fb_perturbed(bs, s, **kwargs, samplers=None)

        unit_samplers = _build_unit_samplers(bs.coils, correlation_length_m=0.5)
        fb_scaled = compute_fb_perturbed(bs, s, **kwargs, samplers=unit_samplers)

        np.testing.assert_allclose(fb_direct, fb_scaled, rtol=1e-8)


# ---------------------------------------------------------------------------
# Tests for _repair_sampler_L numerical stability fix
# ---------------------------------------------------------------------------


class TestRepairSamplerL:
    """Verify that the L-matrix repair catches ill-conditioned samplers."""

    def test_well_conditioned_sampler_not_repaired(self):
        """A sampler with good conditioning should be left untouched."""
        from simsopt.geo import GaussianSampler

        from stellcoilbench.sensitivity import _repair_sampler_L

        qp = np.linspace(0, 1, 50, endpoint=False)
        sampler = GaussianSampler(qp, 1.0, 0.5, n_derivs=1)
        L_orig = sampler.L.copy()

        was_repaired = _repair_sampler_L(sampler)
        assert not was_repaired
        np.testing.assert_array_equal(sampler.L, L_orig)

    def test_defective_sampler_is_repaired(self):
        """A sampler whose LDLT gives inflated RMS should be detected and fixed."""
        from simsopt.geo import GaussianSampler

        from stellcoilbench.sensitivity import _repair_sampler_L

        qp = np.linspace(0, 1, 50, endpoint=False)
        sampler = GaussianSampler(qp, 1.0, 0.3, n_derivs=1)
        n = len(qp)

        # Corrupt L to simulate the LDLT numerical instability
        rng = np.random.default_rng(0)
        sampler.L = rng.standard_normal(sampler.L.shape) * 50.0

        LLT_bad = sampler.L @ sampler.L.T
        rms_bad = np.sqrt(np.trace(LLT_bad[:n, :n]) / n)
        assert rms_bad > 5.0, "Pre-condition: L should be grossly wrong"

        was_repaired = _repair_sampler_L(sampler)
        assert was_repaired

        LLT_fixed = sampler.L @ sampler.L.T
        rms_fixed = np.sqrt(np.trace(LLT_fixed[:n, :n]) / n)
        assert 0.5 < rms_fixed < 2.0, (
            f"Position RMS after repair = {rms_fixed}, expected ~1.0"
        )

    def test_perturbation_rms_matches_sigma(self, simple_coils_and_surface):
        """After build+repair, drawn samples should have RMS ~ sigma."""
        from simsopt.geo import PerturbationSample

        from stellcoilbench.sensitivity import _build_unit_samplers

        bs, _ = simple_coils_and_surface
        samplers = _build_unit_samplers(bs.coils, correlation_length_m=0.5)

        from numpy.random import Generator, PCG64DXSM

        sigma = 1.0
        for i, (coil, sampler) in enumerate(zip(bs.coils, samplers)):
            rms_values = []
            for seed in range(20):
                rg = Generator(PCG64DXSM(seed))
                pert = PerturbationSample(sampler, randomgen=rg)
                disp = pert._sample[0] * sigma
                rms = np.sqrt(np.mean(np.sum(disp**2, axis=1)))
                rms_values.append(rms)
            mean_rms = np.mean(rms_values)
            assert 0.1 < mean_rms < 10.0, (
                f"Coil {i}: mean displacement RMS = {mean_rms:.4f}, "
                f"expected O(sigma={sigma})"
            )


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Smoke-test the CLI command registration."""

    def test_sensitivity_help(self):
        """Verify the sensitivity command is registered and shows help."""
        from typer.testing import CliRunner

        from stellcoilbench.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["sensitivity", "--help"])
        assert result.exit_code == 0
        assert "sensitivity" in result.output.lower()
        # Check option names (substrings; Rich may truncate in narrow CI terminals)
        assert "sigma-max" in result.output or "sigma" in result.output
        assert "n-vtk" in result.output or "vtk-samples" in result.output

    def test_submit_case_has_sensitivity_flag(self):
        """Verify submit-case exposes the run-sensitivity flag and sensitivity options."""
        from typer.testing import CliRunner

        from stellcoilbench.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["submit-case", "--help"])
        assert result.exit_code == 0
        # Strip ANSI escape codes (Rich/Click inject them and break substrings)
        output = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]?", "", result.output)
        # Substrings (Rich may truncate option names in narrow CI terminals)
        assert "no-sensitivity" in output or "run-sensitivity" in output
        assert "sensitivity-n-sam" in output or "sensitivity-n-samples" in output
        assert "sensitivity-corre" in output or "sensitivity-correlation" in output
        assert "sensitivity-n-vtk" in output

    def test_sensitivity_cmd_success(self, tmp_path):
        """Test sensitivity command success path with mocked run_sensitivity_analysis."""
        from unittest.mock import MagicMock, patch
        from typer.testing import CliRunner

        from stellcoilbench.cli import app

        from tests.conftest import minimal_coils_json

        coils_path = minimal_coils_json(tmp_path)
        mock_result = MagicMock()
        mock_result.nominal_fb = 1e-6
        mock_result.critical_sigma_m = 0.001
        with patch(
            "stellcoilbench.sensitivity.run_sensitivity_analysis",
            return_value=mock_result,
        ):
            runner = CliRunner()
            result = runner.invoke(
                app,
                ["sensitivity", str(coils_path), "-o", str(tmp_path)],
            )
        assert result.exit_code == 0
        assert "Sensitivity analysis complete" in result.output
        assert "sigma*" in result.output

    def test_sensitivity_cmd_exception(self, tmp_path):
        """Test sensitivity command exits non-zero when analysis raises."""
        from unittest.mock import patch
        from typer.testing import CliRunner

        from stellcoilbench.cli import app

        from tests.conftest import minimal_coils_json

        coils_path = minimal_coils_json(tmp_path)
        with patch(
            "stellcoilbench.sensitivity.run_sensitivity_analysis",
            side_effect=ValueError("mock failure"),
        ):
            runner = CliRunner()
            result = runner.invoke(
                app,
                ["sensitivity", str(coils_path), "-o", str(tmp_path)],
            )
        assert result.exit_code != 0
