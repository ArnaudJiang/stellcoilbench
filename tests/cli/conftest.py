"""Shared fixtures and helpers for CLI tests."""

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from stellcoilbench.config_scheme import ARIES_CS_MINOR_RADIUS
from tests.conftest import minimal_coils_json, write_case_yaml  # noqa: F401


@pytest.fixture
def post_process_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fixture for post-process CLI tests: coils.json, capture dict, and stubbed post_processing.

    Creates coils.json at tmp_path with "{}", a capture dict, and patches
    stellcoilbench.post_processing with a module whose run_post_processing
    captures kwargs into the dict and returns {}.

    Yields
    ------
    tuple[dict[str, Any], Path]
        (captured, coils_path) where captured is updated by each call to
        run_post_processing, and coils_path is the Path to coils.json.
    """
    coils_path = minimal_coils_json(tmp_path)
    captured: dict[str, Any] = {}

    def capture_pp(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {}

    pp_mod = types.ModuleType("stellcoilbench.post_processing")
    pp_mod.run_post_processing = capture_pp
    monkeypatch.setitem(sys.modules, "stellcoilbench.post_processing", pp_mod)
    yield captured, coils_path


class _FakeCompletedProcess:
    """Fake subprocess.CompletedProcess for testing."""

    def __init__(self, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout


def _install_stub_modules(
    monkeypatch, metrics=None, surface: str = "input.TestSurface"
) -> None:
    """Replace heavy imports with lightweight stubs so CLI tests run fast.

    Patches ``stellcoilbench.evaluate`` and ``stellcoilbench.coil_optimization``
    modules, and stubs out ``_run_sensitivity_if_configured`` to prevent the
    sensitivity analysis from attempting to load the fake coils file via
    BiotSavart (which would fail because the stub writes ``"{}"``).

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    metrics : dict | None
        Metrics dict returned by the fake ``optimize_coils``.
    surface : str
        Surface filename used in the fake case config.
    """
    if metrics is None:
        metrics = _make_integration_metrics()

    def load_case_fake(_path):
        return types.SimpleNamespace(
            surface_params={"surface": surface},
            coils_params={},
            optimizer_params={},
        )

    def optimize_coils(**kwargs):
        coils_out_path = kwargs.get("coils_out_path")
        if coils_out_path:
            Path(coils_out_path).write_text("{}")
        return metrics

    monkeypatch.setattr("stellcoilbench.cli.submit_run.load_case", load_case_fake)

    coil_mod = types.ModuleType("stellcoilbench.coil_optimization")
    coil_mod.optimize_coils = optimize_coils

    monkeypatch.setitem(
        sys.modules,
        "stellcoilbench.evaluate",
        types.ModuleType("stellcoilbench.evaluate"),
    )
    monkeypatch.setitem(sys.modules, "stellcoilbench.coil_optimization", coil_mod)

    monkeypatch.setattr(
        "stellcoilbench.cli._shared.run_sensitivity_if_configured",
        lambda **kwargs: None,
    )


@pytest.fixture
def run_case_integration_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fixture for run_case integration tests: stubs, case.yaml, submissions dir.

    Returns a factory that creates (case_yaml_path, submissions_dir). Call it
    with optional metrics, surface, username to customize:

        case_yaml, submissions_dir = run_case_integration_env()
        # or with overrides:
        case_yaml, submissions_dir = run_case_integration_env(
            metrics=_make_integration_metrics(final_squared_flux=3e-7),
            surface="wout.QA_Surface",
            username="carol",
        )
    """

    def _create(
        metrics=None,
        surface: str = "input.TestSurface",
        username: str = "alice",
    ):
        if metrics is None:
            metrics = _make_integration_metrics()
        _install_stub_modules(monkeypatch, metrics=metrics, surface=surface)
        monkeypatch.setattr(
            "stellcoilbench.cli_helpers._detect_github_username",
            lambda: username,
        )
        case_yaml = tmp_path / "case.yaml"
        write_case_yaml(case_yaml, surface=surface)
        submissions_dir = tmp_path / "submissions"
        return case_yaml, submissions_dir

    return _create


def _make_integration_metrics(**overrides):
    """Build a plausible optimisation metrics dict for CLI integration tests."""
    base = {
        "BdotN": 1e-4,
        "BdotN_over_B": 5e-5,
        "final_squared_flux": 1e-6,
        "final_min_cc_separation": 0.08,
        "final_min_cs_separation": 0.15,
        "final_total_length": 3.0,
        "final_max_curvature": 5.0,
        "final_mean_squared_curvature": 2.0,
        "final_length_per_coil": [0.75, 0.75, 0.75, 0.75],
        "_cached_thresholds": {
            "major_radius": 1.0,
            "minor_radius": 0.2,
            "a0": 8.5,
        },
        "target_B_field": 1.0,
        "iterations_used": 1,
        "timing": {
            "coil_initialization": 0.01,
            "biotsavart_setup": 0.02,
            "objective_setup": 0.03,
            "coil_optimization": 0.04,
            "save_and_metrics": 0.05,
        },
    }
    base.update(overrides)
    return base


def _make_metrics(minor_radius=0.2, target_B=1.0, major_radius=None, **overrides):
    """Helper to build a metrics dict with cached thresholds and device params.

    Uses minor_radius for L_scale (ARIES_CS_MINOR_RADIUS / minor_radius).
    major_radius defaults to ~4.5 * minor_radius (ARIES-CS aspect ratio) if not set.
    """
    if major_radius is None:
        major_radius = 4.5 * minor_radius
    a0 = ARIES_CS_MINOR_RADIUS / minor_radius
    m = {
        "target_B_field": target_B,
        "_cached_thresholds": {
            "major_radius": major_radius,
            "minor_radius": minor_radius,
            "a0": a0,
        },
    }
    m.update(overrides)
    return m


def _make_results_dict(**overrides):
    """Build a minimal results dict for autopilot submission tests."""
    base = {
        "target_B_field": 1.0,
        "final_squared_flux": 1e-6,
        "final_min_cs_separation": 0.3,
        "final_min_cc_separation": 0.08,
        "final_total_length": 10.0,
        "final_max_curvature": 4.0,
        "final_average_curvature": 2.0,
        "final_mean_squared_curvature": 5.0,
        "final_arclength_variation": 0.001,
        "final_linking_number": 0,
        "coils_linked_to_surface": True,
        "avg_BdotN_over_B": 0.0004,
        "max_BdotN_over_B": 0.002,
        "_cached_thresholds": {
            "major_radius": 1.7,
            "minor_radius": 0.29,
            "a0": 5.88,
        },
    }
    base.update(overrides)
    return base


def _make_case_config(fourier_continuation=False, **overrides):
    """Build a minimal case config dict for CLI tests."""
    cfg = {
        "surface_params": {
            "surface": "input.LandremanPaul2021_QA",
            "range": "half period",
        },
        "coils_params": {"ncoils": 4, "order": 4},
        "optimizer_params": {
            "algorithm": "augmented_lagrangian",
            "max_iterations": 100,
        },
        "coil_objective_terms": {"total_length": "l2_threshold"},
    }
    if fourier_continuation:
        cfg["fourier_continuation"] = {"enabled": True, "orders": [4, 8, 16]}
    cfg.update(overrides)
    return cfg
