"""Tests for run_structural_analysis orchestrator and config/JSON integration."""

from __future__ import annotations

import json
import pytest

from stellcoilbench.post_processing import _save_post_processing_results


@pytest.mark.parametrize(
    "msh_path,reason_substr",
    [
        (None, lambda r: "mesh" in r.lower() or "no" in r.lower()),
        ("nonexistent.msh", lambda r: "nonexistent.msh" in r),
    ],
    ids=["missing_mesh", "explicit_msh_not_found"],
)
def test_run_structural_analysis_returns_skipped(tmp_path, msh_path, reason_substr):
    """When no .msh file exists or explicit path not found, returns skipped dict."""
    from stellcoilbench.structural_analysis import (
        _DOLFINX_AVAILABLE,
        _SKFEM_AVAILABLE,
        run_structural_analysis,
    )

    if not (_DOLFINX_AVAILABLE or _SKFEM_AVAILABLE):
        pytest.skip("No FEM backend (DOLFINx or scikit-fem) available")

    path = (tmp_path / msh_path) if msh_path else None
    result = run_structural_analysis(
        coils=[],
        bs=None,
        output_dir=tmp_path,
        msh_path=path,
    )
    assert result.get("skipped") is True
    assert "reason" in result
    assert reason_substr(result.get("reason", ""))


def test_no_backend_raises_import_error(tmp_path):
    """When no backend is available, should raise ImportError."""
    from stellcoilbench.structural_analysis import (
        _DOLFINX_AVAILABLE,
        _SKFEM_AVAILABLE,
        run_structural_analysis,
    )

    if _DOLFINX_AVAILABLE or _SKFEM_AVAILABLE:
        pytest.skip("A FEM backend is available; cannot test missing-backend path")

    with pytest.raises(ImportError, match="Structural analysis requires"):
        run_structural_analysis(
            coils=[],
            bs=None,
            output_dir=tmp_path,
            msh_path=None,
        )


class TestConfigSchemeIntegration:
    """Test that PostProcessingConfig correctly handles structural fields."""

    def test_defaults(self):
        from stellcoilbench.config_scheme import PostProcessingConfig

        cfg = PostProcessingConfig()
        assert cfg.run_structural is False
        assert cfg.structural_E is None
        assert cfg.structural_nu is None

    def test_from_case_config(self):
        from stellcoilbench.config_scheme import CaseConfig, PostProcessingConfig

        case_data = {
            "description": "test",
            "surface_params": {},
            "coils_params": {},
            "optimizer_params": {},
            "post_processing_params": {
                "run_structural": True,
                "structural_E": 50e9,
                "structural_nu": 0.25,
            },
        }
        case_cfg = CaseConfig.from_dict(case_data)
        pp_cfg = PostProcessingConfig.from_case_config(case_cfg)
        assert pp_cfg.run_structural is True
        assert pp_cfg.structural_E == 50e9
        assert pp_cfg.structural_nu == 0.25

    def test_overrides(self):
        from stellcoilbench.config_scheme import CaseConfig, PostProcessingConfig

        case_data = {
            "description": "test",
            "surface_params": {},
            "coils_params": {},
            "optimizer_params": {},
            "post_processing_params": {
                "run_structural": False,
            },
        }
        case_cfg = CaseConfig.from_dict(case_data)
        pp_cfg = PostProcessingConfig.from_case_config(
            case_cfg, run_structural=True, structural_E=200e9
        )
        assert pp_cfg.run_structural is True
        assert pp_cfg.structural_E == 200e9


class TestStructuralJsonSerialization:
    """Test that structural metrics are included in post-processing JSON."""

    def test_structural_metrics_in_results_json(self, tmp_path):
        """_save_post_processing_results should include structural metrics."""
        results = {
            "BdotN": 0.01,
            "BdotN_over_B": 0.001,
            "structural_metrics": {
                "max_von_mises_stress_Pa": 1.5e8,
                "mean_von_mises_stress_Pa": 5e7,
                "max_displacement_m": 2e-4,
                "youngs_modulus_Pa": 100e9,
                "poisson_ratio": 0.3,
                "bc_type": "fixed_supports",
                "backend": "skfem",
            },
        }
        _save_post_processing_results(results, tmp_path)

        with open(tmp_path / "post_processing_results.json") as f:
            data = json.load(f)

        assert "structural" in data
        assert data["structural"]["max_von_mises_stress_Pa"] == 1.5e8
        assert data["structural"]["mean_von_mises_stress_Pa"] == 5e7
        assert data["structural"]["max_displacement_m"] == 2e-4
        assert data["structural"]["backend"] == "skfem"

    def test_no_structural_key_when_absent(self, tmp_path):
        """Without structural_metrics, the JSON should have no structural key."""
        results = {"BdotN": 0.01, "BdotN_over_B": 0.001}
        _save_post_processing_results(results, tmp_path)

        with open(tmp_path / "post_processing_results.json") as f:
            data = json.load(f)

        assert "structural" not in data
