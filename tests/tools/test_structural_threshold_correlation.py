"""Unit tests for structural_threshold_correlation script."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tests.conftest import REPO_ROOT, write_case_yaml

# Import module components for testing
sys_path = str(REPO_ROOT)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from tools.structural_threshold_correlation import (
    parse_weights,
    parse_sweep_values,
    load_and_modify_case,
    collect_metrics_from_results,
    collect_von_mises_from_post_processing,
    compute_correlations,
    VM_COL,
    WEIGHT_COL,
)


@pytest.mark.parametrize(
    "weights,n_weights,expected",
    [
        (None, None, "default"),
        ("1e-9,1e-8,1e-7", None, [1e-9, 1e-8, 1e-7]),
    ],
    ids=["default", "explicit"],
)
def test_parse_weights(weights, n_weights, expected) -> None:
    """parse_weights returns default or explicit list."""
    args = MagicMock(weights=weights, n_weights=n_weights)
    result = parse_weights(args)
    if expected == "default":
        assert len(result) >= 3
        assert result[0] < result[-1]
    else:
        assert result == expected


def test_parse_sweep_values() -> None:
    """parse_sweep_values returns weight sweep (param_col, suffix, values)."""
    args = MagicMock(weights=None, n_weights=None)
    values, param_col, suffix = parse_sweep_values(args)
    assert param_col == WEIGHT_COL and suffix == "weight"
    assert len(values) >= 3


@pytest.mark.parametrize(
    "kwargs,term_checks,opt_checks",
    [
        ({"structural_stress_weight": 1e-7}, {"structural_stress": "l2", "structural_stress_weight": 1e-7}, {}),
        ({"structural_stress_weight": 1e-7, "max_iterations": 50}, {"structural_stress": "l2", "structural_stress_weight": 1e-7}, {"max_iterations": 50}),
        ({}, {"structural_stress": "l2"}, {}),
    ],
    ids=["weight", "weight_and_max_iter", "default"],
)
def test_load_and_modify_case(tmp_path: Path, kwargs: dict, term_checks: dict, opt_checks: dict) -> None:
    """load_and_modify_case sets structural_stress=l2 and threshold from case, weight if given."""
    case_path = tmp_path / "case.yaml"
    overrides = {"coil_objective_terms": {"structural_stress": "l2_threshold", "structural_stress_threshold": 0.5}}
    if "max_iterations" in kwargs:
        overrides["optimizer_params"] = {"max_iterations": 1000}
    write_case_yaml(case_path, surface="input.circular_tokamak", **overrides)
    cfg = load_and_modify_case(case_path, **kwargs)
    assert cfg.coil_objective_terms["structural_stress"] == "l2"
    assert cfg.coil_objective_terms["structural_stress_threshold"] == 0.5  # from case
    for k, v in term_checks.items():
        if k != "structural_stress":
            assert cfg.coil_objective_terms[k] == v
    for k, v in opt_checks.items():
        assert cfg.optimizer_params[k] == v


class TestCollectMetricsFromResults:
    """Tests for collect_metrics_from_results."""

    def test_extracts_known_keys(self) -> None:
        """Extracts optimization result keys."""
        results = {
            "final_squared_flux": 1e-6,
            "final_total_length": 200.0,
            "final_max_curvature": 2.5,
        }
        metrics = collect_metrics_from_results(results)
        assert metrics["final_squared_flux"] == 1e-6
        assert metrics["final_total_length"] == 200.0
        assert metrics["final_max_curvature"] == 2.5

    def test_force_from_per_coil(self) -> None:
        """final_max_max_coil_force derived from final_max_force_per_coil."""
        results = {"final_max_force_per_coil": [10.0, 15.0, 12.0]}
        metrics = collect_metrics_from_results(results)
        assert metrics["final_max_max_coil_force"] == 15.0

    def test_von_mises_from_structural_metrics(self) -> None:
        """Extracts max_von_mises_stress_Pa from nested structural_metrics."""
        results = {
            "structural_metrics": {"max_von_mises_stress_Pa": 1.2e8},
        }
        metrics = collect_metrics_from_results(results)
        assert metrics[VM_COL] == 1.2e8

    def test_von_mises_prefers_top_level(self) -> None:
        """Top-level max_von_mises_stress_Pa overrides structural_metrics."""
        results = {
            "max_von_mises_stress_Pa": 2e8,
            "structural_metrics": {"max_von_mises_stress_Pa": 1e8},
        }
        metrics = collect_metrics_from_results(results)
        assert metrics[VM_COL] == 2e8


def test_collect_von_mises_reads_value(tmp_path: Path) -> None:
    """Reads max_von_mises_stress_Pa from post_processing_results.json."""
    (tmp_path / "post_processing_results.json").write_text(
        json.dumps({"structural": {"max_von_mises_stress_Pa": 1.5e8}})
    )
    assert collect_von_mises_from_post_processing(tmp_path) == 1.5e8


@pytest.mark.parametrize(
    "setup",
    [lambda p: None, lambda p: (p / "post_processing_results.json").write_text("{}")],
    ids=["missing_file", "missing_structural"],
)
def test_collect_von_mises_returns_none(tmp_path: Path, setup) -> None:
    """Returns None when file missing or structural key absent."""
    setup(tmp_path)
    assert collect_von_mises_from_post_processing(tmp_path) is None


class TestComputeCorrelations:
    """Tests for compute_correlations."""

    def test_computes_pearson_and_spearman(self) -> None:
        """Computes both Pearson and Spearman vs reference column."""
        df = pd.DataFrame(
            {
                VM_COL: [1e8, 2e8, 3e8, 4e8, 5e8],
                "final_total_length": [200, 210, 220, 230, 240],
                "final_squared_flux": [1e-6, 2e-6, 3e-6, 4e-6, 5e-6],
            }
        )
        corr = compute_correlations(df, VM_COL)
        assert len(corr) == 2
        assert "pearson_r" in corr.columns
        assert "spearman_r" in corr.columns
        assert "n_valid" in corr.columns
        # Perfect positive correlation with length
        length_row = corr[corr["metric"] == "final_total_length"].iloc[0]
        assert length_row["pearson_r"] == pytest.approx(1.0, rel=1e-5)

    def test_skips_weight_col(self) -> None:
        """WEIGHT_COL (sweep variable) is excluded from correlation metrics."""
        df = pd.DataFrame(
            {
                VM_COL: [1e8, 2e8, 3e8, 4e8],
                WEIGHT_COL: [1e-8, 1e-7, 1e-6, 1e-5],
                "x": [1, 2, 3, 4],
            }
        )
        corr = compute_correlations(df, VM_COL)
        assert "x" in corr["metric"].tolist()
        assert WEIGHT_COL not in corr["metric"].tolist()

    def test_handles_nan(self) -> None:
        """Rows with NaN in either variable are dropped."""
        df = pd.DataFrame(
            {
                VM_COL: [1e8, 2e8, np.nan, 4e8],
                "x": [1, 2, 3, 4],
            }
        )
        corr = compute_correlations(df, VM_COL)
        assert len(corr) == 1
        assert corr.iloc[0]["n_valid"] == 3


def test_plot_correlations_creates_figure_with_grid_and_labels(tmp_path: Path) -> None:
    """plot_correlations generates scatter plot with grid and LaTeX labels."""
    pytest.importorskip("matplotlib")
    df = pd.DataFrame(
        {
            VM_COL: [1e8, 2e8, 3e8, 4e8, 5e8],
            "final_min_cc_separation": [0.2, 0.19, 0.18, 0.17, 0.16],
            "final_squared_flux": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
        }
    )
    corr = compute_correlations(df, VM_COL)
    from tools.structural_threshold_correlation import plot_correlations

    plot_correlations(df, corr, tmp_path, save_plots=True)
    scatter_path = tmp_path / "correlation_scatter.png"
    assert scatter_path.exists()
