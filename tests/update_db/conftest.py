"""Shared fixtures for update_db tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def make_submission_dir(
    tmp_path: Path,
    surface: str = "surface1",
    user: str = "user1",
    timestamp: str = "2024-01-01_12-00",
    results: dict | None = None,
    case_yaml: str | None = None,
) -> Path:
    """Create a submission directory under tmp_path with results.json and case.yaml.

    Args:
        tmp_path: Base temporary path (e.g., from pytest tmp_path fixture).
        surface: Surface name for path (surface/user/timestamp).
        user: User name for path.
        timestamp: Timestamp for path.
        results: Results dict for results.json; defaults to minimal valid submission.
        case_yaml: Case YAML content for case.yaml; defaults to minimal surface params.

    Returns:
        Path to the created submission directory.
    """
    submission_dir = tmp_path / surface / user / timestamp
    submission_dir.mkdir(parents=True)
    default_results = {
        "metadata": {"contact": "test_method"},
        "metrics": {"final_normalized_squared_flux": 0.001},
    }
    (submission_dir / "results.json").write_text(
        json.dumps(results if results is not None else default_results)
    )
    default_case_yaml = "surface_params:\n  surface: input.surface1\n"
    (submission_dir / "case.yaml").write_text(
        case_yaml if case_yaml is not None else default_case_yaml
    )
    return submission_dir


@pytest.fixture
def tmp_submissions_dir(tmp_path: Path) -> Path:
    """Yield a temporary submissions directory path."""
    submissions = tmp_path / "submissions"
    submissions.mkdir(parents=True)
    return submissions


@pytest.fixture
def sample_results_json() -> dict:
    """Minimal results.json-compatible dict for submission tests."""
    return {
        "metadata": {"contact": "test_method", "method_version": "v1"},
        "metrics": {"final_normalized_squared_flux": 0.001},
    }
