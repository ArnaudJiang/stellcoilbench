"""Tests for leaderboard generation, writers, and update_database."""

from __future__ import annotations

import json
from pathlib import Path

from tests.update_db.conftest import make_submission_dir

from stellcoilbench.update_db import (
    N_TURNS_MODEL,
    build_leaderboard_json,
    build_methods_json,
    build_surface_leaderboards,
    check_reactor_constraints,
    compute_composite_score,
    update_database,
    write_markdown_leaderboard,
    write_reactor_scale_leaderboard,
    write_rst_leaderboard,
    _get_all_metrics_from_entries,
)


class TestBuildMethodsJson:
    """Tests for build_methods_json function."""

    def test_build_methods_json_empty(self, tmp_path: Path) -> None:
        submissions_root = tmp_path
        repo_root = tmp_path
        methods = build_methods_json(submissions_root, repo_root)
        assert methods == {}

    def test_build_methods_json_single_submission(self, tmp_path: Path) -> None:
        submissions_root = tmp_path.resolve()
        repo_root = tmp_path.resolve()
        make_submission_dir(
            tmp_path,
            surface="surface1",
            user="user1",
            timestamp="2024-01-01_12-00",
            results={
                "metadata": {
                    "contact": "test_method",
                    "hardware": "CPU: Test",
                    "run_date": "2024-01-01T12:00:00",
                },
                "metrics": {
                    "final_normalized_squared_flux": 0.001,
                    "final_total_length": 100.0,
                },
            },
            case_yaml="surface_params:\n  surface: input.surface1\n",
        )
        methods = build_methods_json(submissions_root, repo_root)
        assert len(methods) == 1
        method_key = "test_method:surface1:user1:2024-01-01_12-00"
        assert method_key in methods
        method_data = methods[method_key]
        assert method_data["contact"] == "user1"
        assert method_data["metrics"]["final_squared_flux"] == 0.001
        assert method_data["score_primary"] == 0.001

    def test_build_methods_json_extracts_coil_params(self, tmp_path: Path) -> None:
        submissions_root = tmp_path.resolve()
        repo_root = tmp_path.resolve()
        make_submission_dir(
            tmp_path,
            surface="surface1",
            user="user1",
            timestamp="2024-01-01_12-00",
            results={
                "metadata": {"contact": "test_method"},
                "metrics": {"final_normalized_squared_flux": 0.001},
            },
            case_yaml="""surface_params:
  surface: input.surface1
coils_params:
  ncoils: 4
  order: 16
""",
        )
        methods = build_methods_json(submissions_root, repo_root)
        method_key = "test_method:surface1:user1:2024-01-01_12-00"
        method_data = methods[method_key]
        assert method_data["metrics"]["num_coils"] == 4.0
        assert method_data["metrics"]["coil_order"] == 16.0

    def test_build_methods_json_skips_no_metrics(self, tmp_path: Path) -> None:
        submissions_root = tmp_path
        repo_root = tmp_path
        make_submission_dir(
            tmp_path,
            surface="surface1",
            user="user1",
            timestamp="2024-01-01_12-00",
            results={"metadata": {"contact": "test_method"}, "metrics": {}},
        )
        methods = build_methods_json(submissions_root, repo_root)
        assert methods == {}


class TestBuildLeaderboardJson:
    """Tests for build_leaderboard_json function."""

    def test_build_leaderboard_json_empty(self) -> None:
        leaderboard = build_leaderboard_json({})
        assert leaderboard["entries"] == []
        assert leaderboard["excluded_entries"] == []

    def test_build_leaderboard_json_single_entry(self) -> None:
        methods = {
            "method1:1.0": {
                "contact": "user1",
                "method_version": "1.0",
                "hardware": "CPU: Test",
                "run_date": "2024-01-01T12:00:00",
                "path": "submissions/surface1/user1/2024-01-01_12-00/results.json",
                "score_primary": 0.001,
                "metrics": {"final_normalized_squared_flux": 0.001},
            }
        }
        leaderboard = build_leaderboard_json(methods)
        assert len(leaderboard["entries"]) == 1
        entry = leaderboard["entries"][0]
        assert entry["rank"] == 1
        assert entry["score_primary"] == 0.001
        assert entry["contact"] == "user1"

    def test_build_leaderboard_json_sorts_ascending(self) -> None:
        methods = {
            "method1:1.0": {
                "contact": "user1",
                "method_version": "1.0",
                "hardware": "CPU: Test",
                "run_date": "2024-01-01T12:00:00",
                "path": "path1",
                "score_primary": 0.003,
                "metrics": {},
            },
            "method2:1.0": {
                "contact": "user2",
                "method_version": "1.0",
                "hardware": "CPU: Test",
                "run_date": "2024-01-01T12:00:00",
                "path": "path2",
                "score_primary": 0.001,
                "metrics": {},
            },
        }
        leaderboard = build_leaderboard_json(methods)
        assert len(leaderboard["entries"]) == 2
        assert leaderboard["entries"][0]["score_primary"] == 0.001
        assert leaderboard["entries"][1]["score_primary"] == 0.003

    def test_build_leaderboard_json_skips_no_score(self) -> None:
        methods = {
            "method1:1.0": {
                "contact": "user1",
                "method_version": "1.0",
                "hardware": "CPU: Test",
                "run_date": "2024-01-01T12:00:00",
                "path": "path1",
                "score_primary": None,
                "metrics": {},
            }
        }
        leaderboard = build_leaderboard_json(methods)
        assert leaderboard["entries"] == []


class TestGetAllMetricsFromEntries:
    """Tests for _get_all_metrics_from_entries."""

    def test_final_normalized_squared_flux_excluded(self) -> None:
        entries = [
            {
                "metrics": {
                    "num_coils": 6,
                    "final_normalized_squared_flux": 0.001,
                    "coil_order": 4,
                }
            }
        ]
        result = _get_all_metrics_from_entries(entries)
        assert "final_normalized_squared_flux" not in result
        assert "coil_order" in result
        assert "num_coils" in result

    def test_final_squared_flux_priority(self) -> None:
        entries = [
            {
                "metrics": {
                    "final_average_curvature": 1.0,
                    "final_squared_flux": 0.001,
                    "coil_order": 4,
                }
            }
        ]
        result = _get_all_metrics_from_entries(entries)
        assert result[0] == "final_squared_flux"


class TestWriteMarkdownLeaderboard:
    """Smoke tests for write_markdown_leaderboard."""

    def test_write_markdown_leaderboard(self, tmp_path: Path) -> None:
        leaderboard = {
            "entries": [
                {
                    "rank": 1,
                    "method_key": "method1",
                    "contact": "user1",
                    "method_version": "v1",
                    "score_primary": 0.01,
                    "composite_score": 1.5,
                    "run_date": "2024-01-01T12:00:00",
                    "hardware": "CPU",
                    "path": "submissions/surface/user/ts/results.json",
                    "metrics": {
                        "final_squared_flux": 0.01,
                        "final_linking_number": 0,
                    },
                }
            ]
        }
        out_md = tmp_path / "leaderboard.md"
        write_markdown_leaderboard(leaderboard, out_md)
        content = out_md.read_text()
        assert "CoilBench Leaderboard" in content or "Leaderboard" in content
        assert "Score" in content

    def test_write_markdown_leaderboard_empty_entries(self, tmp_path: Path) -> None:
        write_markdown_leaderboard({"entries": []}, tmp_path / "leaderboard.md")
        content = (tmp_path / "leaderboard.md").read_text()
        assert "_No valid submissions found._" in content


class TestWriteRstLeaderboard:
    """Smoke tests for write_rst_leaderboard."""

    def test_write_rst_leaderboard(self, tmp_path: Path) -> None:
        submissions_root = tmp_path / "submissions"
        submissions_root.mkdir(parents=True)
        submission_dir = make_submission_dir(
            submissions_root,
            surface="surface",
            user="user",
            timestamp="ts",
            case_yaml="surface_params:\n  surface: input.surface\n",
        )
        leaderboard = {
            "entries": [
                {
                    "rank": 1,
                    "method_key": "method1",
                    "contact": "user1",
                    "method_version": "v1",
                    "score_primary": 0.01,
                    "run_date": "2024-01-01T12:00:00",
                    "hardware": "CPU",
                    "path": str(submission_dir / "results.json"),
                    "metrics": {
                        "final_normalized_squared_flux": 0.01,
                        "final_linking_number": 0,
                    },
                }
            ]
        }
        surface_leaderboards = build_surface_leaderboards(
            leaderboard,
            submissions_root=submissions_root,
            plasma_surfaces_dir=tmp_path,
        )
        out_rst = tmp_path / "leaderboard.rst"
        write_rst_leaderboard(leaderboard, out_rst, surface_leaderboards)
        content = out_rst.read_text()
        assert "StellCoilBench" in content or "Leaderboard" in content

    def test_write_rst_leaderboard_empty(self, tmp_path: Path) -> None:
        out_rst = tmp_path / "leaderboard.rst"
        write_rst_leaderboard({"entries": []}, out_rst, {})
        assert out_rst.exists()
        content = out_rst.read_text()
        assert "StellCoilBench" in content


class TestWriteReactorScaleLeaderboard:
    """Smoke test for write_reactor_scale_leaderboard."""

    def test_write_reactor_scale_leaderboard(self, tmp_path: Path) -> None:
        leaderboard = {
            "entries": [
                {
                    "rank": 1,
                    "contact": "test@example.com",
                    "run_date": "2024-01-01",
                    "composite_score": 0.95,
                    "metrics": {"coils_linked_to_surface": True},
                    "reactor_scale": {"max_N_turns_per_coil": 50},
                }
            ],
            "excluded_entries": [],
        }
        out_rst = tmp_path / "reactor_scale.rst"
        write_reactor_scale_leaderboard(leaderboard, {}, out_rst)
        content = out_rst.read_text()
        assert "Reactor" in content or "Leaderboard" in content


class TestUpdateDatabase:
    """Tests for update_database function."""

    def test_update_database_empty(self, tmp_path: Path) -> None:
        repo_root = tmp_path
        submissions_root = tmp_path / "submissions"
        submissions_root.mkdir(parents=True)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True)
        update_database(
            repo_root=repo_root,
            submissions_root=submissions_root,
            docs_dir=docs_dir,
            cases_root=tmp_path / "cases",
            plasma_surfaces_dir=tmp_path / "plasma_surfaces",
        )
        leaderboard_file = docs_dir / "leaderboard.json"
        assert leaderboard_file.exists()
        leaderboard = json.loads(leaderboard_file.read_text())
        assert leaderboard["entries"] == []
        assert (docs_dir / "leaderboard.rst").exists()

    def test_update_database_with_submissions(self, tmp_path: Path) -> None:
        repo_root = tmp_path
        submissions_root = tmp_path / "submissions"
        docs_dir = tmp_path / "docs"
        plasma_surfaces_dir = tmp_path / "plasma_surfaces"
        plasma_surfaces_dir.mkdir(parents=True)
        make_submission_dir(
            submissions_root,
            surface="surf1",
            user="user1",
            timestamp="2024-01-01_12-00",
            results={
                "metadata": {
                    "contact": "user1@example.com",
                    "method_version": "v1",
                    "hardware": "CPU",
                },
                "metrics": {"final_normalized_squared_flux": 0.001},
            },
            case_yaml="surface_params:\n  surface: input.surf1\n",
        )
        update_database(
            repo_root=repo_root,
            submissions_root=submissions_root,
            docs_dir=docs_dir,
            cases_root=tmp_path / "cases",
            plasma_surfaces_dir=plasma_surfaces_dir,
        )
        leaderboard_file = docs_dir / "leaderboard.json"
        assert leaderboard_file.exists()
        leaderboard = json.loads(leaderboard_file.read_text())
        assert len(leaderboard["entries"]) == 1
        assert (docs_dir / "leaderboard.rst").exists()
        assert (docs_dir / "leaderboards" / "surf1.md").exists()


class TestCheckReactorConstraints:
    """Tests for check_reactor_constraints."""

    def test_all_pass(self) -> None:
        metrics = {
            "avg_BdotN_over_B": 5e-3,
            "final_linking_number": 0,
            "coils_linked_to_surface": True,
        }
        reactor = {
            "reactor_scale_min_cs_separation": 2.0,
            "reactor_scale_min_cc_separation": 1.0,
            "reactor_scale_total_length": 180.0,
            "reactor_scale_max_curvature": 0.8,
            "reactor_scale_mean_squared_curvature": 0.5,
            "N_turns_per_coil": [2, 3, 2],
            "finite_build_cc_clearance": 0.5,
        }
        passes, _ = check_reactor_constraints(metrics, reactor)
        assert passes is True

    def test_coils_not_linked_to_surface(self) -> None:
        metrics = {"coils_linked_to_surface": False}
        reactor = {}
        passes, _ = check_reactor_constraints(metrics, reactor)
        assert passes is False


class TestComputeCompositeScore:
    """Tests for compute_composite_score."""

    def test_coils_delinked_score_zero(self) -> None:
        score, details = compute_composite_score({"coils_linked_to_surface": False}, {})
        assert details["infeasible"]
        assert score == 0.0

    def test_empty_metrics_score_none(self) -> None:
        score, _ = compute_composite_score({}, {})
        assert score is None


def test_n_turns_model_exists() -> None:
    """N_TURNS_MODEL is defined."""
    assert N_TURNS_MODEL is not None
