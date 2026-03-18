"""Tests for submission loading, path parsing, and visualization links."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tests.assert_helpers import assert_single_item
from tests.update_db.conftest import make_submission_dir
from stellcoilbench.update_db import _load_submissions
from stellcoilbench.update_db._viz_links import (
    _resolve_submission_dir,
    resolve_visualization_links,
)
from stellcoilbench.update_db.submission_io import (
    _extract_coil_params_from_case,
    _extract_coils_path_from_submission,
    _extract_primary_score,
    _flatten_continuation_metrics_for_reactor,
    _reactor_scale_completeness,
)


class TestLoadSubmissions:
    """Tests for _load_submissions function."""

    def test_load_submissions_empty_dir(self, tmp_path: Path) -> None:
        submissions_root = tmp_path
        submissions = list(_load_submissions(submissions_root))
        assert submissions == []

    def test_load_submissions_nonexistent_dir(self) -> None:
        submissions_root = Path("/nonexistent/directory")
        submissions = list(_load_submissions(submissions_root))
        assert submissions == []

    def test_load_submissions_single_file(self, tmp_path: Path) -> None:
        make_submission_dir(tmp_path)
        submissions_root = tmp_path
        submissions = list(_load_submissions(submissions_root))
        method_key, path, data = assert_single_item(submissions, name="submissions")
        assert method_key == "test_method:surface1:user1:2024-01-01_12-00"
        assert data["metadata"]["contact"] == "test_method"

    def test_load_submissions_skips_non_results_json(self, tmp_path: Path) -> None:
        submissions_root = tmp_path
        submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
        submission_dir.mkdir(parents=True)
        (submission_dir / "other.json").write_text(json.dumps({"test": "data"}))
        submissions = list(_load_submissions(submissions_root))
        assert len(submissions) == 0

    def test_load_submissions_invalid_json(self, tmp_path: Path) -> None:
        submissions_root = tmp_path
        submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
        submission_dir.mkdir(parents=True)
        (submission_dir / "results.json").write_text("invalid json content {")
        submissions = list(_load_submissions(submissions_root))
        assert len(submissions) == 0

    def test_load_submissions_zip_file(self, tmp_path: Path) -> None:
        submissions_root = tmp_path
        zip_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
        zip_dir.mkdir(parents=True)
        zip_path = zip_dir / "all_files.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "results.json",
                json.dumps(
                    {
                        "metadata": {"contact": "test_method"},
                        "metrics": {"final_normalized_squared_flux": 0.002},
                    }
                ),
            )
            zf.writestr("case.yaml", "surface_params:\n  surface: input.surface1\n")
        submissions = list(_load_submissions(submissions_root))
        method_key, path, data = assert_single_item(submissions, name="submissions")
        assert method_key == "test_method:surface1:user1:2024-01-01_12-00"
        assert path == zip_path
        assert data["metrics"]["final_squared_flux"] == 0.002

    def test_load_submissions_with_case_yaml_in_zip(self, tmp_path: Path) -> None:
        submissions_root = tmp_path
        submission_dir = submissions_root / "unknown" / "user1" / "timestamp"
        submission_dir.mkdir(parents=True)
        zip_path = submission_dir / "all_files.zip"
        case_yaml_content = "surface_params:\n  surface: input.test_surface\n"
        results_data = {
            "metadata": {"method_version": "test", "contact": "test@example.com"},
            "metrics": {"final_squared_flux": 0.001},
        }
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("case.yaml", case_yaml_content)
            zf.writestr("results.json", json.dumps(results_data))
        submissions = list(_load_submissions(submissions_root))
        method_key, path, data = assert_single_item(submissions, name="submissions")
        assert "test_surface" in method_key


class TestExtractPrimaryScore:
    """Tests for _extract_primary_score."""

    def test_score_primary_exists(self) -> None:
        metrics = {}
        metrics_numeric = {"score_primary": 0.42}
        result = _extract_primary_score(metrics, metrics_numeric, Path("/tmp/test"))
        assert result == 0.42

    def test_fallback_final_squared_flux(self) -> None:
        metrics = {"final_squared_flux": 1.23e-5}
        metrics_numeric = {}
        result = _extract_primary_score(metrics, metrics_numeric, Path("/tmp/test"))
        assert result == 1.23e-5
        assert metrics_numeric["score_primary"] == 1.23e-5

    def test_no_fallback_returns_none(self) -> None:
        metrics = {}
        metrics_numeric = {}
        result = _extract_primary_score(metrics, metrics_numeric, Path("/tmp/test"))
        assert result is None


class TestExtractCoilParamsFromCase:
    """Tests for _extract_coil_params_from_case."""

    def test_none_returns_empty(self) -> None:
        assert _extract_coil_params_from_case(None) == {}

    def test_order_from_coils_params(self) -> None:
        case = {"coils_params": {"order": 8}}
        result = _extract_coil_params_from_case(case)
        assert result["coil_order"] == 8.0

    def test_ncoils_from_coils_params(self) -> None:
        case = {"coils_params": {"ncoils": 12}}
        result = _extract_coil_params_from_case(case)
        assert result["num_coils"] == 12.0

    def test_fourier_continuation_orders(self) -> None:
        case = {
            "fourier_continuation": {"enabled": True, "orders": [4, 8, 16]},
        }
        result = _extract_coil_params_from_case(case)
        assert result["fourier_continuation_orders"] == "4,8,16"


class TestResolveSubmissionDir:
    """Tests for _resolve_submission_dir."""

    def test_all_files_zip_uses_parent(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "submissions" / "QA" / "user1" / "run1"
        sub_dir.mkdir(parents=True)
        zip_path = sub_dir / "all_files.zip"
        zip_path.touch()
        result = _resolve_submission_dir(zip_path, tmp_path)
        assert result is not None
        assert "submissions" in str(result)

    def test_results_json_uses_parent(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "submissions" / "QA" / "user1" / "run1"
        sub_dir.mkdir(parents=True)
        results_path = sub_dir / "results.json"
        results_path.touch()
        result = _resolve_submission_dir(results_path, tmp_path)
        assert result is not None
        assert "submissions" in str(result)


class TestResolveVisualizationLinks:
    """Tests for resolve_visualization_links."""

    def test_empty_path_returns_dashes(self, tmp_path: Path) -> None:
        entry = {"path": "", "rank": 1}
        result = resolve_visualization_links(entry, tmp_path)
        assert result == ("—", "", "—", "")

    def test_standard_pdf_links(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "submissions" / "QA" / "user1" / "run1"
        sub_dir.mkdir(parents=True)
        (sub_dir / "bn_error_3d_plot.pdf").write_text("dummy")
        (sub_dir / "bn_error_3d_plot_initial.pdf").write_text("dummy")
        entry = {
            "path": "submissions/QA/user1/run1/results.json",
            "rank": 1,
            "metrics": {},
        }
        i_html, i_sort, f_html, f_sort = resolve_visualization_links(entry, tmp_path)
        assert "href=" in i_html
        assert "href=" in f_html
        assert i_sort == "1"
        assert f_sort == "1"

    def test_submission_dir_none_returns_dashes(self, tmp_path: Path) -> None:
        entry = {"path": "/absolute/nonexistent/path/results.json", "rank": 1}
        result = resolve_visualization_links(entry, tmp_path)
        assert result == ("—", "", "—", "")


class TestFlattenContinuationMetrics:
    """Tests for _flatten_continuation_metrics_for_reactor."""

    def test_returns_metrics_unchanged_when_flat(self) -> None:
        metrics = {
            "target_B_field": 1.0,
            "_cached_thresholds": {"major_radius": 1.0, "minor_radius": 0.2},
            "final_min_cc_separation": 0.1,
        }
        result = _flatten_continuation_metrics_for_reactor(metrics)
        assert result is metrics
        assert result["target_B_field"] == 1.0

    def test_merges_last_continuation_step_when_flat_missing(self) -> None:
        metrics = {
            "continuation_results": [
                {"final_squared_flux": 0.01},
                {
                    "target_B_field": 1.0,
                    "_cached_thresholds": {"minor_radius": 0.2},
                    "final_min_cc_separation": 0.15,
                },
            ],
        }
        result = _flatten_continuation_metrics_for_reactor(metrics)
        assert result["target_B_field"] == 1.0
        assert result["final_min_cc_separation"] == 0.15
        # Only last step's keys are merged; final_squared_flux is in first step only
        assert "final_squared_flux" not in result or result["final_squared_flux"] == 0.01

    def test_returns_empty_unchanged(self) -> None:
        assert _flatten_continuation_metrics_for_reactor({}) == {}
        assert _flatten_continuation_metrics_for_reactor(None) is None


class TestReactorScaleCompleteness:
    """Tests for _reactor_scale_completeness."""

    def test_empty_returns_zero(self) -> None:
        assert _reactor_scale_completeness({}) == 0
        assert _reactor_scale_completeness(None) == 0

    def test_excludes_reference_and_error(self) -> None:
        rs = {
            "reference": {"B_field": 5.7},
            "error": "Could not determine",
            "reactor_scale_min_cc_separation": 1.0,
        }
        assert _reactor_scale_completeness(rs) == 1

    def test_counts_numeric_and_list_values(self) -> None:
        rs = {
            "reactor_scale_min_cc_separation": 1.0,
            "N_turns_per_coil": [10, 20],
            "reference": {},
        }
        assert _reactor_scale_completeness(rs) == 2


class TestExtractCoilsPathFromSubmission:
    """Tests for _extract_coils_path_from_submission."""

    def test_zip_with_coils_json_extracts_to_temp(self, tmp_path: Path) -> None:
        zip_dir = tmp_path / "surface1" / "user1" / "run1"
        zip_dir.mkdir(parents=True)
        zip_path = zip_dir / "all_files.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("coils.json", '{"coils": []}')
            zf.writestr("results.json", json.dumps({"metrics": {}}))
        coils_path, cleanup = _extract_coils_path_from_submission(zip_path, tmp_path)
        assert coils_path is not None
        assert coils_path.exists()
        assert coils_path.read_text() == '{"coils": []}'
        cleanup()
        assert not coils_path.exists()

    def test_zip_without_coils_returns_none(self, tmp_path: Path) -> None:
        zip_dir = tmp_path / "surface1" / "user1" / "run1"
        zip_dir.mkdir(parents=True)
        zip_path = zip_dir / "all_files.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("results.json", json.dumps({"metrics": {}}))
        coils_path, cleanup = _extract_coils_path_from_submission(zip_path, tmp_path)
        assert coils_path is None
        cleanup()

    def test_dir_with_coils_returns_existing_path(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "surface1" / "user1" / "run1"
        sub_dir.mkdir(parents=True)
        coils_file = sub_dir / "coils.json"
        coils_file.write_text("{}")
        results_path = sub_dir / "results.json"
        results_path.write_text(json.dumps({"metrics": {}}))
        coils_path, cleanup = _extract_coils_path_from_submission(results_path, tmp_path)
        assert coils_path == coils_file
        assert coils_path.exists()
        cleanup()
        assert coils_file.exists()
