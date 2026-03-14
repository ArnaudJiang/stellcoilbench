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
    _extract_primary_score,
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
