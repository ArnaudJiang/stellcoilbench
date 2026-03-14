"""
Unit tests for evaluate.py and case_loader integration.

Consolidated from test_evaluate and test_evaluate_comprehensive.
"""

from pathlib import Path

import pytest

from tests.conftest import write_case_yaml
from stellcoilbench.case_loader import load_case
from stellcoilbench.config_scheme import SubmissionMetadata
from stellcoilbench.evaluate import SubmissionResults
from stellcoilbench.path_utils import resolve_all


class TestLoadCase:
    """Tests for case_loader.load_case function."""

    def test_load_from_file(self, tmp_path: Path) -> None:
        """Test loading config from a file path."""
        case_path = tmp_path / "case.yaml"
        write_case_yaml(
            case_path,
            description="Test case",
            surface="input.circular_tokamak",
            surface_params={
                "surface": "input.circular_tokamak",
                "range": "half period",
            },
            coils_params={"ncoils": 4, "order": 16},
            optimizer_params={
                "algorithm": "l-bfgs",
                "max_iterations": 200,
                "max_iter_lag": 10,
            },
        )
        config = load_case(case_path)
        assert config.description == "Test case"
        assert config.surface_params["surface"] == "input.circular_tokamak"
        assert config.coils_params["ncoils"] == 4

    def test_load_from_directory(self, tmp_path: Path) -> None:
        """Test loading config from a directory containing case.yaml."""
        case_yaml = tmp_path / "case.yaml"
        write_case_yaml(
            case_yaml,
            description="Test case",
            surface="input.circular_tokamak",
            coils_params={"ncoils": 4},
            optimizer_params={"algorithm": "l-bfgs"},
        )
        config = load_case(tmp_path)
        assert config.description == "Test case"

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_case(Path("/nonexistent/path.yaml"))

    def test_file_not_found_dir_without_case_yaml(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised when dir has no case.yaml."""
        with pytest.raises(FileNotFoundError, match="case.yaml"):
            load_case(tmp_path)

    def test_invalid_config_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid config raises ValueError."""
        case_path = tmp_path / "case.yaml"
        case_path.write_text("description: Test case\n")  # Missing required fields
        with pytest.raises(ValueError, match="Configuration validation failed"):
            load_case(case_path)

    def test_load_case_validate_false_skips_validation(self, tmp_path: Path) -> None:
        """Test that validate=False skips validation and loads minimal config."""
        case_path = tmp_path / "case.yaml"
        write_case_yaml(
            case_path,
            description="x",
            surface_params={},
            coils_params={},
            optimizer_params={},
        )
        cfg = load_case(case_path, validate=False)
        assert cfg.description == "x"


class TestSubmissionResults:
    """Tests for SubmissionResults dataclass."""

    def test_submission_results_init(self):
        metadata = SubmissionMetadata(
            method_version="1.0", contact="test@example.com", hardware="CPU"
        )
        results = SubmissionResults(metadata=metadata, metrics={"score_primary": 0.5})
        assert results.metadata == metadata
        assert results.metrics == {"score_primary": 0.5}

    def test_submission_results_empty_metrics(self):
        metadata = SubmissionMetadata(
            method_version="1.0", contact="test@example.com", hardware="CPU"
        )
        results = SubmissionResults(metadata=metadata, metrics={})
        assert results.metrics == {}


class TestResolveCasePath:
    """Tests for path_utils.resolve_all case_yaml resolution (replaces case_loader.resolve_case_path)."""

    def _resolve_case_yaml(
        self, out_dir=None, case_path_hint=None, surface_filename=None
    ):
        """Helper: resolve case YAML using resolve_all, matching legacy resolve_case_path behavior."""
        if out_dir is not None:
            search_dir = Path(out_dir)
        elif case_path_hint is not None:
            p = (
                Path(case_path_hint)
                if isinstance(case_path_hint, str)
                else case_path_hint
            )
            search_dir = p.parent if p.is_file() else p
        else:
            return None
        resolved = resolve_all(
            search_dir, case_hint=case_path_hint, surface_filename=surface_filename
        )
        return resolved.case_yaml

    def test_resolve_from_file_hint_only(self, tmp_path: Path) -> None:
        """Test hint-only resolution when hint is a file path."""
        case_path = tmp_path / "case.yaml"
        case_path.write_text("description: x\n")
        result = self._resolve_case_yaml(case_path_hint=case_path)
        assert result is not None
        assert result.is_file()
        assert result.name.endswith(".yaml")

    def test_resolve_from_dir_hint_only(self, tmp_path: Path) -> None:
        """Test hint-only resolution when hint is a directory with case.yaml."""
        (tmp_path / "case.yaml").write_text("description: x\n")
        result = self._resolve_case_yaml(case_path_hint=tmp_path)
        assert result is not None
        assert result.is_file()
        assert result.name == "case.yaml"

    def test_resolve_case_path_str_hint(self, tmp_path: Path) -> None:
        """Test resolve_all with case_path_hint as str (not Path)."""
        (tmp_path / "case.yaml").write_text("x: 1")
        assert (
            self._resolve_case_yaml(case_path_hint=str(tmp_path))
            == tmp_path / "case.yaml"
        )

    def test_resolve_none_when_no_hint(self) -> None:
        """Test that None is returned when no hint is provided (hint-only mode)."""
        result = self._resolve_case_yaml(case_path_hint=None)
        assert result is None

    def test_resolve_none_when_dir_has_no_case_yaml(self, tmp_path: Path) -> None:
        """Test that None is returned when dir has no case.yaml (hint-only mode)."""
        result = self._resolve_case_yaml(case_path_hint=tmp_path)
        assert result is None
