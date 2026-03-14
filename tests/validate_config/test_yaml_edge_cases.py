"""
Unit tests for validate_case_yaml_file and related YAML edge cases.
"""

from pathlib import Path

import pytest

from tests.assert_helpers import assert_errors_contain
from stellcoilbench.validate_config import validate_case_yaml_file
from tests.validate_config.conftest import _YAML_INVALID_CASE_PARAMS


class TestValidateCaseYamlFile:
    """Tests for validate_case_yaml_file function."""

    def test_valid_yaml_file(self, tmp_path: Path):
        """Test validation of a valid YAML file."""
        case_path = tmp_path / "case.yaml"
        case_path.write_text("""description: Test case
surface_params:
  surface: input.LandremanPaul2021_QA
  range: half period
coils_params:
  ncoils: 4
  order: 16
optimizer_params:
  algorithm: l-bfgs
  max_iterations: 200
""")
        errors = validate_case_yaml_file(case_path)
        assert errors == []

    def test_invalid_yaml_syntax(self, tmp_path: Path):
        """Test validation with invalid YAML syntax."""
        case_path = tmp_path / "case.yaml"
        case_path.write_text("invalid: yaml: syntax: [")
        errors = validate_case_yaml_file(case_path)
        assert len(errors) > 0
        assert_errors_contain(errors, "YAML parsing error")

    def test_missing_fields_in_file(self, tmp_path: Path):
        """Test validation with missing required fields in file."""
        case_path = tmp_path / "case.yaml"
        case_path.write_text("description: Test case\n")
        errors = validate_case_yaml_file(case_path)
        assert len(errors) > 0
        assert_errors_contain(errors, "Missing required field")

    def test_empty_file(self, tmp_path: Path):
        """Test validation of empty file."""
        case_path = tmp_path / "case.yaml"
        case_path.write_text("")
        errors = validate_case_yaml_file(case_path)
        assert len(errors) > 0
        assert any("File is empty" in e or "no valid YAML" in e for e in errors)

    def test_non_dict_root_in_file(self, tmp_path: Path):
        """Test validation when YAML root is a list, not a dict."""
        case_path = tmp_path / "case.yaml"
        case_path.write_text("- item1\n- item2\n")
        errors = validate_case_yaml_file(case_path)
        assert len(errors) > 0
        assert_errors_contain(errors, "Root element must be a dictionary")

    def test_file_not_found(self):
        """Test validation when file does not exist."""
        errors = validate_case_yaml_file(Path("/nonexistent/file.yaml"))
        assert len(errors) > 0
        assert_errors_contain(errors, "Error reading file")


class TestValidateCaseYamlFileInvalidCases:
    """Tests for validate_case_yaml_file with various invalid case files."""

    @pytest.mark.parametrize("yaml_content,expected_substrs", _YAML_INVALID_CASE_PARAMS)
    def test_validate_case_yaml_file_invalid(
        self, tmp_path, yaml_content, expected_substrs
    ):
        """Parametrized invalid YAML case files produce expected error substrings."""
        case = tmp_path / "case.yaml"
        case.write_text(yaml_content)
        errors = validate_case_yaml_file(case)
        assert len(errors) > 0
        assert_errors_contain(errors, *expected_substrs)

    def test_multiple_errors(self, tmp_path):
        """Case with multiple validation errors returns all of them."""
        case = tmp_path / "multi_error.yaml"
        case.write_text("""description: Multiple errors
surface_params:
  surface: nonexistent_xyz
  range: bad_range
coils_params:
  ncoils: -1
  order: 0
optimizer_params:
  max_iterations: 0
""")
        errors = validate_case_yaml_file(case)
        assert len(errors) >= 2
        error_str = " ".join(errors).lower()
        assert "range" in error_str or "ncoils" in error_str or "order" in error_str


class TestValidateChangedCaseYamlFiles:
    """Tests for workflow-style validation of a list of changed case YAML files."""

    def _validate_changed_files(self, changed_paths: list[Path]) -> list[str]:
        """Mirror the workflow's validation logic."""
        all_errors: list[str] = []
        for p in changed_paths:
            if not p.suffix == ".yaml" or not p.exists():
                continue
            errs = validate_case_yaml_file(p)
            all_errors.extend(errs)
        return all_errors

    def test_valid_single_file(self, tmp_path):
        """Single valid case file passes."""
        case = tmp_path / "good.yaml"
        case.write_text("""description: Good
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
""")
        errors = self._validate_changed_files([case])
        assert errors == []

    def test_invalid_single_file(self, tmp_path):
        """Single invalid case file returns errors."""
        case = tmp_path / "bad.yaml"
        case.write_text("""description: Bad
surface_params:
  surface: nonexistent_xyz
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  max_iterations: 0
""")
        errors = self._validate_changed_files([case])
        assert len(errors) > 0

    def test_multiple_valid_files(self, tmp_path):
        """Multiple valid case files all pass."""
        case1 = tmp_path / "a.yaml"
        case2 = tmp_path / "b.yaml"
        case1.write_text("""description: A
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
""")
        case2.write_text("""description: B
surface_params:
  surface: muse.focus
coils_params:
  ncoils: 4
  order: 8
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 100
""")
        errors = self._validate_changed_files([case1, case2])
        assert errors == []

    def test_non_yaml_paths_skipped(self, tmp_path):
        """Non-.yaml paths are skipped."""
        case = tmp_path / "case.yaml"
        case.write_text("""description: Good
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
""")
        other = tmp_path / "readme.md"
        other.write_text("# Readme")
        errors = self._validate_changed_files([case, other])
        assert errors == []

    def test_nonexistent_path_skipped(self, tmp_path):
        """Nonexistent paths are skipped."""
        case = tmp_path / "exists.yaml"
        case.write_text("""description: Good
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: L-BFGS-B
  max_iterations: 200
""")
        errors = self._validate_changed_files([case, tmp_path / "missing.yaml"])
        assert errors == []

    def test_empty_list(self):
        """Empty list returns no errors."""
        errors = self._validate_changed_files([])
        assert errors == []
