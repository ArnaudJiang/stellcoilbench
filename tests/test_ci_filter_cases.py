"""Tests for tools/ci_filter_cases.py."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

# Import from tools - need to add tools to path or use importlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.ci_filter_cases import (
    case_has_successful_submission,
    discover_cases,
    main,
    normalize_yaml_content,
)


class TestNormalizeYamlContent:
    """Tests for normalize_yaml_content."""

    def test_normalize_simple_yaml(self) -> None:
        content = "a: 1\nb: 2\n"
        result = normalize_yaml_content(content)
        assert "a:" in result and "b:" in result

    def test_strip_submission_fields(self) -> None:
        content = "surface: x\nsource_case_file: cases/foo.yaml\n"
        result = normalize_yaml_content(content, strip_submission_fields=True)
        assert "source_case_file" not in result


class TestCaseHasSuccessfulSubmission:
    """Tests for case_has_successful_submission."""

    def test_nonexistent_case_returns_false(self) -> None:
        assert case_has_successful_submission("/nonexistent/case.yaml") is False

    def test_no_submissions_dir_returns_false(self, tmp_path: Path) -> None:
        case_file = tmp_path / "case.yaml"
        case_file.write_text("surface: x\n")
        assert case_has_successful_submission(str(case_file)) is False

    def test_matching_zip_returns_true(self, tmp_path: Path) -> None:
        case_file = tmp_path / "case.yaml"
        case_content = "surface: x\ncoils: {}\n"
        case_file.write_text(case_content)
        # Zip case.yaml must include source_case_file for filename matching
        zip_case_content = case_content + "source_case_file: case.yaml\n"
        submissions = tmp_path / "submissions"
        submissions.mkdir()
        zip_path = submissions / "s1" / "u1" / "sub.zip"
        zip_path.parent.mkdir(parents=True)
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("case.yaml", zip_case_content)
            zf.writestr(
                "results.json",
                json.dumps({"metadata": {}, "metrics": {}}),
            )
        with pytest.MonkeyPatch.context() as m:
            m.chdir(tmp_path)
            assert case_has_successful_submission(str(case_file)) is True


class TestDiscoverCases:
    """Tests for discover_cases."""

    def test_excludes_done_and_pending(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text("x: 1")
        (tmp_path / "done").mkdir()
        (tmp_path / "done" / "b.yaml").write_text("x: 1")
        (tmp_path / "pending").mkdir()
        (tmp_path / "pending" / "c.yaml").write_text("x: 1")
        cases = discover_cases(tmp_path)
        assert "a.yaml" in str(cases[0])
        assert len(cases) == 1


class TestMain:
    """Tests for main entry point."""

    def test_cases_dir_discovers_and_outputs_json(self, tmp_path: Path, capsys) -> None:
        (tmp_path / "case1.yaml").write_text("surface: x\n")
        with pytest.MonkeyPatch.context() as m:
            m.setattr("sys.argv", ["ci_filter_cases", "--cases-dir", str(tmp_path)])
            exit_code = main()
        assert exit_code == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "to_run" in data
        assert "already_successful" in data
        assert "case1.yaml" in str(data["to_run"][0])

    def test_stdin_mode(self, tmp_path: Path, capsys) -> None:
        case_file = tmp_path / "case.yaml"
        case_file.write_text("surface: x\n")
        with pytest.MonkeyPatch.context() as m:
            m.setattr("sys.argv", ["ci_filter_cases"])
            m.setattr("sys.stdin", io.StringIO(str(case_file) + "\n"))
            m.chdir(tmp_path)
            exit_code = main()
        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        assert "to_run" in data

    def test_github_output_mode(self, tmp_path: Path, capsys) -> None:
        (tmp_path / "case1.yaml").write_text("surface: x\n")
        gh_output = tmp_path / "github_output.txt"
        with pytest.MonkeyPatch.context() as m:
            m.setenv("GITHUB_OUTPUT", str(gh_output))
            m.setattr(
                "sys.argv",
                ["ci_filter_cases", "--cases-dir", str(tmp_path), "--github-output"],
            )
            exit_code = main()
        assert exit_code == 0
        content = gh_output.read_text()
        assert "cases_to_run_json=" in content
        assert "has_cases=" in content
        assert "has_cases=true" in content  # case1 needs to run
