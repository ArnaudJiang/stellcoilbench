"""Unit tests for tools.ci_case_only_validate (case-only PR validation)."""

import sys
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))
import tools.ci_case_only_validate as m


# Path classification
_PATH_ACCEPTED = [
    ("cases/foo.yaml", True),
    ("cases/pending/bar.yaml", True),
    ("cases/other.yaml", True),
]
_PATH_REJECTED = [
    ("cases/a/b.yaml", False),
    ("cases/done/x/coils.json", False),
    ("src/foo.py", False),
    ("plasma_surfaces/x.focus", False),  # plasma accepted by different predicate
]


@pytest.mark.parametrize("path,expected", _PATH_ACCEPTED)
def test_is_accepted_case_file(path: str, expected: bool) -> None:
    assert m._is_accepted_case_file(path) == expected


@pytest.mark.parametrize("path,expected", _PATH_REJECTED)
def test_is_accepted_case_file_rejected(path: str, expected: bool) -> None:
    assert m._is_accepted_case_file(path) == expected


@pytest.mark.parametrize("path,expected", [("plasma_surfaces/x.focus", True), ("plasma_surfaces/a/b.focus", True), ("cases/x.yaml", False)])
def test_is_accepted_plasma_surface_file(path: str, expected: bool) -> None:
    assert m._is_accepted_plasma_surface_file(path) == expected


@pytest.mark.parametrize(
    "changed,bypass,ok,ncase",
    [
        ([], False, True, 0),
        (["cases/basic_tokamak.yaml"], False, True, 0),
        (["cases/basic_tokamak.yaml", "src/foo.py"], False, False, 1),
        (["cases/basic_tokamak.yaml", "src/foo.py"], True, True, 0),
        (["src/foo.py"], False, False, 1),
    ],
)
def test_validate_case_only_pr(changed: list[str], bypass: bool, ok: bool, ncase: int) -> None:
    result_ok, case_yamls, non_case, errors = m.validate_case_only_pr(changed, bypass, REPO_ROOT)
    assert result_ok == ok
    assert len(non_case) == ncase


def test_validate_case_only_pr_invalid_yaml(tmp_path: Path) -> None:
    (tmp_path / "cases").mkdir()
    (tmp_path / "plasma_surfaces").mkdir()
    bad = tmp_path / "cases" / "bad.yaml"
    bad.write_text("surface_params: {surface: nonexistent}\ncoils_params: {}\noptimizer_params: {}")
    ok, _, _, errors = m.validate_case_only_pr(["cases/bad.yaml"], False, tmp_path)
    assert not ok
    assert len(errors) > 0


def test_validate_case_only_pr_plasma_missing(tmp_path: Path) -> None:
    (tmp_path / "plasma_surfaces").mkdir()
    ok, _, _, errors = m.validate_case_only_pr(["plasma_surfaces/missing.focus"], False, tmp_path)
    assert not ok
    assert any("not found" in e for e in errors)


def test_validate_case_only_pr_plasma_ok(tmp_path: Path) -> None:
    (tmp_path / "plasma_surfaces").mkdir()
    (tmp_path / "plasma_surfaces" / "x.focus").write_text("# header")
    ok, _, _, errors = m.validate_case_only_pr(["plasma_surfaces/x.focus"], False, tmp_path)
    assert ok
    assert not errors
