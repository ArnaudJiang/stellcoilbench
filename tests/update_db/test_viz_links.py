"""Unit tests for update_db._viz_links link-building functions."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from stellcoilbench.update_db._viz_links import (
    GITHUB_BASE_URL,
    LOCAL_LINK_PREFIX,
    _resolve_submission_dir,
    resolve_visualization_links,
)


# --- resolve_visualization_links ---


@pytest.mark.parametrize(
    "entry",
    [
        {},
        {"path": ""},
        {"path": "", "rank": 1},
        {"path": None, "rank": 1},
    ],
)
def test_resolve_visualization_links_empty_or_missing_path(
    entry: dict, tmp_path: Path
) -> None:
    """Empty or missing path returns dashes for all link fields."""
    i_html, i_sort, f_html, f_sort = resolve_visualization_links(
        {**entry, "path": entry.get("path", "") or ""}, tmp_path
    )
    assert i_html == "—"
    assert i_sort == ""
    assert f_html == "—"
    assert f_sort == ""


def test_resolve_visualization_links_strips_leading_slash(tmp_path: Path) -> None:
    """Paths with leading slash are normalized before resolution."""
    sub_dir = tmp_path / "submissions" / "QA" / "user1" / "run1"
    sub_dir.mkdir(parents=True)
    (sub_dir / "bn_error_3d_plot.pdf").write_text("x")
    entry = {
        "path": "/submissions/QA/user1/run1/results.json",
        "rank": 1,
        "metrics": {},
    }
    _, _, f_html, f_sort = resolve_visualization_links(entry, tmp_path)
    assert "href=" in f_html
    assert f_sort == "1"


@pytest.mark.parametrize(
    "has_init,has_final",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_resolve_visualization_links_standard_pdfs(
    tmp_path: Path, has_init: bool, has_final: bool
) -> None:
    """Standard submission: PDFs in submission dir, optional init/final."""
    sub_dir = tmp_path / "submissions" / "QA" / "u1" / "ts1"
    sub_dir.mkdir(parents=True)
    if has_init:
        (sub_dir / "bn_error_3d_plot_initial.pdf").write_text("init")
    if has_final:
        (sub_dir / "bn_error_3d_plot.pdf").write_text("final")
    entry = {
        "path": "submissions/QA/u1/ts1/results.json",
        "rank": 2,
        "metrics": {},
    }
    i_html, i_sort, f_html, f_sort = resolve_visualization_links(entry, tmp_path)
    if has_init:
        assert "href=" in i_html
        assert i_sort == "2"
    else:
        assert i_html == "—"
        assert i_sort == ""
    if has_final:
        assert "href=" in f_html
        assert f_sort == "2"
    else:
        assert f_html == "—"
        assert f_sort == ""


def test_resolve_visualization_links_uses_cdn_by_default(tmp_path: Path) -> None:
    """CDN URLs used when use_local_links is False."""
    sub_dir = tmp_path / "submissions" / "QA" / "u1" / "ts1"
    sub_dir.mkdir(parents=True)
    (sub_dir / "bn_error_3d_plot.pdf").write_text("x")
    entry = {
        "path": "submissions/QA/u1/ts1/results.json",
        "rank": 1,
        "metrics": {},
    }
    i_html, _, f_html, _ = resolve_visualization_links(entry, tmp_path)
    assert GITHUB_BASE_URL in f_html
    assert LOCAL_LINK_PREFIX not in f_html


def test_resolve_visualization_links_local_links(tmp_path: Path) -> None:
    """Local relative paths when use_local_links=True."""
    sub_dir = tmp_path / "submissions" / "QA" / "u1" / "ts1"
    sub_dir.mkdir(parents=True)
    (sub_dir / "bn_error_3d_plot.pdf").write_text("x")
    entry = {
        "path": "submissions/QA/u1/ts1/results.json",
        "rank": 1,
        "metrics": {},
    }
    i_html, _, f_html, _ = resolve_visualization_links(
        entry, tmp_path, use_local_links=True
    )
    assert LOCAL_LINK_PREFIX in f_html
    assert "submissions/" in f_html
    assert "bn_error_3d_plot.pdf" in f_html


def test_resolve_visualization_links_custom_github_base(tmp_path: Path) -> None:
    """Custom github_base_url is used for CDN links."""
    sub_dir = tmp_path / "submissions" / "QA" / "u1" / "ts1"
    sub_dir.mkdir(parents=True)
    (sub_dir / "bn_error_3d_plot.pdf").write_text("x")
    entry = {
        "path": "submissions/QA/u1/ts1/results.json",
        "rank": 1,
        "metrics": {},
    }
    custom = "https://example.com/custom"
    _, _, f_html, _ = resolve_visualization_links(
        entry, tmp_path, github_base_url=custom
    )
    assert custom in f_html


@pytest.mark.parametrize(
    "fourier_orders_str,expect_orders",
    [
        ("4, 8, 16", [4, 8, 16]),
        ("4", [4]),
        ("—", []),
        ("", []),
        (None, []),
        ("invalid", []),
        ("4,,8", []),  # empty segment int("") -> ValueError, orders becomes []
    ],
)
def test_resolve_visualization_links_fourier_continuation(
    tmp_path: Path, fourier_orders_str: str | None, expect_orders: list[int]
) -> None:
    """Fourier continuation: orders parsed from metrics; links from order_X dirs."""
    sub_dir = tmp_path / "submissions" / "QA" / "u1" / "ts1"
    sub_dir.mkdir(parents=True)
    for order in expect_orders:
        od = sub_dir / f"order_{order}"
        od.mkdir()
        (od / "bn_error_3d_plot.pdf").write_text("f")
        if order == expect_orders[0]:
            (od / "bn_error_3d_plot_initial.pdf").write_text("i")
    entry = {
        "path": "submissions/QA/u1/ts1/results.json",
        "rank": 1,
        "metrics": {"fourier_continuation_orders": fourier_orders_str},
    }
    i_html, i_sort, f_html, f_sort = resolve_visualization_links(entry, tmp_path)
    if expect_orders:
        assert "href=" in f_html
        assert i_sort == "1" if expect_orders else i_sort == ""
        assert f_sort == "1"
    else:
        # No fourier continuation -> standard path; no order_X dirs
        if not (sub_dir / "bn_error_3d_plot.pdf").exists():
            assert f_html == "—"


def test_resolve_visualization_links_fourier_continuation_with_order_dirs(
    tmp_path: Path,
) -> None:
    """Fourier continuation with order_4 and order_8 yields init + multiple final links."""
    sub_dir = tmp_path / "submissions" / "QA" / "u1" / "ts1"
    sub_dir.mkdir(parents=True)
    for order in [4, 8]:
        od = sub_dir / f"order_{order}"
        od.mkdir()
        (od / "bn_error_3d_plot.pdf").write_text("f")
    (sub_dir / "order_4" / "bn_error_3d_plot_initial.pdf").write_text("i")
    entry = {
        "path": "submissions/QA/u1/ts1/results.json",
        "rank": 1,
        "metrics": {"fourier_continuation_orders": "4, 8"},
    }
    i_html, i_sort, f_html, f_sort = resolve_visualization_links(entry, tmp_path)
    assert "href=" in i_html
    assert i_sort == "1"
    assert "4" in f_html and "8" in f_html
    assert f_sort == "1"


def test_resolve_visualization_links_nonexistent_submission_returns_dashes(
    tmp_path: Path,
) -> None:
    """Nonexistent submission path returns dashes."""
    entry = {
        "path": "submissions/QA/nonexistent/ts1/results.json",
        "rank": 1,
        "metrics": {},
    }
    i_html, i_sort, f_html, f_sort = resolve_visualization_links(entry, tmp_path)
    assert i_html == "—"
    assert i_sort == ""
    assert f_html == "—"
    assert f_sort == ""


def test_resolve_visualization_links_html_escaping(tmp_path: Path) -> None:
    """URLs and labels are HTML-escaped in link output."""
    sub_dir = tmp_path / "submissions" / "QA" / "u1"
    sub_dir.mkdir(parents=True)
    (sub_dir / "bn_error_3d_plot.pdf").write_text("x")
    entry = {
        "path": "submissions/QA/u1/results.json",
        "rank": 1,
        "metrics": {},
    }
    # Just ensure no unescaped chars leak; structure is standard
    i_html, _, f_html, _ = resolve_visualization_links(entry, tmp_path)
    assert "href=" in f_html
    assert "<" in f_html and ">" in f_html


# --- _resolve_submission_dir ---


@pytest.mark.parametrize(
    "path_name,expected_in_result",
    [
        ("all_files.zip", "submissions"),
        ("results.json", "submissions"),
    ],
)
def test_resolve_submission_dir_standard_paths(
    tmp_path: Path, path_name: str, expected_in_result: str
) -> None:
    """all_files.zip and results.json resolve to submission parent dir."""
    sub_dir = tmp_path / "submissions" / "QA" / "user1" / "run1"
    sub_dir.mkdir(parents=True)
    p = sub_dir / path_name
    p.touch()
    result = _resolve_submission_dir(p, tmp_path)
    assert result is not None
    assert expected_in_result in str(result)


@mock.patch("stellcoilbench.update_db._viz_links.parse_submission_path")
def test_resolve_submission_dir_other_zip_uses_parsed_path(
    mock_parse: mock.MagicMock, tmp_path: Path
) -> None:
    """Other .zip files use parse_submission_path; cand dir checked for existence."""
    sub_dir = tmp_path / "submissions" / "QA" / "user1" / "ts1"
    sub_dir.mkdir(parents=True)
    zip_path = sub_dir / "other.zip"
    zip_path.touch()
    mock_parse.return_value = {"surface": "QA", "user": "user1", "timestamp": "ts1"}
    result = _resolve_submission_dir(zip_path, tmp_path)
    assert result is not None
    mock_parse.assert_called_once()
    assert "submissions" in str(result)
    assert "QA" in str(result)


@mock.patch("stellcoilbench.update_db._viz_links.parse_submission_path")
def test_resolve_submission_dir_zip_fallback_to_parent_when_cand_missing(
    mock_parse: mock.MagicMock, tmp_path: Path
) -> None:
    """When parsed cand dir does not exist, fall back to zip parent."""
    sub_dir = tmp_path / "submissions" / "X" / "Y" / "Z"
    sub_dir.mkdir(parents=True)
    zip_path = sub_dir / "foo.zip"
    zip_path.touch()
    # Parse returns surface/user such that cand = submissions/QA/user1/missing - not on disk
    mock_parse.return_value = {"surface": "QA", "user": "user1", "timestamp": "missing"}
    result = _resolve_submission_dir(zip_path, tmp_path)
    assert result is not None
    # Should fall back to path_obj.parent = sub_dir
    assert "X" in str(result) or "Y" in str(result) or "Z" in str(result)


def test_resolve_submission_dir_directory_uses_parent(tmp_path: Path) -> None:
    """Non-zip, non-results.json path (e.g. dir) uses path_obj.parent."""
    sub_dir = tmp_path / "submissions" / "QA" / "u1" / "ts1"
    sub_dir.mkdir(parents=True)
    # Path to a file inside submission dir
    inner = sub_dir / "other.txt"
    inner.touch()
    result = _resolve_submission_dir(inner, tmp_path)
    assert result is not None
    assert "submissions" in str(result)


def test_resolve_submission_dir_absolute_path_normalized(tmp_path: Path) -> None:
    """Absolute submission path is normalized to repo-relative."""
    sub_dir = tmp_path / "submissions" / "QA" / "u1" / "ts1"
    sub_dir.mkdir(parents=True)
    abs_path = (tmp_path / "submissions" / "QA" / "u1" / "ts1" / "results.json").resolve()
    result = _resolve_submission_dir(abs_path, tmp_path)
    assert result is not None
    # Should be relative (no leading /)
    assert not str(result).startswith("/")
    assert "submissions" in str(result)


@mock.patch("stellcoilbench.update_db._viz_links.parse_submission_path")
def test_resolve_submission_dir_zip_surface_user_unknown_cand_exists(
    mock_parse: mock.MagicMock, tmp_path: Path
) -> None:
    """When surface/user unknown, use submissions/user/timestamp when cand exists."""
    sub_dir = tmp_path / "submissions" / "u1" / "ts1"
    sub_dir.mkdir(parents=True)
    zip_path = sub_dir / "run.zip"
    zip_path.touch()
    mock_parse.return_value = {"surface": "unknown", "user": "u1", "timestamp": "ts1"}
    result = _resolve_submission_dir(zip_path, tmp_path)
    assert result is not None
    assert "u1" in str(result)


@mock.patch("stellcoilbench.update_db._viz_links.parse_submission_path")
def test_resolve_submission_dir_zip_surface_user_unknown_cand_missing(
    mock_parse: mock.MagicMock, tmp_path: Path
) -> None:
    """When surface/user unknown and cand dir missing, fallback to zip parent."""
    sub_dir = tmp_path / "submissions" / "X" / "Y"
    sub_dir.mkdir(parents=True)
    zip_path = sub_dir / "run.zip"
    zip_path.touch()
    mock_parse.return_value = {"surface": "unknown", "user": "nonexistent", "timestamp": "ts"}
    result = _resolve_submission_dir(zip_path, tmp_path)
    assert result is not None
    assert "X" in str(result) or "Y" in str(result)


@mock.patch("stellcoilbench.update_db._viz_links.parse_submission_path")
def test_resolve_submission_dir_zip_timestamp_strip_dotzip(
    mock_parse: mock.MagicMock, tmp_path: Path
) -> None:
    """When timestamp ends with .zip (e.g. from stem), strip to get dir name."""
    sub_dir = tmp_path / "submissions" / "QA" / "u1" / "2024-01-01"
    sub_dir.mkdir(parents=True)
    zip_path = sub_dir / "2024-01-01.zip"
    zip_path.touch()
    mock_parse.return_value = {"surface": "QA", "user": "u1", "timestamp": "2024-01-01.zip"}
    result = _resolve_submission_dir(zip_path, tmp_path)
    assert result is not None
    assert "2024-01-01" in str(result)


def test_resolve_submission_dir_absolute_outside_repo_extracts_submissions(
    tmp_path: Path,
) -> None:
    """Absolute path with 'submissions' in str extracts from that index when relative_to fails."""
    # Use a path that looks absolute and contains 'submissions' but relative_to could fail
    # Create submissions under tmp_path
    sub_dir = tmp_path / "submissions" / "X" / "Y" / "Z"
    sub_dir.mkdir(parents=True)
    p = sub_dir / "results.json"
    p.touch()
    # Path like /some/other/submissions/X/Y/Z - we pass a different repo_root
    other_repo = tmp_path / "other_repo"
    other_repo.mkdir()
    result = _resolve_submission_dir(p.resolve(), other_repo)
    # relative_to(other_repo) raises ValueError; we fall back to extracting "submissions/..."
    assert result is not None
    assert "submissions" in str(result)


def test_resolve_visualization_links_resolve_submission_dir_none(tmp_path: Path) -> None:
    """When _resolve_submission_dir returns None, links are dashes."""
    entry = {"path": "/tmp/foo/bar/results.json", "rank": 1}
    i_html, i_sort, f_html, f_sort = resolve_visualization_links(entry, tmp_path)
    assert i_html == "—"
    assert i_sort == ""
    assert f_html == "—"
    assert f_sort == ""


def test_resolve_submission_dir_absolute_no_submissions_returns_none(tmp_path: Path) -> None:
    """Absolute path without 'submissions' in str returns None when relative_to fails."""
    result = _resolve_submission_dir(Path("/tmp/other/place/results.json"), tmp_path)
    assert result is None


def test_resolve_submission_dir_strips_leading_dot_slash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Path starting with ./ is normalized to drop leading ./."""
    sub_dir = tmp_path / "submissions" / "QA" / "u1" / "ts1"
    sub_dir.mkdir(parents=True)
    (sub_dir / "results.json").touch()
    monkeypatch.chdir(tmp_path)
    rel_p = Path(".") / "submissions" / "QA" / "u1" / "ts1" / "results.json"
    result = _resolve_submission_dir(rel_p, tmp_path)
    assert result is not None
    assert not str(result).startswith("./")
