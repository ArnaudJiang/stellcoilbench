"""
Unit tests for tests.assert_helpers.
"""

import pytest

from tests.assert_helpers import (
    assert_errors_contain,
    assert_single_item,
    assert_single_result,
)


class TestAssertErrorsContain:
    """Tests for assert_errors_contain."""

    def test_single_substr_found(self) -> None:
        """Single substring found in errors passes."""
        errors = ["error: missing field X", "other error"]
        assert_errors_contain(errors, "missing field X")  # no raise

    def test_multiple_substrs_all_found(self) -> None:
        """Multiple substrings each found in at least one error passes."""
        errors = ["Missing required field: surface_params", "coils_params invalid"]
        assert_errors_contain(
            errors,
            "Missing required field: surface_params",
            "coils_params",
        )

    def test_substr_in_different_errors(self) -> None:
        """Substrings can appear in different errors."""
        errors = ["error A", "error B"]
        assert_errors_contain(errors, "A", "B")

    def test_missing_substr_raises(self) -> None:
        """Missing substring raises AssertionError with clear message."""
        errors = ["error one", "error two"]
        with pytest.raises(AssertionError) as exc_info:
            assert_errors_contain(errors, "not found")
        assert "not found" in str(exc_info.value)
        assert "Errors:" in str(exc_info.value)

    def test_multiple_missing_raises(self) -> None:
        """Multiple missing substrings are listed in message."""
        errors = ["only this"]
        with pytest.raises(AssertionError) as exc_info:
            assert_errors_contain(errors, "missing1", "missing2")
        msg = str(exc_info.value)
        assert "missing1" in msg
        assert "missing2" in msg


class TestAssertSingleResult:
    """Tests for assert_single_result."""

    def test_one_file_found_returns_path(self, tmp_path) -> None:
        """Exactly one matching file returns its path."""
        subdir = tmp_path / "a" / "b"
        subdir.mkdir(parents=True)
        results = subdir / "results.json"
        results.write_text("{}")
        got = assert_single_result(tmp_path)
        assert got == results

    def test_default_glob(self, tmp_path) -> None:
        """Default glob is **/results.json."""
        (tmp_path / "results.json").write_text("{}")
        got = assert_single_result(tmp_path)
        assert got.name == "results.json"

    def test_custom_glob(self, tmp_path) -> None:
        """Custom glob pattern works."""
        (tmp_path / "foo.zip").write_bytes(b"zip")
        got = assert_single_result(tmp_path, glob_pattern="*.zip")
        assert got.name == "foo.zip"

    def test_zero_files_raises(self, tmp_path) -> None:
        """Zero matching files raises AssertionError."""
        with pytest.raises(AssertionError) as exc_info:
            assert_single_result(tmp_path)
        assert "Expected exactly one" in str(exc_info.value)
        assert "got 0" in str(exc_info.value)

    def test_two_files_raises(self, tmp_path) -> None:
        """Two matching files raises AssertionError."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "results.json").write_text("{}")
        (tmp_path / "b" / "results.json").write_text("{}")
        with pytest.raises(AssertionError) as exc_info:
            assert_single_result(tmp_path)
        assert "got 2" in str(exc_info.value)


class TestAssertSingleItem:
    """Tests for assert_single_item."""

    def test_one_item_returns_it(self) -> None:
        """Exactly one item returns that item."""
        items = [42]
        got = assert_single_item(items)
        assert got == 42

    def test_tuple_item(self) -> None:
        """Works with tuples (e.g. submission data)."""
        items = [("key", "/path", {"x": 1})]
        got = assert_single_item(items)
        assert got == ("key", "/path", {"x": 1})

    def test_custom_name_in_message(self) -> None:
        """Custom name appears in error message."""
        with pytest.raises(AssertionError) as exc_info:
            assert_single_item([], name="submissions")
        assert "submissions" in str(exc_info.value)

    def test_zero_items_raises(self) -> None:
        """Zero items raises AssertionError."""
        with pytest.raises(AssertionError) as exc_info:
            assert_single_item([])
        assert "Expected exactly one" in str(exc_info.value)
        assert "got 0" in str(exc_info.value)

    def test_two_items_raises(self) -> None:
        """Two items raises AssertionError."""
        with pytest.raises(AssertionError) as exc_info:
            assert_single_item([1, 2])
        assert "got 2" in str(exc_info.value)
