"""
Shared assertion helpers for tests.

Use assert_errors_contain for error-list checks and assert_single_result/assert_single_item
for "exactly one" assertions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def assert_errors_contain(errors: list[str], *substrs: str) -> None:
    """Assert that each substring appears in at least one error.

    For each substr, asserts any(substr in e for e in errors).
    Raises AssertionError with a clear message listing missing substrings
    if any are not found.

    Args:
        errors: List of error messages to search.
        *substrs: One or more substrings that must each appear in at least one error.

    Raises:
        AssertionError: If any substr is not found in any error.
    """
    missing: list[str] = []
    for substr in substrs:
        if not any(substr in e for e in errors):
            missing.append(substr)
    if missing:
        msg = f"Expected substrings not found in errors: {missing}. Errors: {errors}"
        raise AssertionError(msg)


def assert_single_result(path: Path, glob_pattern: str = "**/results.json") -> Path:
    """Assert exactly one file matches glob under path and return it.

    files = list(path.rglob(glob_pattern)); assert len(files) == 1; return files[0]

    Args:
        path: Directory to search (e.g., submissions_dir).
        glob_pattern: Glob pattern for rglob; default "**/results.json".

    Returns:
        The single matching file path.

    Raises:
        AssertionError: If zero or more than one file matches.
    """
    files = list(path.rglob(glob_pattern))
    if len(files) != 1:
        msg = (
            f"Expected exactly one file matching {glob_pattern!r} under {path}, "
            f"got {len(files)}: {files}"
        )
        raise AssertionError(msg)
    return files[0]


def assert_single_item(items: list[T], name: str = "items") -> T:
    """Assert list has exactly one item and return it.

    Useful for assert len(submissions) == 1; sub = submissions[0].

    Args:
        items: List that must have exactly one element.
        name: Label for error message (e.g., "submissions").

    Returns:
        The single item.

    Raises:
        AssertionError: If len(items) != 1.
    """
    if len(items) != 1:
        msg = f"Expected exactly one {name}, got {len(items)}: {items[:5]}{'...' if len(items) > 5 else ''}"
        raise AssertionError(msg)
    return items[0]
