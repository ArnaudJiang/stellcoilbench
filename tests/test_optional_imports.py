"""Unit tests for optional_import and related helpers in _optional_imports."""

from __future__ import annotations

from stellcoilbench._optional_imports import optional_import


class TestOptionalImport:
    """Tests for optional_import function."""

    def test_import_existing_module_and_name(self) -> None:
        """optional_import returns the requested attribute when it exists."""
        np = optional_import("numpy", "ndarray", fallback=None)
        assert np is not None
        assert np.__module__ == "numpy"

    def test_import_missing_module_returns_fallback(self) -> None:
        """optional_import returns fallback when module does not exist."""
        result = optional_import(
            "nonexistent_module_xyz_12345", "foo", fallback="my_fallback"
        )
        assert result == "my_fallback"

    def test_import_missing_name_returns_fallback(self) -> None:
        """optional_import returns fallback when name does not exist in module."""
        result = optional_import("numpy", "nonexistent_attr_xyz", fallback=None)
        assert result is None

    def test_import_module_only_empty_name(self) -> None:
        """optional_import with empty name returns the module."""
        mod = optional_import("numpy", "", fallback=None)
        assert mod is not None
        assert mod.__name__ == "numpy"

    def test_import_module_only_missing_returns_fallback(self) -> None:
        """optional_import with empty name returns fallback when module missing."""
        result = optional_import("nonexistent_xyz_67890", "", fallback=None)
        assert result is None
