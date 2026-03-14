"""Tests for virtual casing setup."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from stellcoilbench.coil_optimization._virtual_casing import _setup_virtual_casing


class TestSetupVirtualCasing:
    """Tests for _setup_virtual_casing."""

    def test_setup_virtual_casing_disabled(self) -> None:
        """When virtual_casing is False, returns (None, None)."""
        result = _setup_virtual_casing(
            surface_file="/some/path/input.surface1",
            surface_params={"virtual_casing": False},
            surface_resolution=8,
        )
        assert result == (None, None)

    def test_setup_virtual_casing_import_error(self) -> None:
        """When virtual_casing is enabled but package missing, raises ImportError."""
        with patch(
            "stellcoilbench.coil_optimization._virtual_casing.VIRTUAL_CASING_AVAILABLE",
            False,
        ):
            with pytest.raises(
                ImportError,
                match="virtual_casing package is not installed",
            ):
                _setup_virtual_casing(
                    surface_file="/some/path/input.surface1",
                    surface_params={"virtual_casing": True},
                    surface_resolution=8,
                )

    def test_setup_virtual_casing_no_wout_value_error(self) -> None:
        """When no wout file exists, raises ValueError."""
        with patch(
            "stellcoilbench.coil_optimization._virtual_casing.VIRTUAL_CASING_AVAILABLE",
            True,
        ):
            with patch.object(Path, "exists", return_value=False):
                with pytest.raises(
                    ValueError,
                    match="no VMEC wout file found",
                ):
                    _setup_virtual_casing(
                        surface_file="/some/path/input.surface1",
                        surface_params={"virtual_casing": True},
                        surface_resolution=8,
                    )
