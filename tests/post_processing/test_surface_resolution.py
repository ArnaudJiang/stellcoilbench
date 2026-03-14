"""Unit tests for post_processing._surface_resolution.

Tests _resolve_surface_path_from_hints and _resolve_surface_from_hints
with known inputs and tmp_path fixtures.
"""

from pathlib import Path
from unittest.mock import Mock, patch


from stellcoilbench.post_processing._surface_resolution import (
    _resolve_surface_from_hints,
    _resolve_surface_path_from_hints,
)


class TestResolveSurfacePathFromHints:
    """Unit tests for _resolve_surface_path_from_hints."""

    def test_returns_existing_path_when_surface_filename_exists(
        self, tmp_path: Path
    ) -> None:
        """Direct path: surface_filename points to existing file (lines 47-49)."""
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        result = _resolve_surface_path_from_hints(
            surface_filename=str(surface_file),
            case_yaml_path=None,
            plasma_dir=None,
            coils_path=None,
        )
        assert result == surface_file
        assert result is not None and result.exists()

    def test_returns_existing_path_when_surface_filename_relative_exists(
        self, tmp_path: Path
    ) -> None:
        """Direct path: surface_filename is relative and exists in cwd-like context."""
        surface_file = tmp_path / "input.landreman"
        surface_file.write_text("dummy")
        result = _resolve_surface_path_from_hints(
            surface_filename=str(surface_file),
            case_yaml_path=None,
            plasma_dir=None,
            coils_path=None,
        )
        assert result == surface_file

    def test_returns_none_when_surface_filename_empty_and_no_hints(self) -> None:
        """When surface_filename is None/empty and no hints, returns None."""
        result = _resolve_surface_path_from_hints(
            surface_filename=None,
            case_yaml_path=None,
            plasma_dir=None,
            coils_path=None,
        )
        assert result is None

    def test_returns_none_when_surface_filename_not_found_no_hints(
        self, tmp_path: Path
    ) -> None:
        """surface_filename points to non-existent file, no other hints."""
        result = _resolve_surface_path_from_hints(
            surface_filename=str(tmp_path / "nonexistent.input"),
            case_yaml_path=None,
            plasma_dir=None,
            coils_path=None,
        )
        assert result is None

    def test_uses_resolve_surface_file_path_when_direct_misses(
        self, tmp_path: Path
    ) -> None:
        """Falls through to resolve_surface_file_path (lines 51-58)."""
        plasma_dir = tmp_path / "plasma_surfaces"
        plasma_dir.mkdir()
        surface_file = plasma_dir / "input.test"
        surface_file.write_text("dummy")
        with patch(
            "stellcoilbench.post_processing._surface_resolution.resolve_surface_file_path"
        ) as mock_resolve:
            mock_resolve.return_value = surface_file
            result = _resolve_surface_path_from_hints(
                surface_filename="input.test",
                case_yaml_path=None,
                plasma_dir=plasma_dir,
                coils_path=None,
            )
            mock_resolve.assert_called_once()
            assert result == surface_file
            assert result is not None and result.exists()

    def test_returns_none_when_resolve_surface_file_path_returns_none(
        self, tmp_path: Path
    ) -> None:
        """resolve_surface_file_path returns None, no walk-up."""
        with patch(
            "stellcoilbench.post_processing._surface_resolution.resolve_surface_file_path"
        ) as mock_resolve:
            mock_resolve.return_value = None
            result = _resolve_surface_path_from_hints(
                surface_filename="input.unknown",
                case_yaml_path=None,
                plasma_dir=tmp_path,
                coils_path=None,
            )
            assert result is None

    def test_returns_none_when_resolved_path_does_not_exist(self) -> None:
        """resolve_surface_file_path returns path that doesn't exist (line 57)."""
        with patch(
            "stellcoilbench.post_processing._surface_resolution.resolve_surface_file_path"
        ) as mock_resolve:
            mock_resolve.return_value = Path("/nonexistent/input.test")
            result = _resolve_surface_path_from_hints(
                surface_filename="input.test",
                case_yaml_path=None,
                plasma_dir=None,
                coils_path=None,
            )
            assert result is None

    def test_walk_up_from_coils_same_dir(self, tmp_path: Path) -> None:
        """Walk-up fallback: file in same dir as coils.json (lines 61-74)."""
        coils_json = tmp_path / "coils.json"
        coils_json.write_text("{}")
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        with patch(
            "stellcoilbench.post_processing._surface_resolution.resolve_surface_file_path"
        ) as mock_resolve:
            mock_resolve.return_value = None
            result = _resolve_surface_path_from_hints(
                surface_filename="input.test",
                case_yaml_path=None,
                plasma_dir=None,
                coils_path=coils_json,
            )
            assert result == surface_file
            assert result is not None and result.exists()

    def test_walk_up_from_coils_plasma_surfaces_subdir(self, tmp_path: Path) -> None:
        """Walk-up fallback: file in plasma_surfaces subdir of coils dir (lines 64-70)."""
        coils_dir = tmp_path / "run" / "output"
        coils_dir.mkdir(parents=True)
        coils_json = coils_dir / "coils.json"
        coils_json.write_text("{}")
        plasma_dir = coils_dir / "plasma_surfaces"
        plasma_dir.mkdir()
        surface_file = plasma_dir / "input.LandremanPaul"
        surface_file.write_text("dummy")
        with patch(
            "stellcoilbench.post_processing._surface_resolution.resolve_surface_file_path"
        ) as mock_resolve:
            mock_resolve.return_value = None
            result = _resolve_surface_path_from_hints(
                surface_filename="input.LandremanPaul",
                case_yaml_path=None,
                plasma_dir=None,
                coils_path=coils_json,
            )
            assert result == surface_file
            assert result is not None and result.exists()

    def test_walk_up_from_coils_parent_dirs(self, tmp_path: Path) -> None:
        """Walk-up fallback: file in parent of coils dir (lines 72-74)."""
        coils_dir = tmp_path / "run" / "output" / "deep"
        coils_dir.mkdir(parents=True)
        coils_json = coils_dir / "coils.json"
        coils_json.write_text("{}")
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        with patch(
            "stellcoilbench.post_processing._surface_resolution.resolve_surface_file_path"
        ) as mock_resolve:
            mock_resolve.return_value = None
            result = _resolve_surface_path_from_hints(
                surface_filename="input.test",
                case_yaml_path=None,
                plasma_dir=None,
                coils_path=coils_json,
            )
            assert result == surface_file
            assert result is not None and result.exists()

    def test_walk_up_not_used_when_coils_path_none(self) -> None:
        """Walk-up path not executed when coils_path is None (line 60)."""
        with patch(
            "stellcoilbench.post_processing._surface_resolution.resolve_surface_file_path"
        ) as mock_resolve:
            mock_resolve.return_value = None
            result = _resolve_surface_path_from_hints(
                surface_filename="input.test",
                case_yaml_path=None,
                plasma_dir=None,
                coils_path=None,
            )
            assert result is None
            mock_resolve.assert_called_once()


class TestResolveSurfaceFromHints:
    """Unit tests for _resolve_surface_from_hints."""

    def test_extracts_filename_and_delegates(self, tmp_path: Path) -> None:
        """Extracts surface.filename and delegates to _resolve_surface_path_from_hints."""
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        mock_surface = Mock()
        mock_surface.filename = str(surface_file)
        with patch(
            "stellcoilbench.post_processing._surface_resolution._resolve_surface_path_from_hints"
        ) as mock_inner:
            mock_inner.return_value = surface_file
            result = _resolve_surface_from_hints(
                mock_surface,
                case_yaml_path=None,
                plasma_dir=None,
                coils_path=None,
            )
            mock_inner.assert_called_once_with(
                str(surface_file), None, None, None
            )
            assert result == surface_file

    def test_uses_none_when_surface_has_no_filename(self) -> None:
        """Surface without filename attribute passes None."""
        mock_surface = Mock(spec=[])  # no filename
        del mock_surface.filename
        with patch(
            "stellcoilbench.post_processing._surface_resolution._resolve_surface_path_from_hints"
        ) as mock_inner:
            mock_inner.return_value = None
            result = _resolve_surface_from_hints(
                mock_surface,
                case_yaml_path=Path("/case.yaml"),
                plasma_dir=Path("/plasma"),
                coils_path=Path("/coils.json"),
            )
            mock_inner.assert_called_once_with(
                None, Path("/case.yaml"), Path("/plasma"), Path("/coils.json")
            )
            assert result is None

    def test_getattr_fallback_for_filename(self, tmp_path: Path) -> None:
        """getattr(surface, 'filename', None) handles missing attribute."""
        mock_surface = Mock()
        # filename could be missing or falsy
        mock_surface.filename = ""
        with patch(
            "stellcoilbench.post_processing._surface_resolution._resolve_surface_path_from_hints"
        ) as mock_inner:
            mock_inner.return_value = None
            _resolve_surface_from_hints(
                mock_surface, None, None, None
            )
            mock_inner.assert_called_once_with(None, None, None, None)
