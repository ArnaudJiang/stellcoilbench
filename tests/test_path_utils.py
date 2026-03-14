"""
Unit tests for path_utils.py
"""

from pathlib import Path

import pytest

from tests.conftest import minimal_coils_json, write_case_yaml
from stellcoilbench.path_utils import (
    coils_json_path_from_dir,
    dump_yaml,
    find_dir_up,
    find_file_up,
    get_surface_search_base_dirs,
    get_target_B_from_surface,
    load_yaml,
    load_yaml_safe,
    resolve_case_and_surface,
    resolve_case_yaml_path,
    resolve_surface_file_path,
    resolve_surface_path,
    surface_stem_from_filename,
)


class TestGetTargetBFromSurface:
    """Tests for get_target_B_from_surface."""

    @pytest.mark.parametrize(
        "surface_name,expected_B,additional_names",
        [
            ("muse.focus", 0.15, ["MUSE.focus"]),
            ("input.LandremanPaul2021_QA", 1.0, []),
            (
                "input.LandremanPaul2021_QH",
                5.7,
                ["LandremanPaul2021_QH_reactorScale_lowres"],
            ),
            ("cfqs_2b40", 1.0, []),
            ("HSX_QHS_mn1824", 2.0, []),
            ("W7-X_without_coil_ripple", 2.5, []),
            ("wout_schuetthenneberg_nfp2.nc", 5.7, []),
            ("unknown_surface", 5.7, ["input.Other"]),
        ],
        ids=[
            "muse",
            "landreman_paul_qa",
            "landreman_paul_qh",
            "cfqs",
            "hsx",
            "w7x",
            "schuetthenneberg",
            "default",
        ],
    )
    def test_get_target_B_from_surface(
        self, surface_name, expected_B, additional_names
    ):
        """get_target_B_from_surface returns expected B for known surfaces; default 5.7 otherwise."""
        assert get_target_B_from_surface(surface_name) == expected_B
        for name in additional_names:
            assert get_target_B_from_surface(name) == expected_B


class TestSurfaceStemFromFilename:
    """Tests for surface_stem_from_filename."""

    @pytest.mark.parametrize(
        "filename,expected_stem",
        [
            ("input.LandremanPaul2021_QA", "LandremanPaul2021_QA"),
            ("wout.LandremanPaul2021_QA", "LandremanPaul2021_QA"),
            ("LandremanPaul2021_QA.focus", "LandremanPaul2021_QA"),
            ("LandremanPaul2021_QA", "LandremanPaul2021_QA"),
        ],
        ids=["input_prefix", "wout_prefix", "focus_suffix", "no_prefix_or_suffix"],
    )
    def test_surface_stem_from_filename(self, filename, expected_stem):
        """surface_stem_from_filename strips input/wout prefix and .focus suffix."""
        assert surface_stem_from_filename(filename) == expected_stem


class TestFindPathUp:
    """Tests for find_file_up and find_dir_up."""

    def test_find_file_up(self, tmp_path):
        (tmp_path / "case.yaml").write_text("test")
        found = find_file_up(tmp_path, "case.yaml")
        assert found is not None
        assert found.name == "case.yaml"

    def test_find_file_up_not_found(self, tmp_path):
        assert find_file_up(tmp_path, "nonexistent.yaml") is None

    def test_find_dir_up(self, tmp_path):
        subdir = tmp_path / "plasma_surfaces"
        subdir.mkdir()
        found = find_dir_up(tmp_path, "plasma_surfaces")
        assert found is not None
        assert found.name == "plasma_surfaces"


class TestGetSurfaceSearchBaseDirs:
    """Tests for get_surface_search_base_dirs."""

    def test_returns_list(self):
        dirs = get_surface_search_base_dirs()
        assert isinstance(dirs, list)
        assert (
            Path("plasma_surfaces") in dirs or (Path.cwd() / "plasma_surfaces") in dirs
        )

    def test_with_case_path_dir(self, tmp_path):
        case_dir = tmp_path / "cases" / "LandremanPaul2021_QA"
        case_dir.mkdir(parents=True)
        dirs = get_surface_search_base_dirs(case_path=case_dir)
        assert case_dir in dirs


class TestResolveSurfacePath:
    """Tests for resolve_surface_path."""

    def test_finds_file(self, tmp_path):
        surf_file = tmp_path / "input.test"
        surf_file.write_text("dummy")
        found = resolve_surface_path("input.test", [tmp_path])
        assert found is not None
        assert found.name == "input.test"

    def test_case_insensitive(self, tmp_path):
        (tmp_path / "MUSE.focus").write_text("dummy")
        found = resolve_surface_path("muse.focus", [tmp_path])
        assert found is not None
        assert found.name.lower() == "muse.focus"

    def test_not_found_returns_none(self, tmp_path):
        assert resolve_surface_path("nonexistent", [tmp_path]) is None

    def test_base_dir_not_exists_skipped(self, tmp_path):
        """Base dirs that don't exist are skipped."""
        existing = tmp_path / "exists"
        existing.mkdir()
        (existing / "input.test").write_text("dummy")
        nonexistent = tmp_path / "nonexistent_dir"
        found = resolve_surface_path("input.test", [nonexistent, existing])
        assert found is not None
        assert found.name == "input.test"

    def test_case_insensitive_disabled_exact_match(self, tmp_path):
        """With case_insensitive=False, exact match works."""
        (tmp_path / "MUSE.focus").write_text("dummy")
        found = resolve_surface_path("MUSE.focus", [tmp_path], case_insensitive=False)
        assert found is not None
        assert found.name == "MUSE.focus"


class TestResolveCaseYamlPath:
    """Tests for resolve_case_yaml_path."""

    @pytest.mark.parametrize(
        "setup_type,case_path_hint,expected",
        [
            ("out_dir", None, "case.yaml"),
            ("hint_file", "case_yaml", "case.yaml"),
            ("hint_dir", "case_dir", "case.yaml"),
        ],
        ids=["from_out_dir", "from_hint_file", "from_hint_dir"],
    )
    def test_resolve_case_yaml_path_parametrized(
        self, tmp_path, setup_type, case_path_hint, expected
    ):
        """Parametrized test for resolve_case_yaml_path search strategies."""
        case_yaml = tmp_path / "case.yaml"
        write_case_yaml(case_yaml, description="test")
        if setup_type == "out_dir":
            out_dir = tmp_path
            hint = None
        elif setup_type == "hint_file":
            out_dir = tmp_path / "other"
            hint = case_yaml
        else:  # hint_dir
            case_dir = tmp_path / "case_dir"
            case_dir.mkdir()
            write_case_yaml(case_dir / "case.yaml", description="test")
            out_dir = tmp_path
            hint = case_dir
        found = resolve_case_yaml_path(out_dir, case_path_hint=hint)
        assert found is not None
        if setup_type == "hint_dir":
            assert (found.parent / "case.yaml").exists() or found.name == expected
        else:
            assert found.name == expected


class TestResolveCaseAndSurface:
    """Tests for resolve_case_and_surface and find_case_and_surface_path."""

    def test_resolve_from_case_hint_file(self, tmp_path):
        """When case_hint is a valid case.yaml file, resolve case and surface."""
        case_dir = tmp_path / "cases" / "LandremanPaul2021_QA"
        case_dir.mkdir(parents=True)
        case_yaml = case_dir / "case.yaml"
        write_case_yaml(case_yaml, surface="input.LandremanPaul2021_QA")
        (tmp_path / "plasma_surfaces").mkdir()
        surf_file = tmp_path / "plasma_surfaces" / "input.LandremanPaul2021_QA"
        surf_file.write_text("dummy")
        case_path, surf_path, data = resolve_case_and_surface(
            case_hint=case_yaml, plasma_dir=tmp_path / "plasma_surfaces"
        )
        assert case_path is not None and case_path.exists()
        assert surf_path is not None and surf_path.exists()
        assert "surface_params" in data

    def test_resolve_returns_empty_when_not_found(self, tmp_path):
        """When case cannot be found, return (None, None, {})."""
        coils = minimal_coils_json(tmp_path)
        case_path, surf_path, data = resolve_case_and_surface(
            case_hint=None, coils_path=coils, plasma_dir=tmp_path
        )
        assert case_path is None
        assert surf_path is None
        assert data == {}


class TestResolveSurfaceFilePath:
    """Tests for resolve_surface_file_path."""

    def test_resolves_from_case_yaml(self, tmp_path):
        """Resolve surface path from case YAML surface_params.surface."""
        case_yaml = tmp_path / "case.yaml"
        write_case_yaml(case_yaml, surface="input.test")
        (tmp_path / "input.test").write_text("dummy")
        found = resolve_surface_file_path(
            case_yaml_path=case_yaml, plasma_surfaces_dir=tmp_path
        )
        assert found is not None
        assert found.name == "input.test"


class TestCoilsJsonPathFromDir:
    """Tests for coils_json_path_from_dir."""

    @pytest.mark.parametrize(
        "files_to_create,expected_filename_or_none",
        [
            (
                ["coils.json", "biot_savart_optimized.json"],
                "biot_savart_optimized.json",
            ),
            (["coils.json"], "coils.json"),
            ([], None),
        ],
        ids=[
            "prefers_biot_savart",
            "fallback_to_coils_json",
            "returns_none_when_neither_exists",
        ],
    )
    def test_coils_json_path_from_dir_parametrized(
        self, tmp_path, files_to_create, expected_filename_or_none
    ):
        """Parametrized test for coils_json_path_from_dir behavior."""
        for f in files_to_create:
            (tmp_path / f).write_text("{}")
        found = coils_json_path_from_dir(tmp_path)
        if expected_filename_or_none is None:
            assert found is None
        else:
            assert found is not None
            assert found.name == expected_filename_or_none

    def test_accepts_path_or_str(self, tmp_path):
        minimal_coils_json(tmp_path)
        found = coils_json_path_from_dir(tmp_path)
        assert found is not None
        found_str = coils_json_path_from_dir(str(tmp_path))
        assert found_str is not None
        assert found_str == found


class TestLoadYaml:
    """Tests for load_yaml."""

    def test_loads_dict(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnested:\n  a: 1")
        data = load_yaml(yaml_file)
        assert data["key"] == "value"
        assert data["nested"]["a"] == 1

    def test_empty_returns_dict(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        data = load_yaml(yaml_file)
        assert data == {}

    def test_load_yaml_bytes_content(self):
        assert load_yaml(content=b"a: 1") == {"a": 1}


class TestLoadYamlSafe:
    """Tests for load_yaml_safe."""

    def test_load_yaml_safe_bad_path(self):
        assert load_yaml_safe(path=Path("/nonexistent")) is None

    def test_load_yaml_safe_invalid_yaml(self):
        assert load_yaml_safe(content="[invalid") is None


class TestDumpYaml:
    """Tests for dump_yaml."""

    def test_dump_yaml_returns_str(self):
        assert "key:" in (dump_yaml({"key": 1}) or "")

    def test_dump_yaml_returns_string_when_no_path(self):
        """When path is not given, dump_yaml returns YAML string."""
        s = dump_yaml({"a": 1})
        assert s is not None
        assert "a" in s
        assert "1" in s
