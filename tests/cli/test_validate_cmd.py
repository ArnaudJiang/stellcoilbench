"""Tests for validate-config CLI command."""

import pytest
from typer.testing import CliRunner

from stellcoilbench.cli import app
from tests.conftest import write_case_yaml


runner = CliRunner()


def test_validate_config_valid_case(tmp_path):
    """Valid case YAML with surface in custom plasma_surfaces_dir passes."""
    surfaces_dir = tmp_path / "plasma_surfaces"
    surfaces_dir.mkdir()
    (surfaces_dir / "input.TestSurface").write_text("# minimal")
    case_path = tmp_path / "case.yaml"
    write_case_yaml(case_path, surface="input.TestSurface")

    result = runner.invoke(
        app,
        ["validate-config", str(case_path), "--plasma-surfaces-dir", str(surfaces_dir)],
    )
    assert result.exit_code == 0
    assert "Configuration valid." in result.output


def test_validate_config_invalid_missing_field(tmp_path):
    """Invalid case with missing required field prints errors and exits 1."""
    case_path = tmp_path / "bad.yaml"
    case_path.write_text("description: only description\n")

    result = runner.invoke(app, ["validate-config", str(case_path)])
    assert result.exit_code == 1
    assert "Missing required field" in result.output
    assert "surface_params" in result.output or "coils_params" in result.output


def test_validate_config_nonexistent_file():
    """Nonexistent file path exits with error."""
    result = runner.invoke(app, ["validate-config", "/nonexistent/case.yaml"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "File not found" in result.output


@pytest.mark.parametrize(
    "with_path,extra_keys",
    [
        (False, ["surface_params", "coils_params", "optimizer_params"]),
        (True, ["surface_params"]),
    ],
    ids=["schema_only", "schema_with_path"],
)
def test_validate_config_schema(tmp_path, with_path, extra_keys):
    """--schema emits JSON schema and exits 0 (path is ignored when --schema used)."""
    if with_path:
        write_case_yaml(tmp_path / "case.yaml")
        args = ["validate-config", str(tmp_path / "case.yaml"), "--schema"]
    else:
        args = ["validate-config", "--schema"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    for key in extra_keys:
        assert key in result.output


def test_validate_config_missing_path_without_schema():
    """Without --schema, missing case_path raises error (typer shows help)."""
    result = runner.invoke(app, ["validate-config"])
    assert result.exit_code != 0
