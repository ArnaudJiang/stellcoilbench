"""Tests for list-cases CLI command."""

from typer.testing import CliRunner

from stellcoilbench.cli import app
from tests.conftest import write_case_yaml


runner = CliRunner()


def test_list_cases_shows_cases(tmp_path):
    """list-cases displays case table with filename, surface, description."""
    case1 = tmp_path / "case1.yaml"
    write_case_yaml(case1, surface="input.SurfaceA")
    case1.write_text(
        "description: My first case\n"
        "surface_params:\n  surface: input.SurfaceA\n"
        "coils_params:\n  ncoils: 4\n  order: 4\n"
        "optimizer_params:\n  max_iterations: 1\n"
    )
    case2 = tmp_path / "case2.yaml"
    write_case_yaml(case2, surface="input.SurfaceB")
    case2.write_text(
        "description: Second case\n"
        "surface_params:\n  surface: input.SurfaceB\n"
        "coils_params:\n  ncoils: 6\n  order: 6\n"
        "optimizer_params:\n  max_iterations: 1\n"
    )

    result = runner.invoke(app, ["list-cases", str(tmp_path)])
    assert result.exit_code == 0
    assert "case1.yaml" in result.output
    assert "case2.yaml" in result.output
    assert "input.SurfaceA" in result.output
    assert "input.SurfaceB" in result.output
    assert "Filename" in result.output
    assert "Surface" in result.output
    assert "Description" in result.output


def test_list_cases_skips_pending_and_done(tmp_path):
    """list-cases skips cases/pending/ and cases/done/ subdirs."""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "main.yaml").write_text(
        "description: main\nsurface_params:\n  surface: x\n"
        "coils_params:\n  ncoils: 4\n  order: 4\noptimizer_params:\n  max_iterations: 1\n"
    )
    (tmp_path / "pending").mkdir()
    (tmp_path / "pending" / "queued.json").write_text("{}")
    (tmp_path / "done").mkdir()
    (tmp_path / "done" / "finished.yaml").write_text(
        "description: done\nsurface_params:\n  surface: y\n"
        "coils_params:\n  ncoils: 4\n  order: 4\noptimizer_params:\n  max_iterations: 1\n"
    )

    result = runner.invoke(app, ["list-cases", str(tmp_path)])
    assert result.exit_code == 0
    assert "main.yaml" in result.output
    assert "queued.json" not in result.output
    assert "pending" not in result.output
    assert "done" not in result.output


def test_list_cases_nonexistent_dir():
    """list-cases exits 1 when directory does not exist."""
    result = runner.invoke(app, ["list-cases", "/nonexistent/cases/dir"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_list_cases_empty_dir(tmp_path):
    """list-cases with empty directory prints message."""
    tmp_path.mkdir(exist_ok=True)
    result = runner.invoke(app, ["list-cases", str(tmp_path)])
    assert result.exit_code == 0
    assert "No case YAML files found" in result.output
