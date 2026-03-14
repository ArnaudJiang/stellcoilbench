"""Unit tests for structural_mesh_convergence script."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.conftest import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.structural_mesh_convergence import (
    parse_args,
    run_convergence_study,
    write_csv,
    plot_convergence,
    main,
)


def test_parse_args_defaults() -> None:
    """parse_args returns expected defaults when no args given."""
    args = parse_args([])
    assert args.coils is None
    assert args.case == REPO_ROOT / "cases" / "LandremanPaulQA_structural_optimization.yaml"
    assert "structural_convergence_output" in str(args.output_dir)
    assert not args.no_plots


def test_parse_args_explicit() -> None:
    """parse_args parses --coils, --resolutions, --no-plots."""
    args = parse_args(["--coils", "/tmp/coils.json", "--resolutions", "0.2,0.1", "--no-plots"])
    assert args.coils == Path("/tmp/coils.json")
    assert args.resolutions == "0.2,0.1"
    assert args.no_plots is True


def test_parse_args_quadrature_degrees() -> None:
    """parse_args parses --quadrature-degrees and -q."""
    args = parse_args(["--quadrature-degrees", "1"])
    assert args.quadrature_degrees == "1"
    args_q = parse_args(["-q", "1,2"])
    assert args_q.quadrature_degrees == "1,2"


def test_parse_args_polynomial_degrees() -> None:
    """parse_args parses --polynomial-degrees and -p."""
    args = parse_args(["--polynomial-degrees", "1,2,3"])
    assert args.polynomial_degrees == "1,2,3"
    args_p = parse_args(["-p", "1,3"])
    assert args_p.polynomial_degrees == "1,3"


def test_run_convergence_study_mocked(
    tmp_path: Path,
) -> None:
    """run_convergence_study loops over resolutions, calls mesh gen and structural analysis."""
    coils_path = tmp_path / "coils.json"
    case_path = tmp_path / "case.yaml"
    coils_path.write_text("{}")
    case_path.write_text(
        "description: t\nsurface_params:\n  surface: input.test\n"
        "coils_params:\n  ncoils: 2\n  order: 2\noptimizer_params:\n  max_iterations: 1\n"
    )
    (tmp_path / "input.test").write_text("&INDATA\nNFP=2\n/")

    fake_coils = [MagicMock()]
    fake_bs = MagicMock()
    width, height = 0.1, 0.1

    def mock_finite_build(*args, **kwargs):
        msh_path = kwargs.get("msh_path") or args[1] if len(args) > 1 else None
        if msh_path:
            Path(msh_path).parent.mkdir(parents=True, exist_ok=True)
            Path(msh_path).touch()
        return (Path(msh_path), [0]) if msh_path else None

    def mock_run_structural(**kwargs):
        output_dir = kwargs.get("output_dir")
        vtk_path = str(output_dir / "structural_results.vtk") if output_dir else ""
        return {
            "skipped": False,
            "mean_displacement_m": 1e-4,
            "max_displacement_m": 2e-4,
            "mean_von_mises_stress_Pa": 5e7,
            "max_von_mises_stress_Pa": 1e8,
            "structural_vtk": vtk_path,
        }

    with (
        patch("tools.structural_mesh_convergence._load_coils_and_config") as mock_load,
        patch(
            "tools.structural_mesh_convergence._get_plot_configs",
            return_value=[("default", None, None, None)],
        ),
        patch(
            "stellcoilbench.finite_build.finite_build_coils_to_msh",
            side_effect=mock_finite_build,
        ),
        patch(
            "stellcoilbench.structural_analysis.run_structural_analysis",
            side_effect=mock_run_structural,
        ),
    ):
        mock_load.return_value = (fake_coils, fake_bs, width, height, 2, True)
        out_dir = tmp_path / "out"
        results = run_convergence_study(
            coils_path=coils_path,
            case_path=case_path,
            output_dir=out_dir,
            resolutions=[0.2, 0.1],
            make_plots=False,
        )
    # 2 resolutions × 1 config = 2 results (spring BC only)
    assert len(results) == 2
    assert results[0]["mesh_size_m"] == 0.2
    assert results[0]["max_displacement_m"] == 2e-4
    assert results[0]["max_stress_Pa"] == 1e8
    assert results[0]["use_spring_bc"] is True
    assert results[1]["mesh_size_m"] == 0.1
    assert not results[0]["skipped"]
    assert "structural_results.vtk" in (results[0].get("structural_vtk") or "")


def test_run_convergence_study_mesh_failure_marked_skipped(tmp_path: Path) -> None:
    """When mesh generation fails, result is marked skipped with reason."""
    coils_path = tmp_path / "coils.json"
    case_path = tmp_path / "case.yaml"
    coils_path.write_text("{}")
    case_path.write_text(
        "description: t\nsurface_params:\n  surface: input.test\n"
        "coils_params:\n  ncoils: 2\n  order: 2\noptimizer_params:\n  max_iterations: 1\n"
    )
    (tmp_path / "input.test").write_text("&INDATA\nNFP=2\n/")

    fake_coils = [MagicMock()]
    fake_bs = MagicMock()

    with (
        patch("tools.structural_mesh_convergence._load_coils_and_config") as mock_load,
        patch(
            "tools.structural_mesh_convergence._get_plot_configs",
            return_value=[("default", None, None, None)],
        ),
        patch(
            "stellcoilbench.finite_build.finite_build_coils_to_msh",
            return_value=None,
        ),
    ):
        mock_load.return_value = (fake_coils, fake_bs, 0.1, 0.1, 2, True)
        results = run_convergence_study(
            coils_path=coils_path,
            case_path=case_path,
            output_dir=tmp_path / "out",
            resolutions=[0.2],
            make_plots=False,
        )
    # 1 resolution × 1 config = 1 result (skipped)
    assert len(results) == 1
    assert results[0]["skipped"] is True
    assert "Mesh generation failed" in (results[0].get("reason") or "")


def test_run_convergence_study_two_configs_mocked(tmp_path: Path) -> None:
    """run_convergence_study runs both default and DOLFINx (p=2) when configs include both."""
    coils_path = tmp_path / "coils.json"
    case_path = tmp_path / "case.yaml"
    coils_path.write_text("{}")
    case_path.write_text(
        "description: t\nsurface_params:\n  surface: input.test\n"
        "coils_params:\n  ncoils: 2\n  order: 2\noptimizer_params:\n  max_iterations: 1\n"
    )
    (tmp_path / "input.test").write_text("&INDATA\nNFP=2\n/")

    fake_coils = [MagicMock()]
    fake_bs = MagicMock()
    width, height = 0.1, 0.1

    def mock_finite_build(*args, **kwargs):
        msh_path = kwargs.get("msh_path") or args[1] if len(args) > 1 else None
        if msh_path:
            Path(msh_path).parent.mkdir(parents=True, exist_ok=True)
            Path(msh_path).touch()
        return (Path(msh_path), [0]) if msh_path else None

    def mock_run_structural(**kwargs):
        output_dir = kwargs.get("output_dir")
        vtk_path = str(output_dir / "structural_results.vtk") if output_dir else ""
        return {
            "skipped": False,
            "mean_displacement_m": 1e-4,
            "max_displacement_m": 2e-4,
            "mean_von_mises_stress_Pa": 5e7,
            "max_von_mises_stress_Pa": 1e8,
            "structural_vtk": vtk_path,
        }

    configs_both = [
        ("default", None, None, None),
        ("DOLFINx (p=2)", "dolfinx", None, 2),
    ]
    with (
        patch("tools.structural_mesh_convergence._load_coils_and_config") as mock_load,
        patch(
            "tools.structural_mesh_convergence._get_plot_configs",
            return_value=configs_both,
        ),
        patch(
            "stellcoilbench.finite_build.finite_build_coils_to_msh",
            side_effect=mock_finite_build,
        ),
        patch(
            "stellcoilbench.structural_analysis.run_structural_analysis",
            side_effect=mock_run_structural,
        ),
    ):
        mock_load.return_value = (fake_coils, fake_bs, width, height, 2, True)
        out_dir = tmp_path / "out"
        results = run_convergence_study(
            coils_path=coils_path,
            case_path=case_path,
            output_dir=out_dir,
            resolutions=[0.2, 0.1],
            make_plots=False,
        )
    # 2 resolutions × 2 configs = 4 results (spring BC only)
    assert len(results) == 4
    labels = [r["label"] for r in results]
    assert "default" in labels
    assert "DOLFINx (p=2)" in labels
    assert sum(1 for r in results if r["label"] == "default") == 2
    assert sum(1 for r in results if r["label"] == "DOLFINx (p=2)") == 2


def test_run_convergence_study_q1_only_mocked(tmp_path: Path) -> None:
    """run_convergence_study with q_degrees=[1] produces only q=1 p=1,2 configs."""
    coils_path = tmp_path / "coils.json"
    case_path = tmp_path / "case.yaml"
    coils_path.write_text("{}")
    case_path.write_text(
        "description: t\nsurface_params:\n  surface: input.test\n"
        "coils_params:\n  ncoils: 2\n  order: 2\noptimizer_params:\n  max_iterations: 1\n"
    )
    (tmp_path / "input.test").write_text("&INDATA\nNFP=2\n/")

    fake_coils = [MagicMock()]
    fake_bs = MagicMock()
    width, height = 0.1, 0.1

    def mock_finite_build(*args, **kwargs):
        msh_path = kwargs.get("msh_path") or args[1] if len(args) > 1 else None
        if msh_path:
            Path(msh_path).parent.mkdir(parents=True, exist_ok=True)
            Path(msh_path).touch()
        return (Path(msh_path), [0]) if msh_path else None

    def mock_run_structural(**kwargs):
        output_dir = kwargs.get("output_dir")
        vtk_path = str(output_dir / "structural_results.vtk") if output_dir else ""
        return {
            "skipped": False,
            "mean_displacement_m": 1e-4,
            "max_displacement_m": 2e-4,
            "mean_von_mises_stress_Pa": 5e7,
            "max_von_mises_stress_Pa": 1e8,
            "structural_vtk": vtk_path,
        }

    with (
        patch("tools.structural_mesh_convergence._load_coils_and_config") as mock_load,
        patch(
            "tools.structural_mesh_convergence._get_plot_configs",
            side_effect=lambda q_degrees=None, p_degrees=None: [
                (f"q={q} p={p}", "dolfinx", q, p)
                for q in (q_degrees or [1, 2, 3, 4])
                for p in (p_degrees or [1, 2])
            ],
        ),
        patch(
            "stellcoilbench.finite_build.finite_build_coils_to_msh",
            side_effect=mock_finite_build,
        ),
        patch(
            "stellcoilbench.structural_analysis.run_structural_analysis",
            side_effect=mock_run_structural,
        ),
    ):
        mock_load.return_value = (fake_coils, fake_bs, width, height, 2, True)
        results = run_convergence_study(
            coils_path=coils_path,
            case_path=case_path,
            output_dir=tmp_path / "out",
            resolutions=[0.2],
            make_plots=False,
            q_degrees=[1],
        )
    # 1 resolution × 2 configs (q=1 p=1, q=1 p=2) = 2 results (spring BC only)
    assert len(results) == 2
    labels = [r["label"] for r in results]
    assert all("q=1" in lbl for lbl in labels)
    assert "q=2" not in " ".join(labels)
    assert results[0]["quadrature_degree"] == 1


def test_write_csv_format(tmp_path: Path) -> None:
    """write_csv produces table with expected columns."""
    results = [
        {
            "mesh_size_m": 0.2,
            "label": "default",
            "use_spring_bc": True,
            "n_nodes": 1000,
            "avg_displacement_m": 1e-4,
            "max_displacement_m": 2e-4,
            "avg_stress_Pa": 5e7,
            "max_stress_Pa": 1e8,
            "structural_vtk": "/out/res_0_200/structural_results.vtk",
            "skipped": False,
            "reason": None,
        },
    ]
    csv_path = tmp_path / "convergence.csv"
    write_csv(results, csv_path)
    content = csv_path.read_text()
    assert "mesh_size_m" in content
    assert "n_nodes" in content
    assert "avg_displacement_m" in content
    assert "max_displacement_m" in content
    assert "avg_stress_Pa" in content
    assert "max_stress_Pa" in content
    assert "skipped" in content
    assert "structural_vtk" in content
    assert "0.2" in content
    assert "1000" in content


def test_plot_convergence_creates_file(tmp_path: Path) -> None:
    """plot_convergence writes PNG when enough valid results."""
    results = [
        {"mesh_size_m": 0.2, "label": "default", "avg_displacement_m": 1e-4,
         "max_displacement_m": 2e-4, "avg_stress_Pa": 5e7, "max_stress_Pa": 1e8,
         "p95_stress_Pa": 8e7, "skipped": False},
        {"mesh_size_m": 0.1, "label": "default", "avg_displacement_m": 8e-5,
         "max_displacement_m": 1.5e-4, "avg_stress_Pa": 6e7, "max_stress_Pa": 1.2e8,
         "p95_stress_Pa": 9e7, "skipped": False},
    ]
    plot_path = tmp_path / "convergence_plots.png"
    plot_convergence(results, plot_path)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_plot_convergence_skips_when_insufficient_results(tmp_path: Path) -> None:
    """plot_convergence skips when fewer than 2 valid results."""
    results = [
        {"mesh_size_m": 0.2, "max_displacement_m": 2e-4, "skipped": False},
    ]
    plot_path = tmp_path / "convergence_plots.png"
    plot_convergence(results, plot_path)
    assert not plot_path.exists()


def test_main_requires_coils_when_default_missing(tmp_path: Path) -> None:
    """main returns 1 when --coils not given and default path does not exist."""
    with patch("tools.structural_mesh_convergence.DEFAULT_COILS", tmp_path / "nonexistent.json"):
        assert main(argv=["--no-plots"]) == 1


def test_main_rejects_nonexistent_coils(tmp_path: Path) -> None:
    """main returns 1 when coils path does not exist."""
    assert main(argv=["--coils", str(tmp_path / "missing.json"), "--no-plots"]) == 1
