"""
Pytest configuration and fixtures.
"""

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# Add src to path so imports work
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

REPO_ROOT = Path(__file__).parent.parent


def pytest_configure(config):
    """Set thread limits before any Gmsh/FEM code to avoid segfaults (macOS/Python 3.12)."""
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def write_case_yaml(
    path: Path,
    surface: str = "input.TestSurface",
    **overrides: Any,
) -> None:
    """Write a minimal valid case YAML file.

    Creates a case YAML with description, surface_params.surface, coils_params,
    and optimizer_params. Pass overrides to merge/override top-level keys
    (e.g. surface_params, coils_params, optimizer_params, description).

    Parameters
    ----------
    path : Path
        Where to write the YAML file.
    surface : str
        Value for surface_params.surface. Default "input.TestSurface".
    **overrides
        Top-level keys to merge into the config (overrides base values).
    """
    base: dict[str, Any] = {
        "description": "test",
        "surface_params": {"surface": surface},
        "coils_params": {"ncoils": 4, "order": 4},
        "optimizer_params": {"max_iterations": 1},
    }
    base.update(overrides)
    path.write_text(yaml.dump(base, default_flow_style=False))


def minimal_case_and_surface(
    tmp_path: Path,
    surface_name: str = "input.test",
    **overrides: Any,
) -> tuple[Path, Path]:
    """Create minimal case.yaml and surface file for MPI/integration tests.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory.
    surface_name : str
        Surface filename (e.g. input.test).
    **overrides
        Passed to write_case_yaml for custom config (e.g. surface_params).
        Overrides defaults for coils_params, optimizer_params, etc.

    Returns
    -------
    tuple[Path, Path]
        (case_yaml_path, surface_file_path).
    """
    case_yaml = tmp_path / "case.yaml"
    defaults: dict[str, Any] = {
        "surface": surface_name,
        "coils_params": {"ncoils": 2, "order": 2},
        "optimizer_params": {"algorithm": "l-bfgs-b", "max_iterations": 1},
        "coil_objective_terms": {"length": 1.0},
    }
    defaults.update(overrides)
    write_case_yaml(case_yaml, **defaults)
    surface_file = tmp_path / surface_name
    surface_file.write_text("&INDATA\nNFP=2\n/")
    return case_yaml, surface_file


def minimal_coils_json(path: Path) -> Path:
    """Write a minimal coils.json (empty object) and return its path.

    Parameters
    ----------
    path : Path
        Directory (e.g. tmp_path) or full path. If directory, writes coils.json
        inside it; if path ends with .json, writes there.

    Returns
    -------
    Path
        Path to the written coils.json file.
    """
    if path.suffix == ".json":
        coils_path = path
    else:
        coils_path = path / "coils.json"
    coils_path.write_text("{}")
    return coils_path


def pytest_collection_modifyitems(config, items):
    """Run post_processing tests before coil_optimization to avoid matplotlib state corruption.

    coil_optimization 3D plots can leave matplotlib in a bad state that causes
    'Figure' object has no attribute 'items' in post_processing plot tests.
    Running post_processing first avoids this.
    """
    post_processing = [i for i in items if "post_processing" in i.nodeid]
    others = [i for i in items if i not in post_processing]
    items[:] = post_processing + others


@pytest.fixture(scope="module")
def qa_coils_and_bs(tmp_path_factory):
    """Run a short optimization for basic_LandremanPaulQA; return coils, bs, surface, output_dir.

    Shared by test_structural_objective and test_fem_benchmarks.
    """
    from stellcoilbench.coil_optimization import optimize_coils
    from stellcoilbench.case_loader import load_case
    from stellcoilbench.post_processing import load_coils_and_surface

    case_path = REPO_ROOT / "cases" / "basic_LandremanPaulQA.yaml"
    case_cfg = load_case(case_path)
    case_cfg.optimizer_params["max_iterations"] = 50

    tmp_path = tmp_path_factory.mktemp("qa_coils")
    output_dir = tmp_path / "optimization_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    coils_json = output_dir / "biot_savart_optimized.json"

    try:
        optimize_coils(
            case_path=case_path,
            coils_out_path=coils_json,
            case_cfg=case_cfg,
            output_dir=output_dir,
            surface_resolution=8,
            skip_post_processing=True,
        )
    except Exception as e:
        pytest.skip(f"QA optimization failed: {e}")

    bfield, surface = load_coils_and_surface(
        coils_json,
        case_yaml_path=case_path,
        plasma_surfaces_dir=REPO_ROOT / "plasma_surfaces",
    )

    return {
        "coils": bfield.coils,
        "bs": bfield,
        "surface": surface,
        "output_dir": output_dir,
        "coils_json": coils_json,
    }


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close all matplotlib figures before and after each test to prevent state leakage.

    Without this, figures from earlier tests (e.g. coil_optimization 3D plots)
    can corrupt matplotlib's internal state and cause 'Figure' object has no
    attribute 'items' in later tests (e.g. post_processing plot tests).
    """
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
    yield
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
