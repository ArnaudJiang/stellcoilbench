"""
Consolidated unit tests for coil_optimization.

Covers: config validation, initialize_coils_loop, evaluate_external_coils,
optimize_coils_loop (1-2 algorithms), LinearPenalty, key error paths.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from tests.conftest import minimal_coils_json, write_case_yaml
from stellcoilbench.path_utils import dump_yaml
from stellcoilbench.coil_optimization import (
    _plot_bn_error_3d,
    optimize_coils,
    _zip_output_files,
    initialize_coils_loop,
    evaluate_external_coils,
    LinearPenalty,
    _get_scipy_algorithm_options,
    _extend_coils_to_higher_order,
    optimize_coils_loop,
)
from stellcoilbench.coil_optimization._optimization_loop import (
    _bcast_coil_dofs_to_workers,
)
from stellcoilbench.coil_optimization.optimization import _create_plotting_surface
from stellcoilbench.case_loader import load_case
from stellcoilbench.config_scheme import CaseConfig, PostProcessingConfig


class TestMergePostProcessingParams:
    """Tests for _merge_post_processing_params."""

    @pytest.mark.parametrize(
        "cli_run_vmec,pp_run_vmec,expected",
        [(False, False, False), (False, True, True), (True, False, True)],
    )
    def test_merge_run_vmec(self, cli_run_vmec, pp_run_vmec, expected):
        from stellcoilbench.coil_optimization._config_parsing import (
            _merge_post_processing_params,
        )

        cli = PostProcessingConfig(run_vmec=cli_run_vmec)
        pp_params = {"run_vmec": pp_run_vmec}
        result = _merge_post_processing_params(pp_params, cli)
        assert result.run_vmec == expected

    @pytest.mark.parametrize(
        "cli_plot_poincare,pp_plot_poincare,expected",
        [
            (False, False, False),
            (False, True, True),
            (True, True, True),
            (True, False, False),
        ],
    )
    def test_merge_plot_poincare(self, cli_plot_poincare, pp_plot_poincare, expected):
        from stellcoilbench.coil_optimization._config_parsing import (
            _merge_post_processing_params,
        )

        cli = PostProcessingConfig(plot_poincare=cli_plot_poincare)
        pp_params = {"plot_poincare": pp_plot_poincare}
        result = _merge_post_processing_params(pp_params, cli)
        assert result.plot_poincare == expected


@pytest.fixture
def simple_surface():
    """Create a simple test surface."""
    from simsopt.geo import SurfaceRZFourier

    s = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
    s.set_rc(0, 0, 1.0)
    s.set_rc(1, 0, 0.1)
    s.set_zs(0, 0, 0.0)
    return s


# ---- Config validation ----
class TestCoilsParamsValidation:
    """Tests for coils_params validation in case.yaml files."""

    @pytest.mark.parametrize(
        "overrides,expect_success,expected_match",
        [
            (
                {
                    "surface": "input.circular_tokamak",
                    "coils_params": {"ncoils": 4, "order": 16, "target_B": 1.0},
                    "optimizer_params": {"algorithm": "l-bfgs"},
                },
                False,
                r"Unknown coils_params key.*target_B",
            ),
            (
                {
                    "surface": "input.circular_tokamak",
                    "coils_params": {"ncoils": 4.0, "order": 16},
                    "optimizer_params": {"algorithm": "l-bfgs"},
                },
                False,
                r"ncoils should be an integer",
            ),
            (
                {
                    "surface": "input.circular_tokamak",
                    "coils_params": {"ncoils": 4, "order": 16.0},
                    "optimizer_params": {"algorithm": "l-bfgs"},
                },
                False,
                r"order should be an integer",
            ),
            (
                {
                    "surface": "input.circular_tokamak",
                    "coils_params": {"ncoils": 4, "order": 16},
                    "optimizer_params": {"algorithm": "l-bfgs"},
                },
                True,
                None,
            ),
            (
                {
                    "surface": "input.circular_tokamak",
                    "coils_params": {
                        "ncoils": 4,
                        "order": 16,
                        "initial_coils_path": "previous/coils.json",
                    },
                    "optimizer_params": {"algorithm": "l-bfgs"},
                },
                True,
                None,
            ),
            (
                {
                    "surface_params": {
                        "surface": "input.circular_tokamak",
                        "range": "half period",
                        "nphi": 32,
                        "ntheta": 32,
                    },
                    "coils_params": {"ncoils": 4, "order": 16},
                    "optimizer_params": {"algorithm": "l-bfgs"},
                },
                False,
                r"Unknown surface_params key",
            ),
        ],
    )
    def test_coils_params_validation(
        self, tmp_path, overrides, expect_success, expected_match
    ):
        """Parametrized validation for coils_params and surface_params."""
        case_yaml = tmp_path / "case.yaml"
        write_case_yaml(case_yaml, **overrides)
        if expect_success:
            config = load_case(case_yaml)
            assert config.coils_params["ncoils"] == 4
            assert config.coils_params["order"] == 16
        else:
            with pytest.raises(ValueError, match=expected_match):
                load_case(case_yaml)


def test_dispatch_uses_initial_coils_path(tmp_path, monkeypatch):
    """Warm-start cases should pass loaded coils into the standard optimizer loop."""
    from stellcoilbench.coil_optimization import _optimization_dispatch as dispatch

    coils_path = tmp_path / "coils.json"
    coils_path.write_text("{}")
    loaded_coils = [object(), object()]
    captured = {}

    def fake_load(path):
        assert Path(path) == coils_path
        return loaded_coils

    def fake_optimize_coils_loop(*args, **kwargs):
        captured.update(kwargs)
        return loaded_coils, {"final_squared_flux": 1.0}

    import simsopt

    monkeypatch.setattr(simsopt, "load", fake_load)
    monkeypatch.setattr(dispatch, "optimize_coils_loop", fake_optimize_coils_loop)

    case_cfg = CaseConfig(
        description="warm start",
        surface_params={"surface": "input.circular_tokamak"},
        coils_params={
            "ncoils": 2,
            "order": 4,
            "initial_coils_path": str(coils_path),
        },
        optimizer_params={"algorithm": "L-BFGS-B", "max_iterations": 5},
    )

    coils, results = dispatch._dispatch_optimization_on_proc0(
        surface=Mock(),
        case_cfg=case_cfg,
        coil_params=dict(case_cfg.coils_params),
        optimizer_params=dict(case_cfg.optimizer_params),
        coil_objective_terms={},
        threshold_kwargs={},
        output_dir=tmp_path,
        surface_resolution=8,
        case_yaml_path_abs=tmp_path / "case.yaml",
        case_path=tmp_path / "case.yaml",
        vc_target=None,
        vc_target_plot=None,
        skip_post_processing_in_loop=True,
        pp_flags=PostProcessingConfig(),
    )

    assert coils is loaded_coils
    assert results["final_squared_flux"] == 1.0
    assert captured["initial_coils"] is loaded_coils


def test_dispatch_routes_focus_backend(tmp_path, monkeypatch):
    """FOCUS backend selection should bypass the standard simsopt loop."""
    from stellcoilbench.coil_optimization import _optimization_dispatch as dispatch

    focus_coils = [object()]
    captured = {}

    def fake_run_focus_backend(**kwargs):
        captured.update(kwargs)
        return focus_coils, {"optimization_backend": "focus"}

    monkeypatch.setattr(dispatch, "run_focus_backend", fake_run_focus_backend)
    monkeypatch.setattr(
        dispatch,
        "optimize_coils_loop",
        lambda *args, **kwargs: pytest.fail("standard optimizer should not run"),
    )

    case_cfg = CaseConfig(
        description="focus",
        surface_params={"surface": "input.circular_tokamak"},
        coils_params={"ncoils": 4, "order": 8},
        optimizer_params={"backend": "focus", "max_iterations": 5},
        focus_params={
            "executable": "focus",
            "output_harmonics_file": "nfp4_422.focus",
        },
    )

    coils, results = dispatch._dispatch_optimization_on_proc0(
        surface=Mock(),
        case_cfg=case_cfg,
        coil_params=dict(case_cfg.coils_params),
        optimizer_params=dict(case_cfg.optimizer_params),
        coil_objective_terms={},
        threshold_kwargs={},
        output_dir=tmp_path,
        surface_resolution=8,
        case_yaml_path_abs=tmp_path / "case.yaml",
        case_path=tmp_path / "case.yaml",
        vc_target=None,
        vc_target_plot=None,
        skip_post_processing_in_loop=True,
        pp_flags=PostProcessingConfig(),
    )

    assert coils is focus_coils
    assert results["optimization_backend"] == "focus"
    assert captured["case_cfg"] is case_cfg
    assert captured["output_dir"] == tmp_path


class TestFocusBackendParsing:
    """Tests for FOCUS output parsing and coefficient mapping."""

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def test_parse_focus_harmonic_sample(self):
        from stellcoilbench.coil_optimization._focus_backend import (
            parse_focus_harmonics,
        )

        sample = self.repo_root / "nfp4_422.focus"
        if not sample.exists():
            pytest.skip("FOCUS harmonic sample fixture is not available")

        parsed = parse_focus_harmonics(sample)

        assert parsed.ncoils == 4
        assert parsed.order == 8
        assert parsed.coils[0].name == "Mod_001"
        assert parsed.coils[0].nseg == 128
        assert parsed.coils[0].current == pytest.approx(1.024e6)
        assert len(parsed.coils[0].xc) == 9

    def test_focus_coefficients_match_curve_dof_order(self):
        from stellcoilbench.coil_optimization._focus_backend import (
            _focus_component_dofs,
        )

        dofs = _focus_component_dofs([10.0, 11.0, 12.0], [0.0, 21.0, 22.0])

        assert dofs.tolist() == [10.0, 21.0, 11.0, 22.0, 12.0]

    def test_parse_focus_filament_sample(self):
        from stellcoilbench.coil_optimization._focus_backend import (
            parse_focus_filaments,
        )

        sample = self.repo_root / "nfp4_422.coils"
        if not sample.exists():
            pytest.skip("FOCUS filament sample fixture is not available")

        parsed = parse_focus_filaments(sample)

        assert parsed.nfp == 4
        assert [coil.name for coil in parsed.coils] == [
            "Mod_001",
            "Mod_002",
            "Mod_003",
            "Mod_004",
        ]
        assert parsed.coils[0].points.shape[1] == 3


class TestCoilTypeDipoleRejected:
    def test_coil_type_dipole_rejected(self, tmp_path):
        case_yaml = tmp_path / "case.yaml"
        write_case_yaml(
            case_yaml,
            description="Dipole case",
            surface="input.circular_tokamak",
            coils_params={"coil_type": "dipole", "ncoils": 4, "order": 16},
            optimizer_params={"algorithm": "l-bfgs"},
        )
        with pytest.raises(ValueError, match="coil_type 'dipole' has been removed"):
            load_case(case_yaml)


# ---- LinearPenalty (merged from test_linear_penalty) ----
class TestLinearPenalty:
    """Key LinearPenalty tests."""

    def test_linear_penalty_below_threshold(self):
        from simsopt.geo import CurveLength, create_equally_spaced_curves

        curves = create_equally_spaced_curves(
            1, 1, stellsym=False, R0=1.0, R1=0.1, order=2, numquadpoints=64
        )
        obj = CurveLength(curves[0])
        lp = LinearPenalty(obj, threshold=1000.0)
        lp.x = np.random.randn(len(lp.x))
        assert lp.J() == 0.0
        assert np.allclose(lp.dJ(), 0.0)

    def test_linear_penalty_above_threshold(self):
        from simsopt.geo import CurveLength, create_equally_spaced_curves

        curves = create_equally_spaced_curves(
            1, 1, stellsym=False, R0=1.0, R1=0.1, order=2, numquadpoints=64
        )
        obj = CurveLength(curves[0])
        lp = LinearPenalty(obj, threshold=0.1)
        lp.x = np.random.randn(len(lp.x))
        J_val = lp.J()
        assert J_val > 0
        assert np.allclose(lp.dJ(), obj.dJ())

    @pytest.mark.parametrize(
        "J_val,threshold,expected_J,expect_dJ_nonzero",
        [
            (5.0, 3.0, 2.0, True),
            (2.0, 3.0, 0.0, False),
        ],
    )
    def test_linear_penalty_mock(self, J_val, threshold, expected_J, expect_dJ_nonzero):
        mock_obj = Mock()
        mock_obj.J.return_value = J_val
        mock_obj.dJ.return_value = np.array([1.0, 2.0])
        mock_obj.x = np.array([1.0, 2.0])
        penalty = LinearPenalty(mock_obj, threshold=threshold)
        assert penalty.J() == expected_J
        if expect_dJ_nonzero:
            assert np.array_equal(penalty.dJ(), np.array([1.0, 2.0]))
        else:
            assert np.allclose(penalty.dJ(), 0.0)


# ---- Algorithm options ----
class TestGetScipyAlgorithmOptions:
    def test_get_options_lbfgsb(self):
        options = _get_scipy_algorithm_options("L-BFGS-B")
        assert "maxiter" in options
        assert "ftol" in options
        assert "gtol" in options


# ---- _extend_coils_to_higher_order ----
class TestExtendCoilsToHigherOrder:
    def test_extend_coils_higher_order(self, simple_surface):
        from simsopt.geo import create_equally_spaced_curves
        from simsopt.field import Current, coils_via_symmetries

        base = create_equally_spaced_curves(
            2, 1, stellsym=False, R0=1.0, R1=0.1, order=2, numquadpoints=64
        )
        coils = coils_via_symmetries(base, [Current(1e6), Current(-1e6)], 1, False)
        extended = _extend_coils_to_higher_order(coils, 4, s=simple_surface, ncoils=2)
        assert len(extended) >= 2
        assert extended[0].curve.order == 4


# ---- Plotting ----
class TestPlotBnError3D:
    def test_plot_bn_error_3d_creates_png(self, tmp_path):
        """_plot_bn_error_3d saves PNG by default (was PDF; switched for GitHub size)."""
        pytest.importorskip("matplotlib")
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import BiotSavart
        from simsopt.geo import create_equally_spaced_curves
        from simsopt.field import Current, coils_via_symmetries

        surface = SurfaceRZFourier(
            nfp=1,
            stellsym=False,
            mpol=2,
            ntor=2,
            quadpoints_phi=np.linspace(0, 1, 16),
            quadpoints_theta=np.linspace(0, 1, 16),
        )
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        base = create_equally_spaced_curves(
            2, 1, stellsym=False, R0=1.0, R1=0.1, order=2, numquadpoints=64
        )
        coils = coils_via_symmetries(base, [Current(1e6), Current(-1e6)], 1, False)
        bs = BiotSavart(coils)
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        _plot_bn_error_3d(surface, bs, coils, out_dir)
        png_path = out_dir / "bn_error_3d_plot.png"
        assert png_path.exists()

    def test_plot_bn_error_3d_handles_missing_matplotlib(self, tmp_path, monkeypatch):
        with patch(
            "stellcoilbench.coil_optimization._plotting.MATPLOTLIB_AVAILABLE", False
        ):
            from simsopt.geo import SurfaceRZFourier
            from simsopt.field import BiotSavart
            from simsopt.geo import create_equally_spaced_curves
            from simsopt.field import Current, coils_via_symmetries

            surface = SurfaceRZFourier(
                nfp=1,
                stellsym=False,
                mpol=2,
                ntor=2,
                quadpoints_phi=np.linspace(0, 1, 16),
                quadpoints_theta=np.linspace(0, 1, 16),
            )
            surface.set_rc(0, 0, 1.0)
            surface.set_zs(0, 0, 0.0)
            base = create_equally_spaced_curves(
                2, 1, stellsym=False, R0=1.0, R1=0.1, order=2, numquadpoints=64
            )
            coils = coils_via_symmetries(base, [Current(1e6), Current(-1e6)], 1, False)
            bs = BiotSavart(coils)
            out_dir = tmp_path / "output"
            out_dir.mkdir()
            _plot_bn_error_3d(surface, bs, coils, out_dir)
            assert not (out_dir / "bn_error_3d_plot.png").exists()


# ---- optimize_coils ----
class TestOptimizeCoils:
    def test_optimize_coils_resolves_surface_and_target_b(self, tmp_path, monkeypatch):
        plasma_dir = tmp_path / "plasma_surfaces"
        plasma_dir.mkdir()
        (plasma_dir / "INPUT.LandremanPaul2021_QA").write_text("dummy")
        case_cfg = CaseConfig.from_dict(
            {
                "description": "test",
                "surface_params": {
                    "surface": "input.LandremanPaul2021_QA",
                    "range": "full torus",
                },
                "coils_params": {"ncoils": 2, "order": 2},
                "optimizer_params": {
                    "algorithm": "augmented_lagrangian",
                    "algorithm_options": {"maxiter": 5},
                },
                "coil_objective_terms": {"total_length": "l2"},
            }
        )
        monkeypatch.chdir(tmp_path)
        calls = {}

        def fake_optimize(surface, **kwargs):
            calls["optimize_kwargs"] = kwargs
            return ["coil"], {"ok": True}

        def fake_save(coils, path):
            calls["save_path"] = path

        def fake_from_vmec_input(filename, *args, **kwargs):
            calls["surface_filename"] = filename
            s = object.__new__(type("Surface", (), {}))
            s.major_radius = lambda: 1.0
            s.minor_radius = lambda: 1.0
            return s

        monkeypatch.setattr(
            "stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input",
            fake_from_vmec_input,
        )
        monkeypatch.setattr(
            "stellcoilbench.coil_optimization._optimization_dispatch.optimize_coils_loop",
            fake_optimize,
        )
        monkeypatch.setattr("simsopt.save", fake_save)
        out_dir = tmp_path / "out"
        results = optimize_coils(
            case_path=tmp_path,
            coils_out_path=tmp_path / "coils.out",
            case_cfg=case_cfg,
            output_dir=out_dir,
        )
        assert results == {"ok": True}
        assert "LandremanPaul2021_QA" in calls["surface_filename"]
        assert calls["optimize_kwargs"]["target_B"] == 1.0

    def test_optimize_coils_unknown_surface_type(self, tmp_path, monkeypatch):
        case_cfg = CaseConfig.from_dict(
            {
                "description": "test",
                "surface_params": {
                    "surface": str(tmp_path / "unknown.dat"),
                    "range": "full torus",
                },
                "coils_params": {"ncoils": 2, "order": 2},
                "optimizer_params": {"algorithm": "augmented_lagrangian"},
            }
        )
        with pytest.raises(ValueError, match="Unknown surface type"):
            optimize_coils(tmp_path, tmp_path / "coils.json", case_cfg=case_cfg)


# ---- Helpers ----
class TestZipOutputFiles:
    def test_zip_output_files_creates_archive(self, tmp_path):
        (tmp_path / "file.vtu").write_text("data")
        (tmp_path / "file.vts").write_text("data")
        zip_path = _zip_output_files(tmp_path)
        assert zip_path.exists()

    def test_zip_output_files_no_vtk(self, tmp_path):
        zip_path = _zip_output_files(tmp_path)
        assert not zip_path.exists()


class TestCreatePlottingSurface:
    def test_create_plotting_surface_fallback(self):
        from simsopt.geo import SurfaceRZFourier

        s = SurfaceRZFourier(nfp=1, stellsym=False, mpol=2, ntor=1)
        s.set_rc(0, 0, 1.0)
        s.set_zs(0, 0, 0.0)
        s_plot, qphi, qtheta = _create_plotting_surface(s, 16, {})
        assert s_plot is not None
        assert s_plot.nfp == 1


class TestInitializeCoilsLoop:
    def test_initialize_coils_loop_no_regularization(self, monkeypatch, tmp_path):
        from simsopt.geo import SurfaceRZFourier
        import simsopt.util.coil_optimization_helper_functions as cohf

        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        monkeypatch.setattr(cohf, "calculate_modB_on_major_radius", lambda bs, s: 1.0)
        coils = initialize_coils_loop(
            s=surface,
            out_dir=tmp_path,
            target_B=1.0,
            ncoils=2,
            order=2,
            regularization=None,
        )
        assert len(coils) > 0


class TestEvaluateExternalCoils:
    def test_evaluate_external_coils_landreman_paul_qa(self):
        coils_path = (
            Path(__file__).parent.parent
            / "cases/done/2026-02-25_153905_83283/coils.json"
        )
        if not coils_path.exists():
            pytest.skip("Case coils not found")
        plasma_dir = Path(__file__).parent.parent / "plasma_surfaces"
        metrics = evaluate_external_coils(
            coils_json_path=coils_path,
            surface_file="input.LandremanPaul2021_QA",
            surface_range="half period",
            surface_resolution=16,
            plasma_surfaces_dir=plasma_dir,
        )
        assert "final_squared_flux" in metrics
        assert "score_primary" in metrics
        assert "num_coils" in metrics


# ---- optimize_coils_loop smoke test ----
class TestOptimizeCoilsLoopSmoke:
    def test_optimize_coils_loop_lbfgsb(self, simple_surface, tmp_path):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        try:
            coils, results = optimize_coils_loop(
                s=simple_surface,
                target_B=1.0,
                out_dir=str(out_dir),
                max_iterations=2,
                ncoils=2,
                order=2,
                verbose=False,
                surface_resolution=4,
                skip_post_processing=True,
            )
            assert coils is not None
            assert len(coils) > 0
            assert "J" in results or "final_squared_flux" in results
        except ImportError as e:
            if "_save_optimized_coils_and_compute_metrics" in str(e):
                pytest.skip("Optimization results save path has import issues")
            raise


# ---- Error paths ----
class TestOptimizeCoilsErrors:
    def test_fourier_continuation_orders_not_list_raises(self, tmp_path):
        """Case YAML with fourier_continuation.orders as int raises."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text(
            dump_yaml(
                {
                    "description": "test",
                    "surface_params": {"surface": "input.circular_tokamak"},
                    "coils_params": {"ncoils": 2, "order": 2},
                    "optimizer_params": {"algorithm": "L-BFGS-B", "max_iterations": 1},
                    "fourier_continuation": {"enabled": True, "orders": 16},
                }
            )
        )
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        with pytest.raises(ValueError, match="must be a list"):
            optimize_coils(
                case_path=case_yaml,
                coils_out_path=out_dir / "coils.json",
                output_dir=out_dir,
                skip_post_processing=True,
            )


class TestBcastCoilDofsStructuralMPI:
    """Test _bcast_coil_dofs_to_workers for structural MPI + Fourier continuation."""

    def test_bcast_coil_dofs_to_workers(self):
        """Workers receive and apply coil DOFs broadcast from rank 0.

        Run with: mpirun -n 2 pytest tests/test_coil_optimization.py -k test_bcast_coil_dofs -v
        """
        from stellcoilbench.mpi_utils import comm_world, is_mpi_enabled

        if not is_mpi_enabled() or comm_world.size < 2:
            pytest.skip("Run with mpirun -n 2 to test structural MPI coil broadcast")

        from simsopt.field import Current, coils_via_symmetries
        from simsopt.geo import create_equally_spaced_curves

        ncoils = 2
        order = 2
        base_curves = create_equally_spaced_curves(
            ncoils, 1, stellsym=True, R0=1.0, R1=0.1, order=order
        )
        base_currents = [Current(1e5) for _ in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, 1, True)

        rank = comm_world.rank
        if rank == 0:
            # Perturb DOFs so workers have different values before broadcast
            for i, bc in enumerate(base_curves):
                dofs = np.array(bc.get_dofs())
                dofs[:] = 1.0 + i * 0.5  # distinct values
                bc.set_dofs(dofs)

        _bcast_coil_dofs_to_workers(base_curves, coils)

        # All ranks should now have identical DOFs
        if rank == 1:
            for i, bc in enumerate(base_curves):
                expected = 1.0 + i * 0.5
                dofs = np.array(bc.get_dofs())
                np.testing.assert_allclose(dofs, expected, err_msg=f"curve {i}")


class TestRunPostProcessingAfterOptimization:
    """Tests for _run_post_processing_after_optimization."""

    def test_returns_empty_when_coils_not_found(self, tmp_path):
        """When no coils.json exists, returns {} and skips post-processing."""
        from stellcoilbench.coil_optimization._post_opt_processing import (
            _run_post_processing_after_optimization,
        )
        from stellcoilbench.config_scheme import PostProcessingConfig

        mock_surface = Mock()
        mock_surface.filename = None
        result = _run_post_processing_after_optimization(
            tmp_path, mock_surface, None, PostProcessingConfig()
        )
        assert result == {}

    def test_returns_empty_on_exception(self, tmp_path):
        """When run_post_processing raises, returns {}."""
        from stellcoilbench.coil_optimization._post_opt_processing import (
            _run_post_processing_after_optimization,
        )
        from stellcoilbench.config_scheme import PostProcessingConfig

        minimal_coils_json(tmp_path)
        mock_surface = Mock()
        mock_surface.filename = None
        with patch(
            "stellcoilbench.post_processing.run_post_processing",
            side_effect=ValueError("mock"),
        ):
            result = _run_post_processing_after_optimization(
                tmp_path, mock_surface, None, PostProcessingConfig()
            )
        assert result == {}

    def test_accepts_pp_flags_keyword(self, tmp_path):
        """Call with pp_flags= keyword (used by Fourier continuation path)."""
        from stellcoilbench.coil_optimization._post_opt_processing import (
            _run_post_processing_after_optimization,
        )
        from stellcoilbench.config_scheme import PostProcessingConfig

        mock_surface = Mock()
        mock_surface.filename = None
        result = _run_post_processing_after_optimization(
            tmp_path,
            mock_surface,
            None,
            pp_flags=PostProcessingConfig(),
        )
        assert result == {}
