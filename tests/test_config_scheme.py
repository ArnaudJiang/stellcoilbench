"""
Unit tests for config_scheme.py
"""

from pathlib import Path

import pytest

from stellcoilbench.config_scheme import (
    CaseConfig,
    PostProcessingConfig,
    SubmissionMetadata,
)


class TestCaseConfig:
    """Tests for CaseConfig dataclass."""

    @pytest.mark.parametrize(
        "data,check_fn",
        [
            (
                {
                    "description": "Test case",
                    "surface_params": {"surface": "input.test"},
                    "coils_params": {"ncoils": 4},
                    "optimizer_params": {"algorithm": "l-bfgs"},
                },
                lambda c: (
                    c.description == "Test case"
                    and c.surface_params == {"surface": "input.test"}
                    and c.coils_params == {"ncoils": 4}
                    and c.optimizer_params == {"algorithm": "l-bfgs"}
                    and c.coil_objective_terms is None
                ),
            ),
            (
                {
                    "description": "Test case",
                    "surface_params": {"surface": "input.test"},
                    "coils_params": {"ncoils": 4},
                    "optimizer_params": {"algorithm": "l-bfgs"},
                    "coil_objective_terms": {"total_length": "l2_threshold"},
                },
                lambda c: c.coil_objective_terms == {"total_length": "l2_threshold"},
            ),
            (
                {
                    "description": "Test case",
                    "surface_params": {"surface": "input.test"},
                    "coils_params": {"ncoils": 4},
                    "optimizer_params": {"algorithm": "l-bfgs"},
                    "scoring": {"primary": "score_primary"},
                },
                lambda c: c.scoring == {"primary": "score_primary"},
            ),
            (
                {
                    "description": "",
                    "surface_params": {},
                    "coils_params": {},
                    "optimizer_params": {},
                },
                lambda c: (
                    c.description == ""
                    and c.surface_params == {}
                    and c.coils_params == {}
                    and c.optimizer_params == {}
                    and c.coil_objective_terms is None
                ),
            ),
        ],
        ids=["minimal", "optional_fields", "scoring", "missing_fields"],
    )
    def test_from_dict(self, data, check_fn):
        """CaseConfig.from_dict handles minimal, optional, scoring, and missing fields."""
        config = CaseConfig.from_dict(data)
        assert check_fn(config)


class TestSubmissionMetadata:
    """Tests for SubmissionMetadata dataclass."""

    def test_submission_metadata_creation(self):
        """Test creating SubmissionMetadata."""
        metadata = SubmissionMetadata(
            method_version="1.0.0",
            contact="test@example.com",
            hardware="CPU: Test",
        )

        assert metadata.method_version == "1.0.0"
        assert metadata.contact == "test@example.com"
        assert metadata.hardware == "CPU: Test"


class TestPostProcessingConfig:
    """Tests for PostProcessingConfig dataclass."""

    def test_defaults(self):
        """Test that all defaults are set correctly."""
        pp = PostProcessingConfig()

        assert pp.run_vmec is False
        assert pp.helicity_m == 1
        assert pp.helicity_n == 0
        assert pp.ns == 50
        assert pp.plot_boozer is True
        assert pp.plot_poincare is True
        assert pp.nfieldlines == 20
        assert pp.run_simple is False
        assert pp.simple_executable_path is None
        assert pp.run_vmec_original is False
        assert pp.plot_finite_build is False
        assert pp.finite_build_width is None
        assert pp.finite_build_height is None

    def test_custom_values(self):
        """Test construction with non-default values."""
        pp = PostProcessingConfig(
            run_vmec=True,
            helicity_m=2,
            helicity_n=1,
            ns=100,
            nfieldlines=20,
            simple_executable_path=Path("/usr/bin/simple"),
            finite_build_width=0.05,
            finite_build_height=0.10,
        )

        assert pp.run_vmec is True
        assert pp.helicity_m == 2
        assert pp.helicity_n == 1
        assert pp.ns == 100
        assert pp.nfieldlines == 20
        assert pp.simple_executable_path == Path("/usr/bin/simple")
        assert pp.finite_build_width == 0.05
        assert pp.finite_build_height == 0.10

    @pytest.mark.parametrize(
        "pp_params,kw_overrides,check_fn",
        [
            (
                {
                    "run_vmec": True,
                    "plot_poincare": False,
                    "plot_finite_build": True,
                    "finite_build_width": 0.02,
                },
                {},
                lambda pp: (
                    pp.run_vmec is True
                    and pp.plot_poincare is False
                    and pp.plot_finite_build is True
                    and pp.finite_build_width == 0.02
                    and pp.plot_boozer is True
                ),
            ),
            (
                None,
                {},
                lambda pp: (
                    pp.run_vmec is False
                    and pp.plot_poincare is True
                    and pp.plot_boozer is True
                ),
            ),
            ({"nfieldlines": 50}, {}, lambda pp: pp.nfieldlines == 50),
            ({}, {}, lambda pp: pp.nfieldlines == 20),
            ({"nfieldlines": 10}, {"nfieldlines": 30}, lambda pp: pp.nfieldlines == 30),
            (
                {"run_vmec": False},
                {"run_vmec": True, "ns": 200},
                lambda pp: pp.run_vmec is True and pp.ns == 200,
            ),
        ],
        ids=[
            "basic",
            "none_params",
            "extracts_nfieldlines",
            "nfieldlines_default",
            "nfieldlines_override",
            "overrides",
        ],
    )
    def test_from_case_config(self, pp_params, kw_overrides, check_fn):
        """PostProcessingConfig.from_case_config extracts params and applies overrides."""
        cc = CaseConfig(
            description="test",
            surface_params={},
            coils_params={},
            optimizer_params={},
            post_processing_params=pp_params,
        )
        pp = PostProcessingConfig.from_case_config(cc, **kw_overrides)
        assert check_fn(pp)

    def test_to_run_post_processing_kwargs(self):
        """Test to_run_post_processing_kwargs builds correct dict with overrides."""
        pp = PostProcessingConfig(run_vmec=True, nfieldlines=50)
        kw = pp.to_run_post_processing_kwargs(helicity_n=-1, ns=100)
        assert kw["run_vmec"] is True
        assert kw["helicity_n"] == -1
        assert kw["ns"] == 100
        assert kw["nfieldlines"] == 50
        kw2 = pp.to_run_post_processing_kwargs(extra_key="x")
        assert kw2["extra_key"] == "x"


class TestPostProcessingConfigFlags:
    """Tests for PostProcessingConfig dataclass (optimization-related defaults)."""

    def test_defaults(self):
        """Test that all defaults are set correctly."""
        pp = PostProcessingConfig()

        assert pp.run_vmec is False
        assert pp.run_simple is False
        assert pp.plot_boozer is True
        assert pp.nfieldlines == 20
        assert pp.plot_finite_build is False
        assert pp.finite_build_width is None
        assert pp.finite_build_height is None
        assert pp.run_structural is False
        assert pp.structural_E is None
        assert pp.structural_nu is None

    def test_custom_nfieldlines(self):
        """Test construction with custom nfieldlines."""
        pp = PostProcessingConfig(
            plot_poincare=True,
            nfieldlines=50,
        )

        assert pp.plot_poincare is True
        assert pp.nfieldlines == 50


class TestOptimizationOutcome:
    """Tests for OptimizationOutcome dataclass."""

    def test_defaults(self):
        """Test that all defaults are set correctly."""
        from stellcoilbench.coil_optimization._results import OptimizationOutcome

        oc = OptimizationOutcome()

        assert oc.Jf is None
        assert oc.total_current == 0.0
        assert oc.target_B == 5.7
        assert oc.iterations_used == 0
        assert oc.skip_post_processing is False
        assert oc.case_path is None
        assert oc.pp_flags.run_vmec is False
        assert oc.pp_flags.plot_boozer is True
        assert oc.pp_flags.nfieldlines == 20
        assert oc.B_initial is None
        assert oc.base_curves == []
        assert oc.th == {}
        assert oc.cached_thresholds == {}

    def test_custom_values(self):
        """Test construction with non-default values."""
        from stellcoilbench.coil_optimization._results import OptimizationOutcome

        pp = PostProcessingConfig(run_vmec=True, plot_poincare=True, nfieldlines=50)
        oc = OptimizationOutcome(
            total_current=1e6,
            target_B=3.0,
            iterations_used=500,
            ncoils=8,
            skip_post_processing=True,
            pp_flags=pp,
        )

        assert oc.total_current == 1e6
        assert oc.target_B == 3.0
        assert oc.iterations_used == 500
        assert oc.ncoils == 8
        assert oc.skip_post_processing is True
        assert oc.pp_flags.run_vmec is True
        assert oc.pp_flags.nfieldlines == 50


class TestComputeTotalCurrent:
    """Tests for compute_total_current helper."""

    def test_basic_sum(self):
        """Test summing currents from mock coils."""
        from stellcoilbench.coil_optimization._results import compute_total_current
        from types import SimpleNamespace

        coils = [
            SimpleNamespace(current=SimpleNamespace(get_value=lambda: 1e5)),
            SimpleNamespace(current=SimpleNamespace(get_value=lambda: 2e5)),
            SimpleNamespace(current=SimpleNamespace(get_value=lambda: 3e5)),
        ]

        assert compute_total_current(coils) == 6e5
        assert compute_total_current(coils, 2) == 3e5

    def test_empty(self):
        """Test with empty coil list."""
        from stellcoilbench.coil_optimization._results import compute_total_current

        assert compute_total_current([]) == 0.0

    def test_fallback_on_attribute_error(self):
        """Test graceful fallback when .current is missing."""
        from stellcoilbench.coil_optimization._results import compute_total_current
        from types import SimpleNamespace

        coils = [SimpleNamespace()]
        result = compute_total_current(coils)
        assert result == 0.0


class TestProc0Warning:
    """Tests for proc0_warning helper."""

    def test_prefixes_message(self, capsys):
        """Test that proc0_warning adds the Warning: prefix."""
        from stellcoilbench.mpi_utils import proc0_warning

        proc0_warning("something went wrong")
        captured = capsys.readouterr()
        assert "Warning: something went wrong" in captured.out


class TestProc0Try:
    """Tests for proc0_try context manager."""

    def test_suppresses_and_returns_default(self, capsys):
        """proc0_try catches exception, logs warning, suppresses."""
        from stellcoilbench.mpi_utils import proc0_try

        result = None
        with proc0_try("operation failed: {e}", ValueError, default=None):
            raise ValueError("bad value")
        assert result is None
        captured = capsys.readouterr()
        assert "Warning: operation failed: bad value" in captured.out

    def test_preserves_result_on_success(self, capsys):
        """When no exception, block runs and result is set."""
        from stellcoilbench.mpi_utils import proc0_try

        result = None
        with proc0_try("should not appear: {e}", default=None):
            result = 42
        assert result == 42
        assert "should not appear" not in capsys.readouterr().out

    def test_on_catch_called(self, capsys):
        """on_catch callback runs when exception caught."""
        from stellcoilbench.mpi_utils import proc0_try

        seen = []

        def on_catch():
            seen.append(1)

        with proc0_try("failed: {e}", RuntimeError, on_catch=on_catch):
            raise RuntimeError("oops")
        assert seen == [1]
        assert "Warning: failed: oops" in capsys.readouterr().out

    def test_reraise_still_logs_and_reraises(self, capsys):
        """When reraise=True, warning is logged and exception is re-raised."""
        from stellcoilbench.mpi_utils import proc0_try

        with pytest.raises(OSError, match="permission denied"):
            with proc0_try("file error: {e}", OSError, reraise=True):
                raise OSError("permission denied")
        assert "Warning: file error: permission denied" in capsys.readouterr().out

    def test_default_exc_types(self, capsys):
        """When no exc_types given, defaults to OSError, RuntimeError, ValueError."""
        from stellcoilbench.mpi_utils import proc0_try

        with proc0_try("default types: {e}"):
            raise ValueError("val")
        assert "Warning: default types: val" in capsys.readouterr().out

    def test_on_catch_receives_exception(self, capsys):
        """on_catch may accept the exception as first argument."""
        from stellcoilbench.mpi_utils import proc0_try

        caught = []

        def on_catch(exc):
            caught.append(type(exc).__name__)

        with proc0_try("err: {e}", TypeError, on_catch=on_catch):
            raise TypeError("wrong type")
        assert caught == ["TypeError"]
        assert "Warning: err: wrong type" in capsys.readouterr().out
