"""
Unit tests for case config validation (validate_case_config, _is_valid_*, _validate_surface_exists).
"""

import pytest

from tests.assert_helpers import assert_errors_contain
from stellcoilbench.validate_config import (
    _is_valid_non_negative_number,
    _is_valid_positive_number,
    _validate_surface_exists,
    validate_case_config,
)
from tests.validate_config.conftest import _EDGE_CASE_PARAMS, _base_config, _merge


class TestIsValidNumberEdgeCases:
    """Edge cases for _is_valid_non_negative_number and _is_valid_positive_number."""

    @pytest.mark.parametrize(
        "val,expected",
        [
            (True, False),
            (None, False),
            ("", False),
            ("1.5", True),
            (-0.1, False),
        ],
    )
    def test_is_valid_non_negative_edge_cases(self, val, expected):
        assert _is_valid_non_negative_number(val) == expected

    @pytest.mark.parametrize(
        "val,expected",
        [
            (0, False),
            (0.0, False),
            (1, True),
            ("0.1", True),
        ],
    )
    def test_is_valid_positive_edge_cases(self, val, expected):
        assert _is_valid_positive_number(val) == expected


class TestValidateCaseConfig:
    """Tests for validate_case_config function."""

    def test_valid_config(self):
        """Test validation of a valid configuration."""
        data = _merge(
            _base_config(),
            {
                "description": "Test case",
                "surface_params": {"range": "half period"},
                "optimizer_params": {"max_iterations": 200, "max_iter_lag": 10},
            },
        )
        errors = validate_case_config(data)
        assert errors == []

    def test_valid_focus_backend_config(self):
        """FOCUS backend configs validate when executable and output are configured."""
        data = _merge(
            _base_config(),
            {
                "optimizer_params": {"backend": "focus", "max_iterations": 10},
                "focus_params": {
                    "executable": "focus",
                    "output_harmonics_file": "nfp4_422.focus",
                    "parser": "focus_fourier",
                },
            },
        )
        errors = validate_case_config(data)
        assert errors == []

    def test_focus_backend_requires_executable(self):
        """FOCUS backend needs an executable unless skip_run is true."""
        data = _merge(
            _base_config(),
            {
                "optimizer_params": {"backend": "focus"},
                "focus_params": {"output_harmonics_file": "nfp4_422.focus"},
            },
        )
        errors = validate_case_config(data)
        assert_errors_contain(errors, "focus_params.executable is required")

    def test_focus_backend_rejects_unknown_parser(self):
        """FOCUS parser names are validated early."""
        data = _merge(
            _base_config(),
            {
                "optimizer_params": {"backend": "focus"},
                "focus_params": {
                    "executable": "focus",
                    "output_harmonics_file": "nfp4_422.focus",
                    "parser": "not-a-parser",
                },
            },
        )
        errors = validate_case_config(data)
        assert_errors_contain(errors, "focus_params.parser must be one of")

    def test_valid_finite_section_field(self):
        """finite_section_field accepts a uniform rectangular filament bundle."""
        data = _merge(
            _base_config(),
            {
                "finite_section_field": {
                    "enabled": True,
                    "width": 0.10,
                    "height": 0.10,
                    "n_width": 3,
                    "n_height": 3,
                    "current_distribution": "uniform",
                }
            },
        )
        errors = validate_case_config(data)
        assert errors == []

    def test_invalid_finite_section_field(self):
        """finite_section_field rejects invalid dimensions and distributions."""
        data = _merge(
            _base_config(),
            {
                "finite_section_field": {
                    "enabled": "yes",
                    "width": 0.0,
                    "height": -0.1,
                    "n_width": 0,
                    "n_height": 2.5,
                    "current_distribution": "gaussian",
                    "extra": True,
                }
            },
        )
        errors = validate_case_config(data)
        assert_errors_contain(
            errors,
            "finite_section_field.enabled must be a boolean",
            "finite_section_field.width must be a positive number",
            "finite_section_field.height must be a positive number",
            "finite_section_field.n_width must be a positive integer",
            "finite_section_field.n_height must be a positive integer",
            "finite_section_field.current_distribution must be 'uniform'",
            "Unknown finite_section_field key: 'extra'",
        )

    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        data = {"description": "Test case"}
        errors = validate_case_config(data)
        assert len(errors) == 3
        assert_errors_contain(
            errors,
            "Missing required field: surface_params",
            "Missing required field: coils_params",
            "Missing required field: optimizer_params",
        )

    def test_invalid_surface_params(self):
        """Test validation with invalid surface_params."""
        data = _merge(
            _base_config(),
            {
                "surface_params": "not a dict",
                "coils_params": {},
                "optimizer_params": {},
            },
        )
        errors = validate_case_config(data)
        assert any("surface_params must be a dictionary" in e for e in errors)

    def test_missing_surface_field(self):
        """Test validation with missing surface field.
        Uses inline dict because _merge treats {} as merge (preserves base keys);
        we need surface_params to be truly empty.
        """
        data = {
            "description": "Test case",
            "surface_params": {},
            "coils_params": {},
            "optimizer_params": {},
        }
        errors = validate_case_config(data)
        assert_errors_contain(errors, "surface_params must contain 'surface' field")

    def test_invalid_range(self):
        """Test validation with invalid range."""
        data = _merge(
            _base_config(),
            {
                "surface_params": {"range": "invalid_range"},
                "coils_params": {},
                "optimizer_params": {},
            },
        )
        errors = validate_case_config(data)
        assert_errors_contain(errors, "surface_params.range must be one of")

    def test_surface_target_b_override(self):
        """surface_params.target_B is an allowed positive physical override."""
        data = _merge(
            _base_config(),
            {"surface_params": {"target_B": 1.25}},
        )
        errors = validate_case_config(data)
        assert not errors

    def test_surface_target_b_override_must_be_positive(self):
        """surface_params.target_B must be a positive number when provided."""
        data = _merge(
            _base_config(),
            {"surface_params": {"target_B": 0.0}},
        )
        errors = validate_case_config(data)
        assert_errors_contain(errors, "surface_params.target_B must be a positive number")

    @pytest.mark.parametrize(
        "field_path,invalid_value,expected_substr",
        [
            (
                "coils_params",
                {"ncoils": -1},
                "coils_params.ncoils must be a positive integer",
            ),
            (
                "coils_params",
                {"order": 0},
                "coils_params.order must be a positive integer",
            ),
            (
                "optimizer_params",
                {"max_iterations": 0},
                "optimizer_params.max_iterations must be a positive integer",
            ),
            (
                "optimizer_params",
                {"max_iter_lag": 0},
                "optimizer_params.max_iter_lag must be a positive integer",
            ),
        ],
    )
    def test_invalid_positive_int_fields(
        self, field_path, invalid_value, expected_substr
    ):
        """Test validation with invalid positive integer fields."""
        if field_path == "coils_params":
            data = _merge(
                _base_config(), {"coils_params": invalid_value, "optimizer_params": {}}
            )
        else:
            data = _merge(
                _base_config(),
                {
                    "coils_params": {"ncoils": 4, "order": 4},
                    "optimizer_params": {**invalid_value, "algorithm": "l-bfgs"},
                },
            )
        errors = validate_case_config(data)
        assert any(expected_substr in e for e in errors)

    def test_non_dict_coils_params(self):
        """Test validation when coils_params is not a dict."""
        data = _merge(_base_config(), {"coils_params": "not a dict"})
        errors = validate_case_config(data)
        assert_errors_contain(errors, "coils_params must be a dictionary")

    def test_non_dict_optimizer_params(self):
        """Test validation when optimizer_params is not a dict."""
        data = _merge(_base_config(), {"optimizer_params": "not a dict"})
        errors = validate_case_config(data)
        assert_errors_contain(errors, "optimizer_params must be a dictionary")

    def test_non_dict_coil_objective_terms(self):
        """Test validation when coil_objective_terms is not a dict."""
        data = _merge(_base_config(), {"coil_objective_terms": "not a dict"})
        errors = validate_case_config(data)
        assert any("coil_objective_terms must be a dictionary" in e for e in errors)

    def test_valid_coil_objective_terms(self):
        """Test validation with valid coil_objective_terms."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "coil_objective_terms": {
                    "total_length": "l2_threshold",
                    "coil_curvature": "lp_threshold",
                    "coil_curvature_p": 2,
                    "coil_torsion": "lp_threshold",
                    "coil_torsion_p": 2,
                    "torsion_threshold": 1.0,
                    "torsion_weight": 0.1,
                },
            },
        )
        errors = validate_case_config(data)
        assert errors == []

    def test_structural_animation_vtk_keys_valid(self):
        """structural_animation_vtk and structural_animation_subdir validate."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "coil_objective_terms": {
                    "structural_stress": "l2_threshold",
                    "structural_stress_threshold": 1e7,
                    "structural_animation_vtk": True,
                    "structural_animation_subdir": "vtk_frames",
                },
            },
        )
        errors = validate_case_config(data)
        assert errors == []

    def test_structural_animation_vtk_must_be_bool(self):
        """structural_animation_vtk rejects non-boolean."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "coil_objective_terms": {
                    "structural_stress": "l2_threshold",
                    "structural_stress_threshold": 1e7,
                    "structural_animation_vtk": "yes",
                },
            },
        )
        errors = validate_case_config(data)
        assert_errors_contain(errors, "structural_animation_vtk must be a boolean")

    def test_invalid_coil_objective_term_option(self):
        """Test validation with invalid coil_objective_term option."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "coil_objective_terms": {"total_length": "invalid_option"},
            },
        )
        errors = validate_case_config(data)
        assert_errors_contain(errors, "total_length must be one of")

    def test_unknown_coil_objective_term(self):
        """Test validation with unknown coil_objective_term."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "coil_objective_terms": {"unknown_term": "l2"},
            },
        )
        errors = validate_case_config(data)
        assert_errors_contain(errors, "Unknown coil_objective_terms key")

    def test_invalid_p_parameter(self):
        """Test validation with invalid _p parameter."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "coil_objective_terms": {
                    "coil_curvature": "lp_threshold",
                    "coil_curvature_p": -1,
                },
            },
        )
        errors = validate_case_config(data)
        assert_errors_contain(errors, "coil_curvature_p must be a positive number")

    def test_valid_named_weights(self):
        """Test validation with valid named weight parameters."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "optimizer_params": {"algorithm": "L-BFGS-B"},
                "coil_objective_terms": {
                    "total_length": "l2_threshold",
                    "length_weight": 2.0,
                    "linking_number": "",
                    "linking_weight": 1000.0,
                    "flux_weight": 1.5,
                },
            },
        )
        errors = validate_case_config(data)
        assert errors == []

    def test_valid_link_guard_params(self):
        """Test validation with topology guard parameters."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "coil_objective_terms": {
                    "link_guard": True,
                    "link_guard_interval": 5,
                    "link_guard_penalty": 1e12,
                    "link_guard_tolerance": 0.5,
                    "link_guard_rollback": True,
                },
            },
        )
        errors = validate_case_config(data)
        assert errors == []

    def test_invalid_link_guard_params(self):
        """Test validation rejects invalid topology guard parameters."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "coil_objective_terms": {
                    "link_guard": "yes",
                    "link_guard_interval": 0,
                    "link_guard_penalty": -1.0,
                    "link_guard_rollback": "yes",
                },
            },
        )
        errors = validate_case_config(data)
        assert_errors_contain(
            errors,
            "link_guard must be a boolean",
            "link_guard_interval must be a positive integer",
            "link_guard_penalty must be a positive number",
            "link_guard_rollback must be a boolean",
        )

    def test_all_named_weights_valid(self):
        """Test validation with all named weight parameters."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "coil_objective_terms": {
                    "length_weight": 1.0,
                    "cc_weight": 100.0,
                    "cs_weight": 50.0,
                    "curvature_weight": 0.5,
                    "arclength_variation_weight": 0.1,
                    "msc_weight": 2.0,
                    "force_weight": 10.0,
                    "torque_weight": 5.0,
                    "flux_weight": 1.0,
                    "linking_weight": 1000.0,
                },
            },
        )
        errors = validate_case_config(data)
        assert errors == []

    def test_invalid_negative_weight(self):
        """Test validation with negative weight parameter."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "coil_objective_terms": {"linking_weight": -1.0},
            },
        )
        errors = validate_case_config(data)
        assert_errors_contain(errors, "linking_weight must be a non-negative number")

    def test_invalid_string_weight(self):
        """Test validation with non-numeric string weight parameter."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "coil_objective_terms": {"length_weight": "high"},
            },
        )
        errors = validate_case_config(data)
        assert_errors_contain(errors, "length_weight must be a non-negative number")

    def test_valid_scientific_notation_weight(self):
        """Test validation with scientific notation weight (as string from YAML)."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "coil_objective_terms": {"linking_weight": "1e3"},
            },
        )
        errors = validate_case_config(data)
        assert errors == []

    def test_zero_weight_valid(self):
        """Test validation with zero weight (should be valid - disables the term)."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "coil_objective_terms": {"length_weight": 0.0},
            },
        )
        errors = validate_case_config(data)
        assert errors == []

    def test_unknown_weight_rejected(self):
        """Test that unknown weight parameters are rejected."""
        data = _merge(
            _base_config(),
            {
                "coils_params": {"ncoils": 4},
                "coil_objective_terms": {"unknown_weight": 1.0},
            },
        )
        errors = validate_case_config(data)
        assert_errors_contain(errors, "Unknown coil_objective_terms key")


class TestValidateCaseConfigEdgeCases:
    """Tests for edge case validation branches in validate_case_config."""

    @pytest.mark.parametrize("updates,expected_substrs", _EDGE_CASE_PARAMS)
    def test_validate_case_config_edge_cases(self, updates, expected_substrs):
        """Parametrized edge cases: invalid configs produce expected error substrings."""
        data = _merge(_base_config(), updates)
        errors = validate_case_config(data)
        assert len(errors) > 0
        assert_errors_contain(errors, *expected_substrs)

    def test_p_param_valid_string(self):
        """Valid string _p parameter should pass."""
        data = _merge(_base_config(), {"coil_objective_terms": {"curvature_p": "2.5"}})
        errors = validate_case_config(data)
        assert not any("curvature_p" in e for e in errors)


class TestValidateSurfaceExists:
    """Tests for _validate_surface_exists and surface existence checks."""

    def test_surface_exists_valid(self):
        """Known surface in plasma_surfaces passes validation."""
        surface_params = {"surface": "input.LandremanPaul2021_QA"}
        errors = _validate_surface_exists(surface_params, "cases/test.yaml: ")
        assert errors == []

    def test_surface_exists_invalid(self):
        """Unknown surface name returns error listing available surfaces."""
        surface_params = {"surface": "nonexistent_surface_xyz"}
        errors = _validate_surface_exists(surface_params, "cases/test.yaml: ")
        assert len(errors) == 1
        assert "nonexistent_surface_xyz" in errors[0]
        assert "plasma_surfaces" in errors[0]
        assert "Available:" in errors[0]

    def test_surface_exists_empty_surface_skipped(self):
        """Missing surface key does not run the check."""
        surface_params = {}
        errors = _validate_surface_exists(surface_params, "cases/test.yaml: ")
        assert errors == []

    def test_surface_exists_custom_dir_found(self, tmp_path):
        """Custom surfaces_dir with matching file passes."""
        (tmp_path / "my_surface.focus").write_text("# header\n0 0 1 0 0 0")
        surface_params = {"surface": "my_surface.focus"}
        errors = _validate_surface_exists(
            surface_params, "cases/test.yaml: ", surfaces_dir=tmp_path
        )
        assert errors == []

    def test_surface_exists_custom_dir_not_found(self, tmp_path):
        """Custom surfaces_dir without file returns error."""
        surface_params = {"surface": "missing.focus"}
        errors = _validate_surface_exists(
            surface_params, "cases/test.yaml: ", surfaces_dir=tmp_path
        )
        assert len(errors) == 1
        assert "missing.focus" in errors[0]
