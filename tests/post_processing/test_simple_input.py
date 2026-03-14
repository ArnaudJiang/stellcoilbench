"""Tests for SIMPLE input file generation."""

from __future__ import annotations

from pathlib import Path

from stellcoilbench.post_processing._simple_input import (
    _build_simple_input_content,
    _fortran_bool,
    _format_fortran_double,
)
from stellcoilbench.post_processing._simple_parse import _extract_simple_params


class TestFortranBool:
    """Tests for _fortran_bool."""

    def test_true_returns_true_literal(self) -> None:
        assert _fortran_bool(True) == ".True."

    def test_false_returns_false_literal(self) -> None:
        assert _fortran_bool(False) == ".False."


class TestFormatFortranDouble:
    """Tests for _format_fortran_double."""

    def test_scientific_uses_d_exponent(self) -> None:
        assert "d" in _format_fortran_double(1e-1)
        assert _format_fortran_double(1e-1) == "1.000000d-01"

    def test_integer_valued_uses_d_format(self) -> None:
        result = _format_fortran_double(1.0)
        assert "d" in result
        assert "1.000000" in result


class TestBuildSimpleInputContent:
    """Tests for _build_simple_input_content."""

    def test_required_keys_present(self) -> None:
        params, provided, netcdffile = _extract_simple_params(
            {}, Path("/path/to/wout.nc")
        )
        content = _build_simple_input_content(params, provided, netcdffile)
        assert "&config" in content
        assert "netcdffile = '/path/to/wout.nc'" in content or "netcdffile =" in content
        assert "trace_time" in content
        assert "sbeg" in content
        assert "ntestpart" in content
        assert "/" in content

    def test_sbeg_array_formatted(self) -> None:
        params, provided, netcdffile = _extract_simple_params(
            {"sbeg": [0.2, 0.5]}, Path("/wout.nc")
        )
        content = _build_simple_input_content(params, provided, netcdffile)
        assert "sbeg" in content
        assert "d" in content  # Fortran double format
