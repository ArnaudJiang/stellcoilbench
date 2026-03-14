"""Tests for update_db formatting functions."""

from __future__ import annotations

import math

import pytest

from stellcoilbench.update_db import (
    _format_date,
    _format_metric_value,
    _format_numeric_for_leaderboard,
    _metric_definition,
    _metric_detailed_definition,
    _metric_display_name,
    _metric_shorthand,
    _shorthand_to_html_math,
    _shorthand_to_math,
)


# --- Edge-case parametrized tests for formatters ---


@pytest.mark.parametrize(
    "value,metric_name,expected",
    [
        ([], "final_squared_flux", "[]"),
        ([], "coil_order", "[]"),
        (math.nan, "final_squared_flux", "nan"),
        ("x" * 200, "final_squared_flux", "x" * 200),
        ("", "other_metric", ""),
        (1.5e-10, "final_squared_flux", "1.5e-10"),
        ([4, 6, 8], "fourier_continuation_orders", "[4, 6, 8]"),
    ],
    ids=[
        "empty_list_flux",
        "empty_list_coil_order",
        "nan_flux",
        "long_string",
        "empty_string_other",
        "tiny_value",
        "fourier_list_nonempty",
    ],
)
def test_format_metric_value_edge_cases(
    value: object, metric_name: str, expected: str
) -> None:
    """_format_metric_value handles empty lists, NaN, long strings, fourier lists."""
    assert _format_metric_value(value, metric_name) == expected


@pytest.mark.parametrize(
    "value,metric_name,compact,expected_substr",
    [
        (2.5e-8, "final_squared_flux", True, "e"),
        (2.5e-8, "final_squared_flux", False, "e"),
        (-1e-50, "final_squared_flux", True, "0"),
    ],
    ids=["compact_scientific", "standard_scientific", "compact_near_zero"],
)
def test_format_metric_value_compact(
    value: float, metric_name: str, compact: bool, expected_substr: str
) -> None:
    """_format_metric_value compact mode uses ultra-compact notation."""
    result = _format_metric_value(value, metric_name, compact=compact)
    assert expected_substr in result


def test_format_metric_value_nan_integer_metric_raises() -> None:
    """_format_metric_value raises for NaN with integer metrics (int(round(nan)) invalid)."""
    with pytest.raises((ValueError, OverflowError)):
        _format_metric_value(math.nan, "coil_order")


@pytest.mark.parametrize(
    "value,expected",
    [
        (math.nan, "nan"),
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
        ((1, 2), "—"),
        (object(), "—"),
        ("", "—"),
        (0.5, "5.00e-01"),
        (50.0, "50"),
    ],
    ids=[
        "nan",
        "inf",
        "neg_inf",
        "tuple",
        "object",
        "empty_str",
        "small_float",
        "large_float",
    ],
)
def test_format_numeric_for_leaderboard_edge_cases(value: object, expected: str) -> None:
    """_format_numeric_for_leaderboard handles NaN, inf, non-numeric, edge values."""
    result = _format_numeric_for_leaderboard(value)
    if expected == "—":
        assert result == expected
    elif expected == "nan":
        assert result == "nan"
    elif expected in ("inf", "-inf"):
        assert expected in result
    else:
        assert expected in result


@pytest.mark.parametrize(
    "metric_key,expected_substr",
    [
        ("", ""),
        ("final_squared_flux", "Final Squared Flux"),
        ("x", "X"),
        ("a_b_c_d", "A B C D"),
        ("_" * 5, "     "),
    ],
    ids=["empty", "multi_word", "single_char", "many_underscores", "underscores_only"],
)
def test_metric_display_name_edge_cases(metric_key: str, expected_substr: str) -> None:
    """_metric_display_name handles empty, single char, many underscores."""
    result = _metric_display_name(metric_key)
    assert expected_substr in result


@pytest.mark.parametrize(
    "shorthand,expected_substr",
    [
        ("", ":math:"),
        ("d_cc", "d_{cc}"),
        ("d_cc_xy", "d_{cc}_{xy}"),
        ("unknownxyz", ":math:`unknownxyz`"),
        ("n", ":math:`n`"),
        ("max(B_n)", r"\max"),
        ("avg(QS)", "avg"),
    ],
    ids=[
        "empty",
        "two_part_subscript",
        "multi_subscript",
        "unknown_no_underscore",
        "simple_var",
        "max_func",
        "avg_func",
    ],
)
def test_shorthand_to_math_edge_cases(shorthand: str, expected_substr: str) -> None:
    """_shorthand_to_math handles empty, underscores, unknown, functions."""
    result = _shorthand_to_math(shorthand)
    assert expected_substr in result


@pytest.mark.parametrize(
    "shorthand,expected",
    [
        ("", ""),
        ("f_B", "f<sub>B</sub>"),
        ("unknown_xyz", "unknown_xyz"),
        ("Score", "Score"),
        ("d_cc", "d<sub>cc</sub>"),
    ],
    ids=["empty", "mapped", "unknown_passthrough", "plain_label", "subscript"],
)
def test_shorthand_to_html_math_edge_cases(shorthand: str, expected: str) -> None:
    """_shorthand_to_html_math handles empty, unknown, plain labels."""
    assert _shorthand_to_html_math(shorthand) == expected


@pytest.mark.parametrize(
    "metric_name,expected_type,check",
    [
        ("nonexistent_metric_xyz", str, "Nonexistent Metric Xyz"),
        ("final_squared_flux", str, "Squared flux"),
    ],
    ids=["missing_key_fallback", "known_metric"],
)
def test_metric_definition_edge_cases(
    metric_name: str, expected_type: type, check: str
) -> None:
    """_metric_definition returns display_name for missing keys."""
    result = _metric_definition(metric_name)
    assert isinstance(result, expected_type)
    assert check in result


@pytest.mark.parametrize(
    "metric_name,expected",
    [
        ("nonexistent_xyz", None),
        ("final_normalized_squared_flux", dict),
    ],
    ids=["missing_key_none", "known_detailed_returns_dict"],
)
def test_metric_detailed_definition_edge_cases(
    metric_name: str, expected: type | None
) -> None:
    """_metric_detailed_definition returns None for missing keys."""
    result = _metric_detailed_definition(metric_name)
    if expected is None:
        assert result is None
    else:
        assert isinstance(result, expected)


class TestFormatDateISO8601:
    """ISO 8601 date format handling."""

    @pytest.mark.parametrize(
        "date_str,expected",
        [
            ("2025-12-01", "01/12/25"),
            ("2025-06-15T14:30:00", "15/06/25"),
            ("2025-06-15T14:30:00+00:00", "15/06/25"),
            ("03/05/24", "05/03/24"),
            ("31/01/24", "31/01/24"),
            ("01/31/24", "31/01/24"),
            (None, "_unknown_"),
            ("2024-01-15", "15/01/24"),
            ("invalid", "invalid"),
            ("", ""),
            ("_unknown_", "_unknown_"),
            ("20250615", "20250615"),
            # YY-MM-DD format (ISO-like two-digit year) - lines 156-157
            ("24-06-15", "15/06/24"),
            ("99-01-31", "31/01/99"),
        ],
        ids=[
            "iso_basic",
            "iso_with_time",
            "iso_with_timezone",
            "ambiguous_03_05_24",
            "unambiguous_day_first",
            "unambiguous_month_first",
            "none",
            "iso_date",
            "invalid",
            "empty_string",
            "unknown_sentinel",
            "no_separators",
            "yy_mm_dd_24",
            "yy_mm_dd_99",
        ],
    )
    def test_format_date(self, date_str: str | None, expected: str) -> None:
        """_format_date handles ISO, DD/MM/YY, and edge cases."""
        assert _format_date(date_str) == expected


class TestFormatMetricValue:
    """Tests for _format_metric_value."""

    @pytest.mark.parametrize(
        "value,metric_name,expected",
        [
            (4.7, "coil_order", "5"),
            (6, "num_coils", "6"),
            ("", "fourier_continuation_orders", "—"),
            (None, "fourier_continuation_orders", "—"),
            (1e-150, "final_squared_flux", "0"),
        ],
        ids=[
            "integer_coil_order",
            "integer_num_coils",
            "fourier_empty",
            "fourier_none",
            "near_zero",
        ],
    )
    def test_format_metric_value(self, value, metric_name, expected) -> None:
        """_format_metric_value handles integers, empty/fourier, and near-zero."""
        assert _format_metric_value(value, metric_name) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "—"),
        ({}, "—"),
        ([], "—"),
        ("text", "—"),
    ],
)
def test_format_numeric_for_leaderboard_dash_cases(value, expected) -> None:
    """Non-numeric values return dash."""
    assert _format_numeric_for_leaderboard(value) == expected


@pytest.mark.parametrize(
    "value,kwargs,check",
    [
        (5000, {"scientific_for_large": 1000}, "e"),
        (0.001, {"scientific_for_small": 0.01}, "e"),
    ],
)
def test_format_numeric_for_leaderboard_scientific(value, kwargs, check) -> None:
    """Large/small values use scientific notation when configured."""
    result = _format_numeric_for_leaderboard(value, **kwargs)
    assert check in result


class TestMetricShorthand:
    """Tests for _metric_shorthand."""

    @pytest.mark.parametrize(
        "metric,expected",
        [
            ("final_normalized_squared_flux", "f_B"),
            ("max_BdotN_over_B", "max(B_n)"),
            ("final_average_curvature", "κ̄"),
            ("coil_order", "n"),
            ("num_coils", "N"),
            ("unknown_metric_name", "unknown metric name"),
        ],
    )
    def test_metric_shorthand(self, metric, expected) -> None:
        assert _metric_shorthand(metric) == expected


class TestMetricDefinition:
    """Tests for _metric_definition."""

    def test_known_metric_definitions(self) -> None:
        flux_def = _metric_definition("final_squared_flux")
        assert "Squared flux" in flux_def
        assert _metric_definition("final_linking_number")
        defn = _metric_definition("unknown_metric")
        assert defn == "Unknown Metric" or "unknown" in defn.lower()
