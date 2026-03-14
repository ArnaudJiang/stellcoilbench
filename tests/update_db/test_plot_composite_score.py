"""Tests for update_db._plot_composite_score module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stellcoilbench.update_db._plot_composite_score import (
    _parse_run_date,
    plot_composite_score_vs_date,
)


@pytest.mark.parametrize(
    "input_val",
    ["", "not-a-date", "2026-99-99", 42, None],
    ids=["empty", "invalid", "bad_date", "int", "none"],
)
def test_parse_run_date_returns_none_for_invalid(input_val) -> None:
    """Invalid or non-string input returns None."""
    assert _parse_run_date(input_val) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "date_str,expected_hms",
    [
        ("2026-02-27T01:39:12.423195", (1, 39, 12)),
        ("2026-03-11", (0, 0, 0)),
    ],
    ids=["iso_with_t", "date_only"],
)
def test_parse_run_date_valid_formats(date_str: str, expected_hms: tuple) -> None:
    """Valid date formats parse to datetime."""
    result = _parse_run_date(date_str)
    assert result is not None
    assert result.year == 2026
    h, m, s = expected_hms
    assert (result.hour, result.minute, result.second) == (h, m, s)


@pytest.mark.parametrize(
    "surface_leaderboards",
    [
        {},
        {"surface1": {"entries": []}},
        {"surface1": {"entries": [{"run_date": "2026-03-01"}]}},
    ],
    ids=["empty", "empty_entries", "no_composite_score"],
)
def test_plot_composite_score_no_plottable_returns_false(
    surface_leaderboards: dict, tmp_path: Path
) -> None:
    """No plottable data returns False."""
    result = plot_composite_score_vs_date(
        surface_leaderboards=surface_leaderboards,
        out_path=tmp_path / "plot.png",
    )
    assert result is False


@pytest.mark.parametrize("suffix", [".png", ".svg"], ids=["png", "svg"])
def test_plot_composite_score_writes_file(suffix: str, tmp_path: Path) -> None:
    """Minimal data writes PNG or SVG."""
    out_file = tmp_path / f"composite_score{suffix}"
    result = plot_composite_score_vs_date(
        surface_leaderboards={
            "surface1": {
                "entries": [{"composite_score": 0.85, "run_date": "2026-03-01"}]
            }
        },
        out_path=out_file,
    )
    assert result is True
    assert out_file.exists()


def test_plot_composite_score_savefig_called(tmp_path: Path) -> None:
    """Mock matplotlib; assert savefig called with correct args."""
    out_file = tmp_path / "subdir" / "plot.png"
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    with patch("matplotlib.pyplot") as mock_plt:
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        result = plot_composite_score_vs_date(
            surface_leaderboards={
                "s1": {"entries": [{"composite_score": 0.5, "run_date": "2026-03-10"}]}
            },
            out_path=out_file,
        )
    assert result is True
    mock_fig.savefig.assert_called_once_with(
        out_file, format="png", dpi=150, bbox_inches="tight"
    )
