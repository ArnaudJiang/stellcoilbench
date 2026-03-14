"""Tests for stellcoilbench.utils module.

Covers timed_section, get/clear_timing_results, suppress_output,
and print_timing_summary.
"""

from __future__ import annotations

import time

import pytest

from stellcoilbench.utils import (
    clear_timing_results,
    get_timing_results,
    print_timing_summary,
    suppress_output,
    timed_section,
)


class TestTimedSection:
    """Tests for the timed_section context manager."""

    def setup_method(self) -> None:
        """Clear timing state before each test."""
        clear_timing_results()

    def test_records_elapsed_time(self) -> None:
        """timed_section should record a positive elapsed time."""
        with timed_section("test_section"):
            time.sleep(0.05)

        results = get_timing_results()
        assert "test_section" in results
        assert results["test_section"] >= 0.04

    def test_multiple_sections(self) -> None:
        """Multiple timed_section calls should each be recorded."""
        with timed_section("section_a"):
            time.sleep(0.01)
        with timed_section("section_b"):
            time.sleep(0.01)

        results = get_timing_results()
        assert "section_a" in results
        assert "section_b" in results
        assert len(results) == 2

    def test_overwrites_same_name(self) -> None:
        """Running the same section name twice should overwrite the entry."""
        with timed_section("dup"):
            time.sleep(0.01)
        with timed_section("dup"):
            time.sleep(0.05)

        results = get_timing_results()
        assert results["dup"] >= 0.04

    def test_print_time_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_time=True should emit output (may be suppressed by MPI guard)."""
        with timed_section("printed", print_time=True):
            pass

        results = get_timing_results()
        assert "printed" in results

    def test_records_time_even_on_exception(self) -> None:
        """Timing should still be recorded when the block raises."""
        with pytest.raises(ValueError, match="boom"):
            with timed_section("failing"):
                raise ValueError("boom")

        results = get_timing_results()
        assert "failing" in results


class TestGetClearTimingResults:
    """Tests for get_timing_results and clear_timing_results."""

    def setup_method(self) -> None:
        clear_timing_results()

    def test_get_returns_copy(self) -> None:
        """get_timing_results should return a copy, not the internal dict."""
        with timed_section("x"):
            pass
        results = get_timing_results()
        results["x"] = -999.0
        assert get_timing_results()["x"] != -999.0

    def test_clear_removes_all(self) -> None:
        """clear_timing_results should empty the timing dict."""
        with timed_section("to_clear"):
            pass
        assert len(get_timing_results()) > 0
        clear_timing_results()
        assert len(get_timing_results()) == 0

    def test_empty_initially(self) -> None:
        """After clearing, get_timing_results returns empty dict."""
        assert get_timing_results() == {}

    def test_clear_affects_all_references(self) -> None:
        """clear_timing_results should clear so only new timings appear."""
        with timed_section("pre"):
            pass
        clear_timing_results()
        with timed_section("post"):
            pass
        assert list(get_timing_results().keys()) == ["post"]


class TestPrintTimingSummary:
    """Tests for print_timing_summary."""

    def setup_method(self) -> None:
        clear_timing_results()

    def test_no_data_message(self) -> None:
        """When no timing data exists, should not raise."""
        print_timing_summary()

    def test_with_data(self) -> None:
        """With timing data, should not raise."""
        with timed_section("alpha"):
            time.sleep(0.01)
        with timed_section("beta"):
            time.sleep(0.01)
        print_timing_summary()


class TestSuppressOutput:
    """Tests for the suppress_output context manager."""

    def test_suppresses_stdout(self) -> None:
        """suppress_output should suppress print statements."""
        with suppress_output():
            print("this should not appear")

    def test_restores_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        """After exiting suppress_output, stdout should work again."""
        with suppress_output():
            print("suppressed")
        print("visible")
        captured = capsys.readouterr()
        assert "visible" in captured.out

    def test_context_manager_protocol(self) -> None:
        """suppress_output should be usable as a context manager without error."""
        with suppress_output():
            pass
