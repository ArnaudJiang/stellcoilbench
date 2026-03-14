"""
Unit tests for knowledge.make_postmortem and knowledge.make_run_card.
"""

from __future__ import annotations


class TestMakePostmortem:
    """Tests for knowledge.make_postmortem."""

    def test_success_returns_empty(self):
        """Test make_postmortem returns empty string for success."""
        from knowledge.make_postmortem import make_postmortem

        assert make_postmortem({"success": True}) == ""
        assert make_postmortem({}) == ""

    def test_min_sep_violation_suggestions(self):
        """Test make_postmortem for min_sep_violation."""
        from knowledge.make_postmortem import make_postmortem

        out = make_postmortem(
            {
                "success": False,
                "failure_class": "min_sep_violation",
                "failure_reason": "cc",
            }
        )
        assert "Failure class: min_sep_violation" in out
        assert "Suggest:" in out
        assert "coil-coil" in out or "separation" in out

    def test_timeout_suggestions(self):
        """Test make_postmortem for timeout."""
        from knowledge.make_postmortem import make_postmortem

        out = make_postmortem(
            {"success": False, "failure_class": "timeout", "failure_reason": "exceeded"}
        )
        assert "timeout" in out.lower()
        assert "Suggest:" in out

    def test_unknown_class_suggestion(self):
        """Test make_postmortem for unknown failure class."""
        from knowledge.make_postmortem import make_postmortem

        out = make_postmortem(
            {"success": False, "failure_class": "unknown", "failure_reason": "x"}
        )
        assert "inspect logs" in out or "Suggest:" in out

    def test_negative_margins_included(self):
        """Test make_postmortem includes negative margins."""
        from knowledge.make_postmortem import make_postmortem

        out = make_postmortem(
            {
                "success": False,
                "failure_class": "line_search_fail",
                "failure_reason": "x",
                "margins": {"cc_sep": -0.01, "good": 0.5},
            }
        )
        assert "Negative margins" in out
        assert "cc_sep" in out


class TestMakeRunCard:
    """Tests for knowledge.make_run_card."""

    def test_minimal_summary(self):
        """Test make_run_card with minimal summary."""
        from knowledge.make_run_card import make_run_card

        out = make_run_card({"case_id": "c1", "success": True, "total_score": 1e-4})
        assert "c1" in out
        assert "SUCCESS" in out
        assert "1.00e-04" in out or "1.0000e-04" in out or "1e-04" in out

    def test_failed_with_config(self):
        """Test make_run_card with failed run and case_config."""
        from knowledge.make_run_card import make_run_card

        out = make_run_card(
            {
                "case_id": "c2",
                "success": False,
                "total_score": float("inf"),
                "case_config": {
                    "surface_params": {"surface": "s1"},
                    "coils_params": {"ncoils": 5, "order": 8},
                },
                "failure_class": "timeout",
                "failure_reason": "exceeded limit",
            }
        )
        assert "c2" in out
        assert "FAILED" in out
        assert "s1" in out
        assert "5" in out
        assert "8" in out
        assert "timeout" in out

    def test_with_metrics_and_margins(self):
        """Test make_run_card with metrics and tight margins."""
        from knowledge.make_run_card import make_run_card

        out = make_run_card(
            {
                "case_id": "c3",
                "success": True,
                "total_score": 0.1,
                "metrics": {"final_min_cc_separation": 0.15, "BdotN_over_B": 1e-3},
                "margins": {"cc_sep": 0.05},
            }
        )
        assert "CC separation" in out or "0.15" in out
        assert "B·n" in out or "BdotN" in out or "1e-03" in out
        assert "Tight margins" in out or "cc_sep" in out
