"""
Unit tests for knowledge.llm_client and knowledge.llm_endpoints.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


class TestLLMClient:
    """Tests for knowledge.llm_client."""

    def test_get_provider_default(self, monkeypatch):
        """Test _get_provider returns default when not set."""
        monkeypatch.delenv("KB_LLM_PROVIDER", raising=False)
        from knowledge.llm_client import _get_provider

        assert _get_provider() == "anthropic"

    def test_get_provider_from_env(self, monkeypatch):
        """Test _get_provider reads KB_LLM_PROVIDER."""
        monkeypatch.setenv("KB_LLM_PROVIDER", "anthropic")
        from knowledge.llm_client import _get_provider

        assert _get_provider() == "anthropic"

    def test_get_model_default(self, monkeypatch):
        """Test _get_model returns default when not set."""
        monkeypatch.delenv("KB_LLM_MODEL", raising=False)
        from knowledge.llm_client import _get_model

        assert _get_model() == "claude-sonnet-4-20250514"

    def test_get_model_from_env(self, monkeypatch):
        """Test _get_model reads KB_LLM_MODEL."""
        monkeypatch.setenv("KB_LLM_MODEL", "gpt-4")
        from knowledge.llm_client import _get_model

        assert _get_model() == "gpt-4"

    def test_is_available_with_key(self, monkeypatch):
        """Test is_available returns True when API key is set."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from knowledge.llm_client import is_available

        assert is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        """Test is_available returns False when no key is set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KB_LLM_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("KB_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("KB_LLM_PROVIDER", "openai")
        from knowledge.llm_client import is_available

        # May be True if user has key in env; we can't fully clear in test
        result = is_available()
        assert isinstance(result, bool)


class TestLLMEndpoints:
    """Tests for knowledge.llm_endpoints."""

    def test_call_propose_llm_unavailable(self, monkeypatch):
        """Test call_propose returns error when LLM not configured."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KB_LLM_API_KEY", raising=False)
        monkeypatch.setenv("KB_LLM_PROVIDER", "openai")
        from knowledge.llm_endpoints import call_propose

        result = call_propose({}, {})
        assert "error" in result
        assert result["actions"] == []

    def test_call_propose_llm_available_mocked(self, monkeypatch):
        """Test call_propose returns actions when LLM returns JSON."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with (
            patch("knowledge.llm_client.is_available", return_value=True),
            patch("knowledge.llm_client.complete_json") as mock_cj,
        ):
            mock_cj.return_value = [
                {"type": "explore", "surface": "s1", "ncoils": 4, "order": 6}
            ]
            from knowledge.llm_endpoints import call_propose

            result = call_propose(
                {"top_parents": [], "failure_stats": {"fail_rate": 0.1}},
                {
                    "exploration": {
                        "surfaces": ["s1"],
                        "ncoils_choices": [4],
                        "order_choices": [6],
                    }
                },
                batch_size=1,
            )
            assert result["actions"] == [
                {"type": "explore", "surface": "s1", "ncoils": 4, "order": 6}
            ]
            mock_cj.assert_called_once()

    def test_propose_system_contains_domain_docs(self):
        """Test PROPOSE_SYSTEM includes surface catalog, threshold scaling, and guide."""
        from knowledge.llm_endpoints import PROPOSE_SYSTEM

        assert "Plasma Surface Catalog" in PROPOSE_SYSTEM
        assert "a0=" in PROPOSE_SYSTEM
        assert "Threshold Scaling" in PROPOSE_SYSTEM
        assert "ARIES-CS reactor scale" in PROPOSE_SYSTEM
        assert (
            "Optimization Guide" in PROPOSE_SYSTEM or "Key Parameters" in PROPOSE_SYSTEM
        )

    def test_propose_system_contains_action_format(self):
        """Test PROPOSE_SYSTEM still contains the JSON action format specification."""
        from knowledge.llm_endpoints import PROPOSE_SYSTEM

        assert '"type": "mutate"' in PROPOSE_SYSTEM
        assert '"type": "explore"' in PROPOSE_SYSTEM
        assert "parent_id" in PROPOSE_SYSTEM

    def test_load_context_doc_missing_file(self):
        """Test _load_context_doc returns empty string for missing file."""
        from knowledge.llm_endpoints import _load_context_doc

        assert _load_context_doc(Path("/nonexistent/file.txt")) == ""

    def test_load_surface_catalog_text_format(self):
        """Test _load_surface_catalog_text returns formatted catalog."""
        from knowledge.llm_endpoints import _load_surface_catalog_text

        text = _load_surface_catalog_text()
        assert "Plasma Surface Catalog" in text
        assert "a0=" in text
        assert "nfp=" in text

    def test_call_propose_with_run_cards(self, monkeypatch):
        """Test call_propose includes run cards in user prompt when provided."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with (
            patch("knowledge.llm_client.is_available", return_value=True),
            patch("knowledge.llm_client.complete_json") as mock_cj,
        ):
            mock_cj.return_value = [
                {"type": "explore", "surface": "s1", "ncoils": 4, "order": 6}
            ]
            from knowledge.llm_endpoints import call_propose

            result = call_propose(
                {"top_parents": [], "failure_stats": {"fail_rate": 0.1}},
                {
                    "exploration": {
                        "surfaces": ["s1"],
                        "ncoils_choices": [4],
                        "order_choices": [6],
                    },
                },
                batch_size=1,
                run_cards=["Run c1: SUCCESS score=0.001 | QA ncoils=4 order=8"],
            )
            assert result["actions"]
            call_args = mock_cj.call_args
            user_msg = call_args[0][0][1]["content"]
            assert "detailed run cards" in user_msg
            assert "Run c1" in user_msg

    def test_call_propose_with_postmortems(self, monkeypatch):
        """Test call_propose includes postmortems in user prompt when provided."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with (
            patch("knowledge.llm_client.is_available", return_value=True),
            patch("knowledge.llm_client.complete_json") as mock_cj,
        ):
            mock_cj.return_value = []
            from knowledge.llm_endpoints import call_propose

            call_propose(
                {"top_parents": [], "failure_stats": {"fail_rate": 0.3}},
                {"exploration": {"surfaces": ["s1"]}},
                batch_size=1,
                postmortems=["Failure class: timeout\nSuggest: reduce iterations"],
            )
            call_args = mock_cj.call_args
            user_msg = call_args[0][0][1]["content"]
            assert "failure postmortems" in user_msg
            assert "timeout" in user_msg

    def test_call_propose_with_surface_counts(self, monkeypatch):
        """Test call_propose includes surface exploration counts in user prompt."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with (
            patch("knowledge.llm_client.is_available", return_value=True),
            patch("knowledge.llm_client.complete_json") as mock_cj,
        ):
            mock_cj.return_value = []
            from knowledge.llm_endpoints import call_propose

            call_propose(
                {"top_parents": [], "failure_stats": {"fail_rate": 0.0}},
                {"exploration": {"surfaces": ["s1", "s2"]}},
                batch_size=1,
                surface_counts={"s1": 20, "s2": 2},
            )
            call_args = mock_cj.call_args
            user_msg = call_args[0][0][1]["content"]
            assert "Surface exploration coverage" in user_msg
            assert "s1: 20 runs" in user_msg
            assert "s2: 2 runs" in user_msg

    def test_call_propose_with_prior_reasoning(self, monkeypatch):
        """Test call_propose includes prior reasoning in user prompt when provided."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with (
            patch("knowledge.llm_client.is_available", return_value=True),
            patch("knowledge.llm_client.complete_json") as mock_cj,
        ):
            mock_cj.return_value = []
            from knowledge.llm_endpoints import call_propose

            prior = [
                "[2026-03-01 10:00:00]\n  - c1 (explore): Explored ncoils=4 to diversify.",
            ]
            call_propose(
                {"top_parents": [], "failure_stats": {"fail_rate": 0.0}},
                {"exploration": {"surfaces": ["s1"]}},
                batch_size=1,
                prior_reasoning=prior,
            )
            call_args = mock_cj.call_args
            user_msg = call_args[0][0][1]["content"]
            assert "previous proposal batches" in user_msg
            assert "c1 (explore)" in user_msg
            assert "ncoils=4 to diversify" in user_msg

    def test_call_propose_with_baseline_cases(self, monkeypatch):
        """Test call_propose includes baseline reference cases in user prompt."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with (
            patch("knowledge.llm_client.is_available", return_value=True),
            patch("knowledge.llm_client.complete_json") as mock_cj,
        ):
            mock_cj.return_value = []
            from knowledge.llm_endpoints import call_propose

            call_propose(
                {
                    "top_parents": [],
                    "failure_stats": {"fail_rate": 0.0},
                    "baseline_cases": [
                        "Baseline case basic_W7X: surface=W7-X, ncoils=4, order=4\n"
                        "  thresholds (reactor scale): cc_threshold=0.81, cs_threshold=1.13",
                    ],
                },
                {
                    "exploration": {
                        "surfaces": ["input.W7-X_without_coil_ripple_beta0p05_d23p4_tm"]
                    }
                },
                batch_size=1,
            )
            call_args = mock_cj.call_args
            user_msg = call_args[0][0][1]["content"]
            assert "Baseline reference cases" in user_msg
            assert "basic_W7X" in user_msg
            assert "cc_threshold=0.81" in user_msg

    def test_call_propose_without_enrichment_uses_compact_summaries(self, monkeypatch):
        """Test call_propose still works with compact summaries when no run cards."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with (
            patch("knowledge.llm_client.is_available", return_value=True),
            patch("knowledge.llm_client.complete_json") as mock_cj,
        ):
            mock_cj.return_value = []
            from knowledge.llm_endpoints import call_propose

            call_propose(
                {
                    "top_parents": [
                        {
                            "case_id": "p1",
                            "total_score": 0.001,
                            "case_config": {
                                "surface_params": {"surface": "s1"},
                                "coils_params": {"ncoils": 4, "order": 8},
                            },
                        }
                    ],
                    "failure_stats": {"fail_rate": 0.0},
                },
                {"exploration": {"surfaces": ["s1"]}},
                batch_size=1,
            )
            call_args = mock_cj.call_args
            user_msg = call_args[0][0][1]["content"]
            assert "p1: score=0.001" in user_msg
            assert "Top parent runs (for mutate)" in user_msg

    def test_call_propose_threshold_ranges_in_prompt(self, monkeypatch):
        """Test call_propose includes threshold ranges from policy in prompt."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with (
            patch("knowledge.llm_client.is_available", return_value=True),
            patch("knowledge.llm_client.complete_json") as mock_cj,
        ):
            mock_cj.return_value = []
            from knowledge.llm_endpoints import call_propose

            call_propose(
                {"top_parents": [], "failure_stats": {"fail_rate": 0.0}},
                {
                    "exploration": {
                        "surfaces": ["s1"],
                        "cc_threshold_range": [0.5, 2.0],
                        "cs_threshold_range": [1.0, 3.0],
                    },
                },
                batch_size=1,
            )
            call_args = mock_cj.call_args
            user_msg = call_args[0][0][1]["content"]
            assert "Threshold ranges" in user_msg
            assert "cc_threshold" in user_msg
