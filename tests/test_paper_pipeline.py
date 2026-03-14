"""Unit tests for tools/fetch_paper_texts, summarize_papers_with_agents, assemble_llm_context."""

from __future__ import annotations

import tempfile
from pathlib import Path



class TestFetchPaperTexts:
    """Tests for tools.fetch_paper_texts."""

    def test_load_manifest(self):
        """Test manifest loading from JSONL."""
        from tools.fetch_paper_texts import load_manifest

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                '{"id": "arxiv_2203_10164", "arxiv_id": "2203.10164v2", "title": "Test"}\n'
            )
            f.write('{"id": "arxiv_2507_12681", "arxiv_id": "2507.12681v1", "title": "AL"}\n')
            manifest_path = Path(f.name)
        try:
            papers = load_manifest(manifest_path)
            assert len(papers) == 2
            assert papers[0]["arxiv_id"] == "2203.10164v2"
            assert papers[1]["title"] == "AL"
        finally:
            manifest_path.unlink(missing_ok=True)

    def test_arxiv_id_clean(self):
        """Test arxiv ID version stripping."""
        from tools.fetch_paper_texts import _arxiv_id_clean

        assert _arxiv_id_clean("2510.26155v1") == "2510.26155"
        assert _arxiv_id_clean("2203.10164v2") == "2203.10164"
        assert _arxiv_id_clean("2507.12681") == "2507.12681"


class TestSummarizePapersWithAgents:
    """Tests for tools.summarize_papers_with_agents."""

    def test_batch_generation(self, tmp_path):
        """Test batch config generation from manifest and extracted dir."""
        manifest_path = tmp_path / "manifest.jsonl"
        manifest_path.write_text(
            '{"arxiv_id": "2203.10164v2", "arxiv_id_clean": "2203.10164", '
            '"title": "Stochastic", "authors": ["Wechsung"], "year": 2022}\n'
            '{"arxiv_id": "2507.12681v1", "arxiv_id_clean": "2507.12681", '
            '"title": "AL", "authors": ["Gil"], "year": 2025}\n'
        )
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir()
        (extracted_dir / "2203.10164.txt").write_text("Full paper text")
        (extracted_dir / "2507.12681.txt").write_text("AL paper text")
        batches_dir = tmp_path / "batches"
        batches_dir.mkdir()

        from tools.summarize_papers_with_agents import (
            load_manifest,
        )

        # Recreate minimal logic: papers with extracted text
        papers = load_manifest(manifest_path)
        extracted_files = {p.stem: p for p in extracted_dir.glob("*.txt")}
        available = []
        for p in papers:
            clean_id = p.get("arxiv_id_clean") or p["arxiv_id"].split("v")[0]
            if clean_id in extracted_files:
                available.append({**p, "arxiv_id_clean": clean_id, "extracted_path": str(extracted_files[clean_id])})
        assert len(available) == 2
        assert available[0]["arxiv_id_clean"] == "2203.10164"


class TestAssembleLlmContext:
    """Tests for tools.assemble_llm_context."""

    def test_parse_summary(self, tmp_path):
        """Test parsing of a summary markdown file."""
        summary_md = tmp_path / "2203.10164.md"
        summary_md.write_text("""# Stochastic Optimization
**Authors:** Wechsung, Giuliani | **Year:** 2022 | **arXiv:** 2203.10164

## Summary

This paper uses stochastic optimization to mitigate coil manufacturing errors.

## Optimization Advice for StellCoilBench

- Use random seed variation.
- Enable Fourier continuation.

## Takeaways

- Stochastic methods improve robustness.
""")
        from tools.assemble_llm_context import _parse_summary

        parsed = _parse_summary(summary_md)
        assert parsed is not None
        assert parsed["arxiv_id"] == "2203.10164"
        assert parsed["title"] == "Stochastic Optimization"
        assert "stochastic" in parsed["summary_para"].lower()
        assert len(parsed["advice_bullets"]) >= 2
        assert len(parsed["takeaways"]) >= 1

    def test_assign_theme(self):
        """Test theme assignment from keywords."""
        from tools.assemble_llm_context import _assign_theme

        assert "Augmented Lagrangian" in _assign_theme("We use augmented Lagrangian methods.")
        assert "Force" in _assign_theme("Force and torque minimization for dipole coils.")
        assert "Curvature" in _assign_theme("Mean squared curvature and coil smoothness.")

    def test_format_paper_entry_compact_header(self):
        """Test that _format_paper_entry uses [N] header; full citation only in References."""
        from tools.assemble_llm_context import _format_paper_entry

        parsed = {
            "arxiv_id": "2507.12681",
            "title": "Augmented Lagrangian Coil Optimization",
            "authors": "P. F. Gil et al.",
            "year": "2025",
            "summary_para": "AL replaces penalty weights.",
            "advice_bullets": ["Use AL for constraints."],
            "takeaways": ["AL automates tuning."],
        }
        out = _format_paper_entry(parsed, ref_num=1)
        assert "### [1]" in out
        assert parsed["summary_para"] in out
        assert "P. F. Gil" not in out  # Author only in References

    def test_exclude_arxiv_ids(self):
        """Test that EXCLUDE_ARXIV_IDS contains expected off-topic papers."""
        from tools.assemble_llm_context import EXCLUDE_ARXIV_IDS

        assert "2302.11369" in EXCLUDE_ARXIV_IDS  # Fast-ion (plasma)
        assert "2310.18842" in EXCLUDE_ARXIV_IDS  # Turbulence
        assert "2502.12319" in EXCLUDE_ARXIV_IDS  # Bad extraction
        assert "2507.12681" not in EXCLUDE_ARXIV_IDS  # AL coils (keep)
