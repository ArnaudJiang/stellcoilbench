#!/usr/bin/env python3
"""
Assemble llm_context.md from per-paper summaries and static reference sections.

Preserves the Optimization Guide and Threshold Scaling from the current
llm_context.md, then appends merged Literature Context from knowledge/summaries/*.md,
and a References section.

Produces:
- knowledge/llm_context_full.md (full merge, all papers)
- knowledge/llm_context.md (production: possibly trimmed for LLM token limits)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LLM_CONTEXT_PATH = REPO_ROOT / "knowledge" / "llm_context.md"
SUMMARIES_DIR = REPO_ROOT / "knowledge" / "summaries"
OUTPUT_FULL = REPO_ROOT / "knowledge" / "llm_context_full.md"
OUTPUT_PROD = REPO_ROOT / "knowledge" / "llm_context.md"

# Static sections end before "## Literature Context"
LITERATURE_HEADER = "## Literature Context"

# Off-topic or poorly extracted papers (plasma/turbulence/theory, not coil optimization)
EXCLUDE_ARXIV_IDS = frozenset({
    "2302.11369", "2310.18842", "2405.07085", "2412.14871", "2505.02546",
    "2509.16320", "2510.13521", "2405.19860", "2404.07322", "2407.04039",
    "2409.20328", "2407.19592", "2502.09350", "2311.07467", "2401.09021",
    "2404.02240", "2409.04221", "2411.16411", "2501.18293", "2503.03711",
    "2505.04211", "2502.12319",
})


def _arxiv_id_from_path(p: Path) -> str:
    """Extract arxiv id from summary filename (e.g., 2203.10164.md)."""
    return p.stem


def _theme_keywords() -> dict[str, list[str]]:
    """Map theme names to keywords for auto-categorization."""
    return {
        "Augmented Lagrangian & Constraint Handling": [
            "augmented lagrangian", "lagrangian", "constraint", "penalty",
            "dual variable", "fourier continuation",
        ],
        "Force and Torque Minimization": [
            "force", "torque", "electromagnetic", "lorentz",
            "dipole", "mn/m", "kn/m",
        ],
        "Coil-Coil and Coil-Surface Separation": [
            "separation", "coil-coil", "coil-surface", "cc_threshold",
            "cs_threshold", "minimum distance",
        ],
        "Curvature and Mean Squared Curvature": [
            "curvature", "msc", "mean squared curvature", "smoothness",
        ],
        "Stochastic and Global Optimization": [
            "stochastic", "global optimization", "manufacturing error",
            "random seed", "noise",
        ],
        "Device-Specific (MUSE, W7-X, etc.)": [
            "muse", "tabletop", "w7-x", "w7x", "device-scale",
            "lab-scale", "reactor-scale",
        ],
        "Other Methods and Coil Design": [
            "adjoint", "gradient-based", "single-stage", "proxy",
            "finite-build", "planar coil", "coil design",
        ],
    }


def _assign_theme(summary_text: str) -> str:
    """Assign a theme based on summary content (case-insensitive)."""
    text_lower = summary_text.lower()
    best_theme = "Other Methods and Coil Design"
    best_score = 0
    for theme, keywords in _theme_keywords().items():
        score = sum(1 for k in keywords if k in text_lower)
        if score > best_score:
            best_score = score
            best_theme = theme
    return best_theme


def _extract_sections(md_content: str) -> tuple[str, str]:
    """
    Split markdown into (before_literature, literature_and_after).
    Returns (static_part, literature_part) where static_part ends before ## Literature Context.
    """
    idx = md_content.find(LITERATURE_HEADER)
    if idx < 0:
        return md_content, ""
    static = md_content[:idx].rstrip()
    literature = md_content[idx:]
    return static, literature


def _get_static_prefix(llm_context_path: Path) -> str:
    """Extract everything before ## Literature Context from llm_context.md."""
    if not llm_context_path.exists():
        raise FileNotFoundError(f"Missing {llm_context_path}")
    text = llm_context_path.read_text(encoding="utf-8")
    static, _ = _extract_sections(text)
    return static


def _parse_summary(path: Path) -> dict | None:
    """
    Parse a summary .md file.
    Returns dict with keys: arxiv_id, title, authors, year, summary_para, advice_bullets, takeaways.
    """
    text = path.read_text(encoding="utf-8")
    # Simple parsing: # title, ## Summary, ## Optimization Advice, ## Takeaways
    lines = text.split("\n")
    title = ""
    authors = ""
    year = ""
    summary_para = ""
    advice_bullets: list[str] = []
    takeaways: list[str] = []
    section = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("**Authors:**"):
            # **Authors:** X | **Year:** Y | **arXiv:** Z
            parts = line.split("|")
            for p in parts:
                p = p.strip()
                if p.startswith("**Authors:**"):
                    authors = p.replace("**Authors:**", "").strip()
                elif p.startswith("**Year:**"):
                    year = p.replace("**Year:**", "").strip()
        elif line.startswith("## Summary"):
            section = "summary"
        elif "## Optimization Advice" in line or "Optimization Advice for StellCoilBench" in line:
            section = "advice"
        elif line.startswith("## Takeaways"):
            section = "takeaways"
        elif section == "summary":
            stripped = line.strip()
            if stripped:
                summary_para += " " + stripped if summary_para else stripped
        elif section == "advice" and line.strip().startswith("-"):
            advice_bullets.append(line.strip()[1:].strip())
        elif section == "takeaways" and line.strip().startswith("-"):
            t = line.strip()[1:].strip()
            if t and t not in ("--", "---", "-"):
                takeaways.append(t)

    arxiv_id = _arxiv_id_from_path(path)
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "year": year,
        "summary_para": summary_para.strip(),
        "advice_bullets": advice_bullets,
        "takeaways": takeaways,
        "raw_text": text,
    }


def _format_paper_entry(parsed: dict, ref_num: int) -> str:
    """Format one paper's content for the Literature Context section.

    Header is just [N] to save tokens; full citation is in References at end.
    """
    lines = [
        f"### [{ref_num}]",
        "",
        parsed["summary_para"],
        "",
    ]
    if parsed["advice_bullets"]:
        lines.append("**Optimization Advice:**")
        for b in parsed["advice_bullets"]:
            lines.append(f"- {b}")
        lines.append("")
    if parsed["takeaways"]:
        for t in parsed["takeaways"]:
            if t.strip() in ("--", "---", "-"):
                continue
            lines.append(f"- {t}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Run assembly."""
    parser = argparse.ArgumentParser(
        description="Assemble llm_context.md from paper summaries"
    )
    parser.add_argument(
        "--summaries-dir",
        type=Path,
        default=SUMMARIES_DIR,
        help="Directory with per-paper summary .md files",
    )
    parser.add_argument(
        "--llm-context",
        type=Path,
        default=LLM_CONTEXT_PATH,
        help="Current llm_context.md for static sections",
    )
    parser.add_argument(
        "--output-full",
        type=Path,
        default=OUTPUT_FULL,
        help="Output path for full merged context",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PROD,
        help="Output path for production (trimmed) context",
    )
    parser.add_argument(
        "--max-papers-prod",
        type=int,
        default=60,
        help="Max papers in production llm_context (for token limits)",
    )
    args = parser.parse_args()

    summary_files = sorted(args.summaries_dir.glob("*.md"))
    if not summary_files:
        print("No summary files found. Run subagent summarization first.", file=sys.stderr)
        return 1

    static_prefix = _get_static_prefix(args.llm_context)

    parsed_list: list[dict] = []
    for p in summary_files:
        parsed = _parse_summary(p)
        if parsed and parsed.get("summary_para"):
            if parsed["arxiv_id"] in EXCLUDE_ARXIV_IDS:
                continue
            parsed["theme"] = _assign_theme(parsed["raw_text"])
            parsed_list.append(parsed)

    # Group by theme
    themes: dict[str, list[dict]] = {}
    for p in parsed_list:
        t = p["theme"]
        themes.setdefault(t, []).append(p)

    theme_order = list(_theme_keywords().keys())
    for t in themes:
        if t not in theme_order:
            theme_order.append(t)

    # Build Literature Context
    ref_num = 1
    refs: list[str] = []
    lit_lines: list[str] = [
        "",
        "",
        LITERATURE_HEADER,
        "",
        f"Curated excerpts from {len(parsed_list)} stellarator coil optimization papers.",
        "",
    ]
    paper_to_ref: dict[str, int] = {}

    for theme in theme_order:
        if theme not in themes:
            continue
        papers_in_theme = themes[theme]
        lit_lines.append(f"### Theme: {theme}")
        lit_lines.append("")
        for p in papers_in_theme:
            paper_to_ref[p["arxiv_id"]] = ref_num
            lit_lines.append(_format_paper_entry(p, ref_num))
            refs.append(
                f"[{ref_num}] {p['authors']}. {p['title']}. arXiv:{p['arxiv_id']} ({p['year']})."
            )
            ref_num += 1
        lit_lines.append("")

    lit_lines.append("## References")
    lit_lines.append("")
    lit_lines.extend(refs)

    full_content = static_prefix + "\n".join(lit_lines)

    args.output_full.parent.mkdir(parents=True, exist_ok=True)
    args.output_full.write_text(full_content, encoding="utf-8")
    print(f"Wrote {args.output_full} ({len(parsed_list)} papers)")

    # Production: trim to max_papers_prod by taking first N refs (prioritize earlier themes)
    if len(parsed_list) > args.max_papers_prod:
        prod_parsed = []
        for theme in theme_order:
            if theme not in themes:
                continue
            for p in themes[theme]:
                prod_parsed.append(p)
                if len(prod_parsed) >= args.max_papers_prod:
                    break
            if len(prod_parsed) >= args.max_papers_prod:
                break
        # Rebuild prod content with trimmed list
        prod_themes: dict[str, list[dict]] = {}
        for p in prod_parsed:
            t = p["theme"]
            prod_themes.setdefault(t, []).append(p)
        prod_ref_num = 1
        prod_refs: list[str] = []
        prod_lit: list[str] = [
            "",
            LITERATURE_HEADER,
            "",
            f"Curated excerpts from {len(prod_parsed)} stellarator coil optimization papers.",
            "",
        ]
        for theme in theme_order:
            if theme not in prod_themes:
                continue
            prod_lit.append(f"### Theme: {theme}")
            prod_lit.append("")
            for p in prod_themes[theme]:
                prod_lit.append(_format_paper_entry(p, prod_ref_num))
                prod_refs.append(
                    f"[{prod_ref_num}] {p['authors']}. {p['title']}. arXiv:{p['arxiv_id']} ({p['year']})."
                )
                prod_ref_num += 1
            prod_lit.append("")
        prod_lit.append("## References")
        prod_lit.append("")
        prod_lit.extend(prod_refs)
        prod_content = static_prefix + "\n".join(prod_lit)
    else:
        prod_content = full_content

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prod_content, encoding="utf-8")
    print(f"Wrote {args.output} ({min(len(parsed_list), args.max_papers_prod)} papers)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
