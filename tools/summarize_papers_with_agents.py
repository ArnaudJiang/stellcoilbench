#!/usr/bin/env python3
"""
Generate batch configs for subagent summarization and provide the prompt template.

Reads knowledge/extracted/*.txt and knowledge/papers_manifest.jsonl, splits papers
into batches of ~20-25, and writes knowledge/batches/batch_001.json, batch_002.json, etc.
Each batch config lists {arxiv_id, title, authors, year, extracted_path} for the agent.

Also prints the prompt template for running mcp_task subagents.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "knowledge" / "papers_manifest.jsonl"
EXTRACTED_DIR = REPO_ROOT / "knowledge" / "extracted"
BATCHES_DIR = REPO_ROOT / "knowledge" / "batches"
BATCH_SIZE = 25


def _arxiv_id_clean(arxiv_id: str) -> str:
    """Strip version suffix from arxiv_id."""
    return re.sub(r"v\d+$", "", arxiv_id, flags=re.IGNORECASE)


def load_manifest(path: Path) -> list[dict]:
    """Load papers from JSONL manifest."""
    papers: list[dict] = []
    if not path.exists():
        return papers
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return papers


def main() -> int:
    """Generate batch configs."""
    parser = argparse.ArgumentParser(
        description="Generate batch configs for paper summarization subagents"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to papers_manifest.jsonl",
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=EXTRACTED_DIR,
        help="Directory with extracted .txt files",
    )
    parser.add_argument(
        "--batches-dir",
        type=Path,
        default=BATCHES_DIR,
        help="Output directory for batch JSON files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Papers per batch",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the subagent prompt template to stdout",
    )
    args = parser.parse_args()

    papers = load_manifest(args.manifest)
    if not papers:
        print("No papers in manifest.", file=sys.stderr)
        return 1

    # Build paper -> extracted path mapping
    extracted_files = {p.stem: p for p in args.extracted_dir.glob("*.txt")}

    # Only include papers we have extracted text for
    available: list[dict] = []
    for p in papers:
        arxiv_id_raw = p.get("arxiv_id") or p.get("id", "")
        if not arxiv_id_raw:
            continue
        clean_id = _arxiv_id_clean(arxiv_id_raw)
        ext_path = extracted_files.get(clean_id)
        if ext_path is None or not ext_path.exists():
            continue
        available.append({
            **p,
            "arxiv_id_clean": clean_id,
            "extracted_path": str(ext_path),
        })

    if not available:
        print("No papers with extracted text found. Run fetch_paper_texts.py first.", file=sys.stderr)
        return 1

    args.batches_dir.mkdir(parents=True, exist_ok=True)

    # Split into batches
    batches: list[list[dict]] = []
    for i in range(0, len(available), args.batch_size):
        batches.append(available[i : i + args.batch_size])

    for idx, batch in enumerate(batches):
        out_path = args.batches_dir / f"batch_{idx + 1:03d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2)
        print(f"Wrote {out_path} ({len(batch)} papers)")

    print(f"\nGenerated {len(batches)} batches, {len(available)} papers total (of {len(papers)} in manifest)")

    if args.print_prompt:
        _print_prompt_template(args.batches_dir, len(batches))

    return 0


def _print_prompt_template(batches_dir: Path, num_batches: int) -> None:
    """Print the prompt to use when invoking mcp_task for each batch."""
    prompt = f"""
SUBAGENT PROMPT TEMPLATE (use for each of {num_batches} batches):

You are summarizing stellarator coil optimization papers for StellCoilBench.

TASK: For each paper in the batch, read the full text at the given extracted_path,
then write a detailed summary file to knowledge/summaries/{{arxiv_id_clean}}.md

OUTPUT FORMAT (per paper):
---
# {{title}}
**Authors:** {{authors (comma-separated)}} | **Year:** {{year}} | **arXiv:** {{arxiv_id_clean}}

## Summary

[One large paragraph: key methods, coil/optimization setup, main results, numerical findings (thresholds, ncoils, order, convergence).]

## Optimization Advice for StellCoilBench

- [Bullet 1: actionable threshold/parameter suggestion]
- [Bullet 2: ...]
- [Bullet 3: ...]
- [3-5 bullets total]

## Takeaways

- [High-level lesson 1]
- [1-3 bullets]
---

INSTRUCTIONS:
1. Load the batch JSON from {batches_dir}/batch_XXX.json (replace XXX with batch number 001 to {num_batches:03d})
2. For each paper in the batch: read the file at extracted_path
3. Write knowledge/summaries/{{arxiv_id_clean}}.md for each paper
4. Focus on: coil optimization, constraint handling, ncoils/order trade-offs, numerical benchmarks, force/torque, curvature, separation
5. Extract actionable advice for StellCoilBench case proposal (thresholds, algorithms, Fourier continuation, etc.)

Workspace: /Users/akaptanoglu/stellcoilbench_fork
"""
    print(prompt, file=sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
