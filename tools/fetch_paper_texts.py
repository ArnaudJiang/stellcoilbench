#!/usr/bin/env python3
"""
Fetch full paper text from arXiv via arxiv-txt.org for papers in the manifest.

Reads knowledge/papers_manifest.jsonl, fetches each paper's full text from
https://arxiv-txt.org/pdf/{arxiv_id}, and saves to knowledge/extracted/{arxiv_id}.txt.

Supports resume: skips papers that already have extracted files.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

# Default paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "knowledge" / "papers_manifest.jsonl"
EXTRACTED_DIR = REPO_ROOT / "knowledge" / "extracted"

# Rate limit: delay between requests (seconds)
REQUEST_DELAY = 1.5
REQUEST_TIMEOUT = 60


def _arxiv_id_clean(arxiv_id: str) -> str:
    """Strip version suffix from arxiv_id (e.g., 2510.26155v1 -> 2510.26155)."""
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


def fetch_paper_text(arxiv_id: str) -> str | None:
    """
    Fetch full paper text from arxiv-txt.org.

    Parameters
    ----------
    arxiv_id : str
        arXiv ID, e.g. "2510.26155" or "2510.26155v1".

    Returns
    -------
    str | None
        Full paper text, or None on failure.
    """
    clean_id = _arxiv_id_clean(arxiv_id)
    url = f"https://arxiv-txt.org/pdf/{clean_id}"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException:
        return None


def main() -> int:
    """Run the fetch pipeline."""
    parser = argparse.ArgumentParser(description="Fetch full paper text from arXiv via arxiv-txt.org")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to papers_manifest.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXTRACTED_DIR,
        help="Directory for extracted .txt files",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-fetch even if output file exists",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help="Seconds to wait between requests",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of papers to fetch (for testing)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    papers = load_manifest(args.manifest)
    if not papers:
        print("No papers in manifest.", file=sys.stderr)
        return 1

    if args.limit:
        papers = papers[: args.limit]

    fetched = 0
    skipped = 0
    failed: list[str] = []

    for i, paper in enumerate(papers):
        arxiv_id_raw = paper.get("arxiv_id") or paper.get("id", "")
        if not arxiv_id_raw:
            continue
        clean_id = _arxiv_id_clean(arxiv_id_raw)
        out_path = args.output_dir / f"{clean_id}.txt"

        if not args.no_resume and out_path.exists():
            skipped += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(papers)}] Skipped (exists): {clean_id}")
            continue

        text = fetch_paper_text(arxiv_id_raw)
        if text is None:
            failed.append(clean_id)
            print(f"  [{i + 1}/{len(papers)}] FAILED: {clean_id}", file=sys.stderr)
        else:
            out_path.write_text(text, encoding="utf-8")
            fetched += 1
            title = paper.get("title", "")[:50]
            print(f"  [{i + 1}/{len(papers)}] Fetched: {clean_id} — {title}...")

        time.sleep(args.delay)

    print(f"\nDone: fetched={fetched}, skipped={skipped}, failed={len(failed)}")
    if failed:
        print("Failed IDs:", ", ".join(failed[:20]), "..." if len(failed) > 20 else "")

    return 0


if __name__ == "__main__":
    sys.exit(main())
