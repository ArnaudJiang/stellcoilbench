"""LLM reasoning history loading and appending."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _load_prior_reasoning(history_path: Path, policy: Dict[str, Any]) -> List[str]:
    """Load prior reasoning blocks from the history file for the LLM prompt.

    Returns a list of strings, each a formatted block from a past batch.
    """
    if not history_path.exists():
        return []
    try:
        data = json.loads(history_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    batches = data if isinstance(data, list) else data.get("batches", [])
    llm_cfg = policy.get("llm_proposer", {})
    max_batches = int(llm_cfg.get("max_prior_reasoning_batches", 50))
    blocks: List[str] = []
    for b in batches[-max_batches:]:
        ts = b.get("timestamp", "?")
        entries = b.get("entries", [])
        lines = [f"[{ts}]"]
        for e in entries:
            cid = e.get("case_id", "?")
            atype = e.get("type", "?")
            reasoning = e.get("reasoning", "")
            lines.append(f"  - {cid} ({atype}): {reasoning}")
        blocks.append("\n".join(lines))
    return blocks


def _append_reasoning_to_history(
    history_path: Path,
    entries: List[Dict[str, Any]],
) -> None:
    """Append a batch's reasoning entries to the history file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_batch = {"timestamp": ts, "entries": entries}

    existing: List[Dict[str, Any]] = []
    if history_path.exists():
        try:
            data = json.loads(history_path.read_text())
            existing = data if isinstance(data, list) else data.get("batches", [])
        except (json.JSONDecodeError, OSError):
            pass

    # Keep last 50 batches (policy controls how many are loaded for prompt)
    all_batches = (existing + [new_batch])[-50:]
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(all_batches, indent=2))
