"""Campaign path and index helpers."""

from __future__ import annotations

from pathlib import Path


DEFAULT_ROOT = Path("experiments/data_twin")


def campaign_dir(campaign_id: str, root: Path | str = DEFAULT_ROOT) -> Path:
    return Path(root) / campaign_id
