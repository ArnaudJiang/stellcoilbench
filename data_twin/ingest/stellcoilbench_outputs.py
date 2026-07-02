"""Placeholder hooks for extracting metrics from StellCoilBench artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_summary_metrics(path: Path | str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("metrics", data)
