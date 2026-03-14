#!/usr/bin/env python3
"""Generate docs/leaderboard/metric_definitions.json from update_db data.

Run from repo root: python tools/generate_metric_definitions.py

Used to bootstrap the externalized metric definitions. The formatting module
loads from this file when it exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    from stellcoilbench.update_db._formatting import _get_builtin_metric_definitions

    out_path = REPO_ROOT / "docs" / "leaderboard" / "metric_definitions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = _get_builtin_metric_definitions().copy()
    # Convert sets to lists for JSON
    if "reactor_scale_exclude" in data and isinstance(
        data["reactor_scale_exclude"], set
    ):
        data["reactor_scale_exclude"] = sorted(data["reactor_scale_exclude"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
