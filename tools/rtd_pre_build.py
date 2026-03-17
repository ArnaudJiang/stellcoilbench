#!/usr/bin/env python
"""Pre-build script for Read the Docs: generate docs/leaderboard.json and RST files."""
from pathlib import Path
import json

out = Path("docs") / "leaderboard.json"
empty = json.dumps({"entries": []})
leaderboard_rst = Path("docs") / "leaderboard.rst"

if leaderboard_rst.exists():
    print("Leaderboard RST exists (from CI), skipping update-db")
else:
    try:
        from stellcoilbench.update_db import update_database

        update_database(Path.cwd())
    except Exception as e:
        print(f"update-db failed ({e}), writing empty leaderboard.json")
        out.write_text(empty)
