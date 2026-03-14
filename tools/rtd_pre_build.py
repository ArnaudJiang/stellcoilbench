#!/usr/bin/env python
"""Pre-build script for Read the Docs: generate docs/leaderboard.json."""
from pathlib import Path
import json
import os

out = Path("docs") / "leaderboard.json"
empty = json.dumps({"entries": []})

if os.environ.get("READTHEDOCS") == "True":
    out.write_text(empty)
    print("update-db skipped (Read the Docs - no submissions)")
else:
    try:
        from stellcoilbench.update_db import update_database
        update_database(Path.cwd())
    except Exception as e:
        print(f"update-db skipped ({e})")
        out.write_text(empty)
