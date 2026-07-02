"""Append-only JSONL storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path(self, filename: str) -> Path:
        return self.root / filename

    def append(self, filename: str, record: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path(filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def read(self, filename: str) -> list[dict[str, Any]]:
        path = self.path(filename)
        if not path.exists():
            return []
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def write_all(self, filename: str, records: list[dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path(filename).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
