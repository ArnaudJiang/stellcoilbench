"""Small filtering helpers."""

from __future__ import annotations

from typing import Any


def match(row: dict[str, Any], **filters: Any) -> bool:
    for key, value in filters.items():
        if value is None:
            continue
        if key == "tags":
            tags = set(row.get("tags") or [])
            wanted = set(value if isinstance(value, list) else [value])
            if not wanted <= tags:
                return False
        elif row.get(key) != value:
            return False
    return True
