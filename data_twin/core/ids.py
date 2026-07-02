"""Stable ID helpers."""

from __future__ import annotations

import hashlib
import json
import uuid


def slug(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    return "_".join(part for part in text.split("_") if part)


def short_hash(data: object, length: int = 10) -> str:
    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def make_id(prefix: str, payload: object | None = None) -> str:
    digest = short_hash(payload if payload is not None else uuid.uuid4().hex, 12)
    return f"{slug(prefix)}_{digest}"
