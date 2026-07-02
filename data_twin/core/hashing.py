"""Content hashing utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def parameter_hash(parameters: dict[str, Any], constraints: dict[str, Any] | None = None) -> str:
    payload = {"parameters": parameters, "constraints": constraints or {}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def file_checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
