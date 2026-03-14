"""YAML load/dump utilities for StellCoilBench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_safe(
    path: Path | None = None, content: str | bytes | None = None
) -> dict[str, Any] | None:
    """
    Load YAML into a dict, returning None on failure.

    Catches OSError and yaml.YAMLError. Use when callers prefer None over raising.

    Parameters
    ----------
    path : Path, optional
        Path to the YAML file. Mutually exclusive with content.
    content : str | bytes, optional
        YAML content as string or bytes. Mutually exclusive with path.

    Returns
    -------
    dict | None
        Parsed YAML content, or None if loading fails.
    """
    try:
        return load_yaml(path=path, content=content)
    except (OSError, yaml.YAMLError):
        return None


def load_yaml(
    path: Path | None = None,
    content: str | bytes | None = None,
) -> dict[str, Any]:
    """
    Load YAML into a dict from a file path or string/bytes content.

    Parameters
    ----------
    path : Path, optional
        Path to the YAML file. Mutually exclusive with content.
    content : str | bytes, optional
        YAML content as string or bytes. Mutually exclusive with path.

    Returns
    -------
    dict
        Parsed YAML content.
    """
    if path is not None:
        return yaml.safe_load(path.read_text()) or {}
    if content is not None:
        s = content.decode("utf-8") if isinstance(content, bytes) else content
        return yaml.safe_load(s) or {}
    raise ValueError("Either path or content must be provided")


def dump_yaml(
    data: dict[str, Any], path: Path | None = None, **kwargs: Any
) -> str | None:
    """
    Write a dict to YAML. If path is given, write to file and return None.
    Otherwise return the YAML string.

    Parameters
    ----------
    data : dict
        Data to serialize.
    path : Path, optional
        If provided, write to this file.
    **kwargs
        Passed to yaml.dump (e.g. default_flow_style, sort_keys).

    Returns
    -------
    str | None
        YAML string if path is None; None if written to file.
    """
    out = yaml.dump(data, default_flow_style=False, sort_keys=False, **kwargs)
    if path is not None:
        path.write_text(out)
        return None
    return out
