"""File and label helpers for reproducible index-research workflows.

The v1 backtesting, simulation, analytics, and reporting APIs remain
available independently.  This module composes them behind a small research
API while keeping cache reuse opt-in and evidence based.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4


def _report_label(value: object, *, fallback: str) -> str:
    """Return a conservative label accepted by the reporting contract."""

    allowed = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz" "0123456789 ._&(),/+-"
    )
    result = "".join(character for character in str(value) if character in allowed)
    while ".." in result:
        result = result.replace("..", ".")
    while "//" in result:
        result = result.replace("//", "/")
    result = result.strip(" ._")[:128]
    return result or fallback


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [name for name in globals() if name.startswith("_")]
