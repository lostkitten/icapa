"""Range-independent identities for effective-date review calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def canonical_digest(value: Any) -> str:
    """Hash a value after deterministic JSON-safe normalization."""

    payload = json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def build_run_fingerprint(
    *,
    index_id: str,
    methodology: object,
    configuration: Mapping[str, Any] | None = None,
    data_revision: Any = "unversioned",
) -> str:
    """Build a range-independent fingerprint for reusable review calculations."""

    if not isinstance(index_id, str) or not index_id.strip():
        raise ValueError("index_id must not be empty")
    return canonical_digest(
        {
            "schema_version": 1,
            "index_id": index_id,
            "methodology": {
                "type": (
                    f"{type(methodology).__module__}."
                    f"{type(methodology).__qualname__}"
                ),
                "configuration": _canonicalize(methodology),
            },
            "configuration": _canonicalize(configuration or {}),
            "data_revision": _canonicalize(data_revision),
        }
    )


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identity values must be finite")
        return value
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonicalize(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_canonicalize(item) for item in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _canonicalize(scalar())
        except (TypeError, ValueError):
            pass
    if callable(value):
        return {"callable": _qualified_type_name(value)}
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            "type": _qualified_type_name(value),
            "configuration": {
                key: _canonicalize(item)
                for key, item in sorted(attributes.items())
                if not key.startswith("_")
            },
        }
    return {"type": _qualified_type_name(value)}


def _qualified_type_name(value: Any) -> str:
    target = value if isinstance(value, type) else type(value)
    if (
        callable(value)
        and not isinstance(value, type)
        and not hasattr(value, "__dict__")
    ):
        target = value
    module = getattr(target, "__module__", "")
    name = getattr(
        target,
        "__qualname__",
        getattr(target, "__name__", target.__class__.__name__),
    )
    return f"{module}.{name}" if module else str(name)


__all__ = ["build_run_fingerprint", "canonical_digest"]
