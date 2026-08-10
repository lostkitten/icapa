"""Secret-safe private scoping for provider cache identities."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ..identity import IdentityError


class UnsafeCacheReuseError(RuntimeError):
    """Raised when read-only reuse lacks verifiable source evidence."""


def private_parameter_scope_digest(
    parameters: Mapping[str, Any] | None,
) -> str:
    """Hash private provider scope for internal cache keys only.

    Unlike ``safe_parameter_identity``, this digest intentionally depends on
    credential, endpoint, and connection values. Only the digest may be
    persisted; callers must never place the source mapping in manifests,
    reports, artifact metadata, or diagnostics.
    """

    try:
        encoded = json.dumps(
            _private_scope_value(dict(parameters or {})),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return sha256(
            b"icapa-private-provider-parameter-scope-v1\0" + encoded
        ).hexdigest()
    except (IdentityError, OSError, TypeError, ValueError) as exc:
        raise UnsafeCacheReuseError(
            "provider parameters cannot be safely scoped for cache reuse"
        ) from exc


def _private_scope_value(value: Any) -> Any:
    """Canonicalize private cache scope without redacting secret values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IdentityError(
                "private provider scope contains a non-finite number"
            )
        return value
    if isinstance(value, Enum):
        return _private_scope_value(value.value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise IdentityError(
                "private provider scope contains a null date"
            )
        return timestamp.isoformat()
    if isinstance(value, Path):
        return str(value.expanduser())
    if isinstance(value, (bytes, bytearray)):
        return {
            "bytes_sha256": sha256(bytes(value)).hexdigest(),
            "length": len(value),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _private_scope_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (set, frozenset)):
        items = [_private_scope_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        )
    if isinstance(value, (list, tuple)):
        return [_private_scope_value(item) for item in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _private_scope_value(scalar())
        except (TypeError, ValueError):
            pass
    raise IdentityError(
        "private provider scope contains an unsupported "
        f"{type(value).__qualname__}"
    )




__all__ = [
    "UnsafeCacheReuseError",
    "private_parameter_scope_digest",
]
