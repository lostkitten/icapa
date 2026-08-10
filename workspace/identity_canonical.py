"""Canonical, secret-safe identities for values and tabular data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


_SECRET_PARTS = (
    "account",
    "connection",
    "credential",
    "dsn",
    "endpoint",
    "host",
    "password",
    "private_key",
    "query",
    "sql",
    "secret",
    "token",
    "uri",
    "url",
    "user",
)

_SENSITIVE_TEXT_PATTERNS = (
    re.compile(
        r"(?i)\b(?:jdbc:|odbc:|postgres(?:ql)?://|mysql://|mariadb://|"
        r"mssql://|sqlserver://|oracle://|snowflake://|redshift://|"
        r"mongodb(?:\+srv)?://)"
    ),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]*@"),
    re.compile(
        r"(?i)\b(?:password|passwd|pwd|token|secret|api[_ -]?key|"
        r"connection[_ -]?string)\s*[:=]\s*[^\s;,]+"
    ),
    re.compile(
        r"(?is)^\s*(?:select\b.+\bfrom\b|insert\s+into\b|update\b.+\bset\b|"
        r"delete\s+from\b|merge\s+into\b|create\s+(?:table|view)\b|"
        r"alter\s+table\b|drop\s+(?:table|view)\b)"
    ),
    re.compile(
        r"(?i)(?:^|;)\s*(?:server|host|database|user\s*id|uid|password|pwd)"
        r"\s*=\s*[^;]+"
    ),
)


class IdentityError(ValueError):
    """Base error for values that cannot be fingerprinted safely."""


class UnfingerprintableComponentError(IdentityError):
    """Raised when cache identity would otherwise omit executable behavior."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes for a supported identity value."""

    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def automatic_digest(value: Any) -> str:
    """Return a SHA-256 digest without requiring a caller-supplied version."""

    return sha256(canonical_json_bytes(value)).hexdigest()


def safe_parameter_identity(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Describe provider parameters without serializing credential values."""

    parameters = dict(parameters or {})
    semantic = {
        str(key): (
            {"redacted": True}
            if _is_secret_key(str(key))
            else canonicalize(value)
        )
        for key, value in sorted(parameters.items(), key=lambda item: str(item[0]))
    }
    digestable = {
        key: value for key, value in semantic.items() if value != {"redacted": True}
    }
    return {
        "keys": sorted(map(str, parameters)),
        "redacted_keys": sorted(
            str(key) for key in parameters if _is_secret_key(str(key))
        ),
        "semantic_digest": automatic_digest(digestable),
    }



def dataframe_content_digest(
    frame: pd.DataFrame,
    *,
    sort_by: Sequence[str] | None = None,
) -> str:
    """Hash DataFrame values, dtypes, columns, and index metadata.

    This helper is available before the optional Parquet backend is imported.
    The Parquet store uses an Arrow-logical digest for persisted artifacts.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    working = frame.copy()
    original_index_names = list(working.index.names)
    implicit_range_index = (
        isinstance(working.index, pd.RangeIndex)
        and working.index.name is None
        and working.index.equals(pd.RangeIndex(len(working)))
    )
    if implicit_range_index:
        safe_index_names: list[str] = []
        flat = working.reset_index(drop=True)
    else:
        safe_index_names = [
            name if name is not None else f"__index_level_{position}__"
            for position, name in enumerate(original_index_names)
        ]
        working.index = working.index.set_names(safe_index_names)
        flat = working.reset_index()
    if sort_by:
        missing = set(sort_by).difference(flat.columns)
        if missing:
            raise KeyError(f"dataframe digest sort columns are missing: {sorted(missing)}")
        flat = flat.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    column_digests: list[dict[str, Any]] = []
    for position in range(len(flat.columns)):
        series = flat.iloc[:, position]
        hash_input = (
            series.map(_canonical_object_hash_token)
            if series.dtype == object
            else series
        )
        hashes = pd.util.hash_pandas_object(
            hash_input,
            index=False,
            categorize=False,
        ).to_numpy(dtype="<u8", copy=False)
        column_digests.append(
            {
                "position": position,
                "name": canonicalize(flat.columns[position]),
                "dtype": str(series.dtype),
                "values_digest": sha256(hashes.tobytes()).hexdigest(),
            }
        )
    payload = {
        "index_names": [] if implicit_range_index else original_index_names,
        "index_columns": safe_index_names,
        "row_count": int(len(flat)),
        "columns": column_digests,
    }
    return automatic_digest(payload)


def _canonical_object_hash_token(value: Any) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        payload: Any = {"missing": True}
    elif isinstance(value, float) and math.isnan(value):
        payload = {"missing": True}
    else:
        payload = {
            "type": (
                f"{type(value).__module__}."
                f"{type(value).__qualname__}"
            ),
            "value": canonicalize(value),
        }
    return canonical_json_bytes(payload).hex()


def canonicalize(value: Any) -> Any:
    """Normalize supported values without silently dropping behavior."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return (
            _sensitive_identity_token(value)
            if _is_sensitive_text(value)
            else value
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IdentityError("identity values must not contain non-finite numbers")
        return value
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise IdentityError("identity values must not contain null dates")
        return timestamp.isoformat()
    if isinstance(value, Path):
        path_text = str(value.expanduser())
        return {
            "file_name": value.name,
            "path_digest": sha256(path_text.encode("utf-8")).hexdigest(),
        }
    if isinstance(value, pd.DataFrame):
        return {"dataframe_digest": dataframe_content_digest(value)}
    if isinstance(value, pd.Series):
        return {
            "series_name": canonicalize(value.name),
            "dataframe_digest": dataframe_content_digest(value.to_frame()),
        }
    if isinstance(value, np.ndarray):
        return canonicalize(value.tolist())
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return canonicalize(scalar())
        except (TypeError, ValueError):
            pass
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: (
                _sensitive_identity_token(getattr(value, item.name))
                if _is_secret_key(item.name)
                else canonicalize(getattr(value, item.name))
            )
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {
            str(key): (
                _sensitive_identity_token(item)
                if _is_secret_key(str(key))
                else canonicalize(item)
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        items = [canonicalize(item) for item in value]
        return sorted(items, key=canonical_json_bytes)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [canonicalize(item) for item in value]
    if inspect.isfunction(value) or inspect.ismethod(value):
        from .identity_components import automatic_callable_identity

        return automatic_callable_identity(value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        from .identity_components import automatic_component_identity

        return {
            "component": automatic_component_identity(value),
            "configuration": {
                key: (
                    _sensitive_identity_token(item)
                    if _is_secret_key(key)
                    else canonicalize(item)
                )
                for key, item in sorted(attributes.items())
                if not key.startswith("_")
            },
        }
    raise IdentityError(
        f"cannot build a stable automatic identity for {type(value).__qualname__}"
    )




def _sensitive_identity_token(value: Any) -> dict[str, Any]:
    """Retain identity sensitivity without serializing the original value."""

    if _is_sensitive_identity_token(value):
        return dict(value)
    payload = {
        "type": (
            f"{type(value).__module__}."
            f"{type(value).__qualname__}"
        ),
        "value": _private_identity_value(value),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "redacted": True,
        "identity_digest": sha256(
            b"icapa-sensitive-identity-v1\0" + encoded
        ).hexdigest(),
    }


def _is_sensitive_identity_token(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"redacted", "identity_digest"}
        and value.get("redacted") is True
        and isinstance(value.get("identity_digest"), str)
        and re.fullmatch(r"[0-9a-f]{64}", value["identity_digest"])
        is not None
    )


def _private_identity_value(value: Any) -> Any:
    """Canonicalize a value used only inside an irreversible digest."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IdentityError(
                "identity values must not contain non-finite numbers"
            )
        return value
    if isinstance(value, Enum):
        return _private_identity_value(value.value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise IdentityError(
                "identity values must not contain null dates"
            )
        return timestamp.isoformat()
    if isinstance(value, Path):
        return str(value.expanduser())
    if isinstance(value, (bytes, bytearray)):
        return {
            "bytes_sha256": sha256(bytes(value)).hexdigest(),
            "length": len(value),
        }
    if isinstance(value, pd.DataFrame):
        return {
            "dataframe_digest": dataframe_content_digest(value)
        }
    if isinstance(value, pd.Series):
        return {
            "series_name": _private_identity_value(value.name),
            "dataframe_digest": dataframe_content_digest(
                value.to_frame()
            ),
        }
    if isinstance(value, np.ndarray):
        return _private_identity_value(value.tolist())
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _private_identity_value(scalar())
        except (TypeError, ValueError):
            pass
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _private_identity_value(
                getattr(value, item.name)
            )
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {
            str(key): _private_identity_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (set, frozenset)):
        items = [_private_identity_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_private_identity_value(item) for item in value]
    if inspect.isfunction(value) or inspect.ismethod(value):
        from .identity_components import automatic_callable_identity

        return automatic_callable_identity(value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            "component_type": (
                f"{type(value).__module__}."
                f"{type(value).__qualname__}"
            ),
            "configuration": {
                str(key): _private_identity_value(item)
                for key, item in sorted(
                    attributes.items(),
                    key=lambda pair: str(pair[0]),
                )
                if not str(key).startswith("_")
            },
        }
    raise IdentityError(
        "cannot build a stable sensitive identity for "
        f"{type(value).__qualname__}"
    )


def _is_secret_key(key: str) -> bool:
    lowered = key.casefold()
    return any(part in lowered for part in _SECRET_PARTS)


def _is_sensitive_text(value: str) -> bool:
    """Detect connection material even when it is stored under a generic key."""

    return any(pattern.search(value) for pattern in _SENSITIVE_TEXT_PATTERNS)


# Preserve the British-English public name for existing integrations.
canonicalise = canonicalize


__all__ = [
    "IdentityError",
    "UnfingerprintableComponentError",
    "automatic_digest",
    "canonical_json_bytes",
    "canonicalise",
    "canonicalize",
    "dataframe_content_digest",
    "safe_parameter_identity",
]
