"""Secret-safe identities for provider calls and returned tabular data."""

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
    "secret",
    "sql",
    "token",
    "uri",
    "url",
    "user",
)


def automatic_data_identity(
    *,
    provider_name: str,
    provider: object,
    capability: str,
    request: Mapping[str, Any],
    frame: pd.DataFrame,
    sort_by: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Identify one provider response by safe metadata and logical content."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("automatic data identity requires a pandas DataFrame")
    safe_request = {
        str(key): canonicalize(value)
        for key, value in request.items()
        if not is_secret_key(str(key))
    }
    return {
        "provider": automatic_provider_identity(
            provider_name,
            provider,
            capability=capability,
            parameters=request,
        ),
        "request_digest": automatic_digest(safe_request),
        "content_digest": dataframe_content_digest(frame, sort_by=sort_by),
        "rows": int(len(frame)),
        "columns": list(map(str, frame.columns)),
    }


def automatic_provider_identity(
    provider_name: str,
    provider: object,
    *,
    capability: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return non-sensitive provider implementation and parameter metadata."""

    if not isinstance(provider_name, str) or not provider_name.strip():
        raise ValueError("provider_name must not be empty")
    if not isinstance(capability, str) or not capability.strip():
        raise ValueError("capability must not be empty")
    return {
        "provider_name": provider_name.strip().lower(),
        "capability": capability.strip(),
        "component": component_identity(provider),
        "parameters": safe_parameter_identity(parameters),
    }


def component_identity(component: object) -> dict[str, Any]:
    """Identify provider adapter code without persisting its connection state."""

    target = component if inspect.isclass(component) else type(component)
    type_name = f"{target.__module__}.{target.__qualname__}"
    try:
        raw_path = inspect.getsourcefile(target) or inspect.getfile(target)
    except (OSError, TypeError):
        raw_path = None
    source_name = None
    source_digest = None
    if raw_path:
        path = Path(raw_path)
        if path.is_file():
            source_name = path.name
            source_digest = sha256(path.read_bytes()).hexdigest()
    return {
        "type": type_name,
        "source_file": source_name,
        "source_digest": source_digest,
    }


def safe_parameter_identity(
    parameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Describe provider parameters without serializing credential values."""

    values = dict(parameters or {})
    semantic = {
        str(key): canonicalize(value)
        for key, value in sorted(values.items(), key=lambda item: str(item[0]))
        if not is_secret_key(str(key))
    }
    return {
        "keys": sorted(map(str, values)),
        "redacted_keys": sorted(
            str(key) for key in values if is_secret_key(str(key))
        ),
        "semantic_digest": automatic_digest(semantic),
    }


def dataframe_content_digest(
    frame: pd.DataFrame,
    *,
    sort_by: Sequence[str] | None = None,
) -> str:
    """Hash DataFrame values, dtypes, columns, and index metadata."""

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
            raise KeyError(
                f"dataframe digest sort columns are missing: {sorted(missing)}"
            )
        flat = flat.sort_values(
            list(sort_by),
            kind="mergesort",
        ).reset_index(drop=True)
    column_digests = []
    for position in range(len(flat.columns)):
        series = flat.iloc[:, position]
        hash_input = (
            series.map(_object_hash_token)
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
    return automatic_digest(
        {
            "index_names": [] if implicit_range_index else original_index_names,
            "index_columns": safe_index_names,
            "row_count": int(len(flat)),
            "columns": column_digests,
        }
    )


def automatic_digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest."""

    payload = json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def canonicalize(value: Any) -> Any:
    """Normalize supported identity values without exposing secrets."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identity values must be finite")
        return value
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError("identity dates must not be null")
        return timestamp.isoformat()
    if isinstance(value, Path):
        return {
            "file_name": value.name,
            "path_digest": sha256(str(value).encode("utf-8")).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        return canonicalize(value.tolist())
    if isinstance(value, np.generic):
        return canonicalize(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: canonicalize(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_") and not is_secret_key(item.name)
        }
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not is_secret_key(str(key))
        }
    if isinstance(value, (set, frozenset)):
        items = [canonicalize(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [canonicalize(item) for item in value]
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def is_secret_key(key: str) -> bool:
    """Return whether a parameter name denotes connection or secret material."""

    normalized = key.casefold().replace("-", "_").replace(" ", "_")
    return any(part in normalized for part in _SECRET_PARTS)


def _object_hash_token(value: Any) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        payload: Any = {"missing": True}
    elif isinstance(value, float) and math.isnan(value):
        payload = {"missing": True}
    else:
        payload = {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": canonicalize(value),
        }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


__all__ = [
    "automatic_data_identity",
    "automatic_provider_identity",
    "dataframe_content_digest",
    "safe_parameter_identity",
]
