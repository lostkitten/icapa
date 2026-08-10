"""Immutable artifact contracts used by index-recipe stages.

The artifact namespace is intentionally open. Core stages can publish stable
keys while private or third-party stages use their own namespaces without
changing this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType
from collections.abc import Sequence
from typing import Any, Mapping, TypeAlias, Union

import numpy as np
import pandas as pd

from .fingerprints import callable_identity


JsonValue: TypeAlias = Union[
    None,
    bool,
    int,
    float,
    str,
    list["JsonValue"],
    dict[str, "JsonValue"],
]


@dataclass(frozen=True, order=True)
class ArtifactKey:
    """A namespaced, versioned identity for one recipe artifact."""

    namespace: str
    name: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for field_name in ("namespace", "name", "schema_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"artifact {field_name} must be a non-empty string")

    @property
    def canonical_name(self) -> str:
        """Return a stable serialization label."""

        return f"{self.namespace}:{self.name}@{self.schema_version}"

    def __str__(self) -> str:
        return self.canonical_name


@dataclass(frozen=True)
class ArtifactRequirement:
    """Declare an artifact consumed by a stage."""

    key: ArtifactKey
    optional: bool = False


@dataclass(frozen=True)
class ArtifactOutput:
    """Declare an artifact produced by a stage."""

    key: ArtifactKey
    optional: bool = False


@dataclass(frozen=True)
class Artifact:
    """A value with a stable digest and safe, immutable metadata."""

    key: ArtifactKey
    value: Any = field(repr=False, compare=False)
    digest: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        digest = self.digest or artifact_digest(self.value)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("artifact digest must be a 64-character SHA-256 hex string")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError("artifact digest must contain hexadecimal characters") from exc
        object.__setattr__(self, "digest", digest.lower())
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_value(
        cls,
        key: ArtifactKey,
        value: Any,
        *,
        digest: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Artifact":
        """Build an artifact, allowing custom values to supply their own digest."""

        return cls(
            key=key,
            value=value,
            digest=digest,
            metadata={} if metadata is None else metadata,
        )


CORE_CONSTITUENTS = ArtifactKey("icapa.core", "constituents")
CORE_FINAL_CONSTITUENTS = ArtifactKey("icapa.core", "final_constituents")
CORE_DAILY_DATA = ArtifactKey("icapa.core", "daily_data")
CORE_TARGET_WEIGHTS = ArtifactKey("icapa.core", "target_weights")
CORE_DIAGNOSTICS = ArtifactKey("icapa.core", "methodology_diagnostics")


def artifact_digest(value: Any) -> str:
    """Return a deterministic digest for standard artifact value types.

    Custom values may implement ``artifact_fingerprint()`` or provide an
    explicit digest to :class:`Artifact`.
    """

    fingerprint = getattr(value, "artifact_fingerprint", None)
    if callable(fingerprint):
        supplied = fingerprint()
        if isinstance(supplied, str) and len(supplied) == 64:
            return supplied.lower()
        return canonical_digest(supplied)
    if isinstance(value, pd.DataFrame):
        return canonical_digest(
            {
                "type": "dataframe",
                "dataframe_digest": dataframe_content_digest(value),
            }
        )
    if isinstance(value, pd.Series):
        return canonical_digest(
            {
                "type": "series",
                "name": canonicalize(value.name),
                "dataframe_digest": dataframe_content_digest(
                    value.to_frame()
                ),
            }
        )
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        payload = (
            str(array.dtype).encode("utf-8")
            + repr(array.shape).encode("utf-8")
            + np.ascontiguousarray(array).tobytes()
        )
        return sha256(payload).hexdigest()
    return canonical_digest(value)


def canonical_digest(value: Any) -> str:
    """Hash a value after deterministic JSON-safe normalization."""

    payload = json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def dataframe_content_digest(
    frame: pd.DataFrame,
    *,
    sort_by: Sequence[str] | None = None,
) -> str:
    """Hash DataFrame values, dtypes, columns, and index metadata."""

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
            raise KeyError(
                f"dataframe digest sort columns are missing: {sorted(missing)}"
            )
        flat = flat.sort_values(
            list(sort_by),
            kind="mergesort",
        ).reset_index(drop=True)
    columns: list[dict[str, Any]] = []
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
        columns.append(
            {
                "position": position,
                "name": canonicalize(flat.columns[position]),
                "dtype": str(series.dtype),
                "values_digest": sha256(hashes.tobytes()).hexdigest(),
            }
        )
    return canonical_digest(
        {
            "index_names": [] if implicit_range_index else original_index_names,
            "index_columns": safe_index_names,
            "row_count": int(len(flat)),
            "columns": columns,
        }
    )


def _canonical_object_hash_token(value: Any) -> str:
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
        allow_nan=False,
    )


def canonicalize(value: Any) -> JsonValue:
    """Return a deterministic JSON-safe representation.

    This function is deliberately conservative. Opaque configuration objects
    retain their qualified type and public attributes; custom stages can
    override this by returning their own canonical configuration.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical values must not contain non-finite numbers")
        return value
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return canonicalize(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: canonicalize(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        items = [canonicalize(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        )
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if callable(value):
        return canonicalize(callable_identity(value))
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            "type": qualified_name(value),
            "configuration": {
                str(key): canonicalize(item)
                for key, item in sorted(attributes.items())
                if not str(key).startswith("_")
            },
        }
    return {"type": qualified_name(value)}


def qualified_name(value: Any) -> str:
    """Return a stable qualified type or callable name."""

    target = value if isinstance(value, type) else type(value)
    if callable(value) and hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    return f"{target.__module__}.{target.__qualname__}"


__all__ = [
    "Artifact",
    "ArtifactKey",
    "ArtifactOutput",
    "ArtifactRequirement",
    "CORE_CONSTITUENTS",
    "CORE_DAILY_DATA",
    "CORE_DIAGNOSTICS",
    "CORE_FINAL_CONSTITUENTS",
    "CORE_TARGET_WEIGHTS",
    "JsonValue",
    "artifact_digest",
    "canonical_digest",
    "canonicalize",
    "dataframe_content_digest",
    "qualified_name",
]
