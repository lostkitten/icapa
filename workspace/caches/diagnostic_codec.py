"""Scalar and container codecs for persisted review diagnostics."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
import math
from typing import Any

import numpy as np
import pandas as pd

from .models import CacheStage
from .diagnostic_tables import (
    _restore_diagnostic_table,
    _validated_numpy_temporal_unit,
    _validated_temporal_unit,
)
from .diagnostic_types import (
    DIAGNOSTIC_ENUM_REGISTRY as _DIAGNOSTIC_ENUM_REGISTRY,
    DIAGNOSTIC_TAG as _DIAGNOSTIC_TAG,
    DiagnosticTable as _DiagnosticTable,
    MAX_DIAGNOSTIC_DEPTH as _MAX_DIAGNOSTIC_DEPTH,
    ReviewDiagnosticSerializationError as _ReviewDiagnosticSerializationError,
)
from ..manifests import ArtifactRef, ManifestIntegrityError
from ..readers import CacheMissError


class DiagnosticDecodingMixin:
    """Decode persisted diagnostic values through a stage store."""

    def _decode_diagnostic_value(
        self,
        value: Any,
        *,
        cache_key: str,
        path: str,
        depth: int = 0,
        allow_table: bool = True,
    ) -> Any:
        if depth > _MAX_DIAGNOSTIC_DEPTH:
            raise ValueError("review diagnostic metadata exceeds maximum depth")
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} has an invalid diagnostic encoding")
        tag = value.get(_DIAGNOSTIC_TAG)
        if tag == "mapping":
            entries = value.get("items")
            if not isinstance(entries, list):
                raise ValueError(f"{path} mapping items must be a list")
            restored: dict[Any, Any] = {}
            for position, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    raise ValueError(
                        f"{path}.items[{position}] must be a mapping"
                    )
                key = self._decode_diagnostic_value(
                    entry.get("key"),
                    cache_key=cache_key,
                    path=f"{path}.items[{position}].key",
                    depth=depth + 1,
                    allow_table=False,
                )
                if not isinstance(key, Hashable):
                    raise ValueError(
                        f"{path}.items[{position}].key is not hashable"
                    )
                if key in restored:
                    raise ValueError(f"{path} contains duplicate mapping keys")
                restored[key] = self._decode_diagnostic_value(
                    entry.get("value"),
                    cache_key=cache_key,
                    path=f"{path}.items[{position}].value",
                    depth=depth + 1,
                    allow_table=allow_table,
                )
            return restored
        if tag in {"list", "tuple"}:
            items = value.get("items")
            if not isinstance(items, list):
                raise ValueError(f"{path} sequence items must be a list")
            restored_items = [
                self._decode_diagnostic_value(
                    item,
                    cache_key=cache_key,
                    path=f"{path}[{position}]",
                    depth=depth + 1,
                    allow_table=allow_table,
                )
                for position, item in enumerate(items)
            ]
            return restored_items if tag == "list" else tuple(restored_items)
        if tag in {"dataframe", "series"}:
            if not allow_table:
                raise ValueError(f"{path} cannot contain a table reference")
            return self._decode_diagnostic_table(
                value,
                cache_key=cache_key,
                path=path,
            )
        return _decode_diagnostic_scalar(
            value,
            tag=tag,
            path=path,
        )

    def _decode_diagnostic_table(
        self,
        value: Mapping[str, Any],
        *,
        cache_key: str,
        path: str,
    ) -> pd.DataFrame | pd.Series:
        binding = value.get("binding")
        artifact_payload = value.get("artifact")
        if not isinstance(binding, str) or not isinstance(
            artifact_payload,
            Mapping,
        ):
            raise ValueError(f"{path} table reference is incomplete")
        try:
            expected = ArtifactRef.from_dict(artifact_payload)
            reference = self.workspace.resolve_artifact(
                stage=CacheStage.REVIEWS,
                cache_key=cache_key,
                name=binding,
            )
        except ManifestIntegrityError as exc:
            if "does not exist in its workspace" in str(exc):
                raise CacheMissError(
                    f"cached review diagnostic artifact is missing: {path}"
                ) from exc
            raise
        if reference is None:
            raise CacheMissError(
                f"cached review diagnostic binding is missing: {path}"
            )
        if not _same_artifact(reference, expected):
            raise ValueError(
                f"cached review diagnostic binding does not match: {path}"
            )
        storage = self._load_reference(reference)
        return _restore_diagnostic_table(
            storage,
            value.get("pandas_schema"),
            value_kind=str(value.get(_DIAGNOSTIC_TAG)),
            decode=lambda item, item_path: self._decode_diagnostic_value(
                item,
                cache_key=cache_key,
                path=item_path,
                allow_table=False,
            ),
            path=path,
        )



def _prepare_diagnostics(
    diagnostics: Mapping[str, Any],
) -> tuple[dict[str, Any], list[_DiagnosticTable]]:
    if not isinstance(diagnostics, Mapping):
        raise _ReviewDiagnosticSerializationError(
            "review diagnostics must be a mapping"
        )
    encoder = _DiagnosticEncoder()
    encoded = encoder.encode(diagnostics, path="diagnostics")
    if not isinstance(encoded, dict):
        raise _ReviewDiagnosticSerializationError(
            "review diagnostics did not encode as a mapping"
        )
    return encoded, encoder.tables


class _DiagnosticEncoder:
    def __init__(self) -> None:
        self.tables: list[_DiagnosticTable] = []
        self._active_containers: set[int] = set()

    def encode(
        self,
        value: Any,
        *,
        path: str,
        depth: int = 0,
        allow_table: bool = True,
    ) -> Any:
        if depth > _MAX_DIAGNOSTIC_DEPTH:
            raise _ReviewDiagnosticSerializationError(
                "review diagnostic metadata exceeds maximum depth"
            )
        if isinstance(value, Enum):
            enum_type = type(value)
            enum_id = _diagnostic_enum_id(enum_type)
            if _DIAGNOSTIC_ENUM_REGISTRY.get(enum_id) is not enum_type:
                raise _ReviewDiagnosticSerializationError(
                    f"{path} contains unregistered diagnostic Enum {enum_id}; "
                    "register it before writing workspace diagnostics"
                )
            return {
                _DIAGNOSTIC_TAG: "enum",
                "enum_id": enum_id,
                "member": value.name,
            }
        if value is pd.NA:
            return {_DIAGNOSTIC_TAG: "pandas_na"}
        if value is pd.NaT:
            return {_DIAGNOSTIC_TAG: "timestamp", "value": "nat", "unit": "ns"}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if math.isnan(value):
                return {_DIAGNOSTIC_TAG: "float", "value": "nan"}
            if math.isinf(value):
                return {
                    _DIAGNOSTIC_TAG: "float",
                    "value": (
                        "positive_infinity"
                        if value > 0
                        else "negative_infinity"
                    ),
                }
            return value
        if isinstance(value, pd.Timestamp):
            return {
                _DIAGNOSTIC_TAG: "timestamp",
                "value": value.isoformat(),
                "unit": value.unit,
            }
        if isinstance(value, np.datetime64):
            if np.isnat(value):
                return {
                    _DIAGNOSTIC_TAG: "numpy_datetime",
                    "value": "nat",
                    "unit": np.datetime_data(value.dtype)[0],
                }
            return {
                _DIAGNOSTIC_TAG: "numpy_datetime",
                "value": str(value),
                "unit": np.datetime_data(value.dtype)[0],
            }
        if isinstance(value, datetime):
            return {
                _DIAGNOSTIC_TAG: "datetime",
                "value": value.isoformat(),
            }
        if isinstance(value, date):
            return {
                _DIAGNOSTIC_TAG: "date",
                "value": value.isoformat(),
            }
        if isinstance(value, pd.Timedelta):
            return {
                _DIAGNOSTIC_TAG: "timedelta",
                "value": int(value.value),
                "unit": "ns",
            }
        if isinstance(value, np.timedelta64):
            unit = np.datetime_data(value.dtype)[0]
            return {
                _DIAGNOSTIC_TAG: "numpy_timedelta",
                "value": (
                    "nat"
                    if np.isnat(value)
                    else int(
                        value.astype(f"timedelta64[{unit}]").astype(np.int64)
                    )
                ),
                "unit": unit,
            }
        if isinstance(value, Decimal):
            if not value.is_finite():
                raise _ReviewDiagnosticSerializationError(
                    f"{path} contains a non-finite Decimal"
                )
            return {
                _DIAGNOSTIC_TAG: "decimal",
                "value": str(value),
            }
        if isinstance(value, np.generic):
            return self.encode(
                value.item(),
                path=path,
                depth=depth,
                allow_table=allow_table,
            )
        if isinstance(value, (pd.DataFrame, pd.Series)):
            if not allow_table:
                raise _ReviewDiagnosticSerializationError(
                    f"{path} cannot contain a tabular label or mapping key"
                )
            tag = "dataframe" if isinstance(value, pd.DataFrame) else "series"
            node: dict[str, Any] = {
                _DIAGNOSTIC_TAG: tag,
                "position": len(self.tables),
            }
            self.tables.append(
                _DiagnosticTable(
                    value=value.copy(deep=True),
                    node=node,
                    path=path,
                )
            )
            return node
        if isinstance(value, Mapping):
            return self._encode_mapping(
                value,
                path=path,
                depth=depth,
                allow_table=allow_table,
            )
        if isinstance(value, (list, tuple)):
            return self._encode_sequence(
                value,
                path=path,
                depth=depth,
                allow_table=allow_table,
            )
        if isinstance(value, Sequence):
            raise _ReviewDiagnosticSerializationError(
                f"{path} contains unsupported sequence type "
                f"{type(value).__qualname__}; use a list or tuple"
            )
        raise _ReviewDiagnosticSerializationError(
            f"{path} contains unsupported diagnostic value type "
            f"{type(value).__qualname__}"
        )

    def _encode_mapping(
        self,
        value: Mapping[Any, Any],
        *,
        path: str,
        depth: int,
        allow_table: bool,
    ) -> dict[str, Any]:
        marker = id(value)
        self._enter_container(marker, path)
        try:
            items = []
            for position, (key, item) in enumerate(value.items()):
                items.append(
                    {
                        "key": self.encode(
                            key,
                            path=f"{path}.items[{position}].key",
                            depth=depth + 1,
                            allow_table=False,
                        ),
                        "value": self.encode(
                            item,
                            path=f"{path}.items[{position}].value",
                            depth=depth + 1,
                            allow_table=allow_table,
                        ),
                    }
                )
            return {
                _DIAGNOSTIC_TAG: "mapping",
                "items": items,
            }
        finally:
            self._active_containers.remove(marker)

    def _encode_sequence(
        self,
        value: list[Any] | tuple[Any, ...],
        *,
        path: str,
        depth: int,
        allow_table: bool,
    ) -> dict[str, Any]:
        marker = id(value)
        self._enter_container(marker, path)
        try:
            return {
                _DIAGNOSTIC_TAG: (
                    "tuple" if isinstance(value, tuple) else "list"
                ),
                "items": [
                    self.encode(
                        item,
                        path=f"{path}[{position}]",
                        depth=depth + 1,
                        allow_table=allow_table,
                    )
                    for position, item in enumerate(value)
                ],
            }
        finally:
            self._active_containers.remove(marker)

    def _enter_container(self, marker: int, path: str) -> None:
        if marker in self._active_containers:
            raise _ReviewDiagnosticSerializationError(
                f"{path} contains a recursive diagnostic structure"
            )
        self._active_containers.add(marker)


def _decode_diagnostic_scalar(
    value: Mapping[str, Any],
    *,
    tag: Any,
    path: str,
) -> Any:
    if tag == "float":
        selected = value.get("value")
        if selected == "nan":
            return float("nan")
        if selected == "positive_infinity":
            return float("inf")
        if selected == "negative_infinity":
            return float("-inf")
    elif tag == "pandas_na":
        return pd.NA
    elif tag == "timestamp":
        selected = value.get("value")
        unit = _validated_temporal_unit(value.get("unit"), path=path)
        if selected == "nat":
            return pd.NaT
        return pd.Timestamp(selected).as_unit(unit)
    elif tag == "numpy_datetime":
        selected = value.get("value")
        unit = _validated_numpy_temporal_unit(value.get("unit"), path=path)
        return np.datetime64("NaT", unit) if selected == "nat" else np.datetime64(
            selected,
            unit,
        )
    elif tag == "datetime":
        selected = value.get("value")
        if isinstance(selected, str):
            return datetime.fromisoformat(selected)
    elif tag == "date":
        selected = value.get("value")
        if isinstance(selected, str):
            return date.fromisoformat(selected)
    elif tag == "timedelta":
        return pd.Timedelta(value.get("value"), unit="ns")
    elif tag == "numpy_timedelta":
        selected = value.get("value")
        unit = _validated_numpy_temporal_unit(value.get("unit"), path=path)
        return (
            np.timedelta64("NaT", unit)
            if selected == "nat"
            else np.timedelta64(selected, unit)
        )
    elif tag == "decimal":
        try:
            return Decimal(value.get("value"))
        except (InvalidOperation, TypeError) as exc:
            raise ValueError(f"{path} contains an invalid Decimal") from exc
    elif tag == "enum":
        return _restore_enum(value, path=path)
    raise ValueError(f"{path} contains an unsupported diagnostic type tag")


def _restore_enum(value: Mapping[str, Any], *, path: str) -> Enum:
    enum_id = value.get("enum_id")
    if enum_id is None:
        module_name = value.get("module")
        qualified_name = value.get("qualname")
        if not all(
            isinstance(item, str) and item
            for item in (module_name, qualified_name)
        ):
            raise ValueError(f"{path} contains an invalid enum reference")
        enum_id = f"{module_name}:{qualified_name}"
    member_name = value.get("member")
    if (
        not isinstance(enum_id, str)
        or not enum_id
        or not isinstance(member_name, str)
        or not member_name
    ):
        raise ValueError(f"{path} contains an invalid enum reference")
    target = _DIAGNOSTIC_ENUM_REGISTRY.get(enum_id)
    if target is None:
        raise ValueError(
            f"{path} diagnostic Enum is not registered: {enum_id}"
        )
    try:
        return target[member_name]
    except KeyError as exc:
        raise ValueError(f"{path} enum member is unavailable") from exc


def _diagnostic_enum_id(enum_type: type[Enum]) -> str:
    return f"{enum_type.__module__}:{enum_type.__qualname__}"




def _same_artifact(first: ArtifactRef, second: ArtifactRef) -> bool:
    return (
        first.content_digest == second.content_digest
        and first.file_checksum == second.file_checksum
        and first.relative_path == second.relative_path
        and first.format == second.format
        and first.schema_version == second.schema_version
        and first.size_bytes == second.size_bytes
    )




__all__ = [
    "DiagnosticDecodingMixin",
    "_prepare_diagnostics",
]
