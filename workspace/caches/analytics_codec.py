"""Analytics result serialization for immutable workspace commits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from ...analytics.contracts import (
    AnalyticsDiagnostic,
    AnalyticsResult,
    BrinsonAttribution,
)
from ...analytics.plugins import (
    AnalyticsPluginResult,
    AnalyticsPluginSpec,
    AnalyticsRunResult,
    AnalyticsSpec,
    MissingInputPolicy,
    ReturnSeries,
)
from ..manifests import ArtifactRef
from .analytics_contracts import (
    LEGACY_FRAME_FIELDS as _LEGACY_FRAME_FIELDS,
    SCHEMA_VERSION as _SCHEMA_VERSION,
    SERIES_COLUMN as _SERIES_COLUMN,
    TYPE_TAG as _TAG,
    AnalyticsWorkspaceCacheIntegrityError,
    AnalyticsWorkspaceCacheSerializationError,
)
from .analytics_frames import (
    analytics_table_digest as _analytics_table_digest,
    pandas_roundtrip_schema as _pandas_roundtrip_schema,
)

@dataclass(slots=True)
class _PreparedTable:
    logical_id: str
    value_kind: str
    frame: pd.DataFrame
    series_name: Any = None
    binding: str | None = None
    reference: ArtifactRef | None = None

    def identity(self) -> dict[str, Any]:
        result = {
            "logical_id": self.logical_id,
            "value_kind": self.value_kind,
            "content_digest": _analytics_table_digest(self.frame),
            "pandas_schema": _pandas_roundtrip_schema(self.frame),
        }
        if self.value_kind == "series":
            result["series_name"] = _encode_value(
                self.series_name,
                path=f"tables[{self.logical_id}].series_name",
            )
        return result

    def commit_entry(self) -> dict[str, Any]:
        if self.binding is None or self.reference is None:
            raise AnalyticsWorkspaceCacheSerializationError(
                "analytics table was not persisted before commit"
            )
        result = {
            "logical_id": self.logical_id,
            "value_kind": self.value_kind,
            "binding": self.binding,
            "artifact": asdict(self.reference),
            "pandas_schema": _pandas_roundtrip_schema(self.frame),
        }
        if self.value_kind == "series":
            result["series_name"] = _encode_value(
                self.series_name,
                path=f"tables[{self.logical_id}].series_name",
            )
        return result


@dataclass(slots=True)
class _PreparedResult:
    payload: dict[str, Any]
    tables: list[_PreparedTable]
    result_digest: str




def _prepare_result(result: AnalyticsRunResult) -> _PreparedResult:
    if not isinstance(result, AnalyticsRunResult):
        raise TypeError("result must be an AnalyticsRunResult")
    tables: list[_PreparedTable] = []
    plugin_payload: list[dict[str, Any]] = []
    for plugin_position, (plugin_id, plugin_result) in enumerate(
        result.plugin_results.items()
    ):
        if not isinstance(plugin_id, str) or not plugin_id:
            raise AnalyticsWorkspaceCacheSerializationError(
                "analytics plugin result IDs must be non-empty strings"
            )
        if not isinstance(plugin_result, AnalyticsPluginResult):
            raise AnalyticsWorkspaceCacheSerializationError(
                f"plugin result {plugin_id!r} has an invalid type"
            )
        plugin_tables: list[dict[str, str]] = []
        for table_position, (table_name, frame) in enumerate(
            plugin_result.tables.items()
        ):
            if not isinstance(table_name, str) or not table_name:
                raise AnalyticsWorkspaceCacheSerializationError(
                    "analytics table names must be non-empty strings"
                )
            if not isinstance(frame, pd.DataFrame):
                raise AnalyticsWorkspaceCacheSerializationError(
                    f"analytics table {plugin_id}.{table_name} is not a DataFrame"
                )
            logical_id = (
                f"plugin:{plugin_position:04d}:{table_position:04d}"
            )
            tables.append(
                _PreparedTable(
                    logical_id=logical_id,
                    value_kind="dataframe",
                    frame=frame.copy(deep=True),
                )
            )
            plugin_tables.append(
                {"name": table_name, "logical_id": logical_id}
            )

        metadata: list[dict[str, Any]] = []
        legacy_metadata_reference = False
        for key, value in plugin_result.metadata.items():
            if not isinstance(key, str):
                raise AnalyticsWorkspaceCacheSerializationError(
                    f"plugin {plugin_id!r} metadata keys must be strings"
                )
            if key == "legacy_result" and isinstance(value, AnalyticsResult):
                if result.legacy_result is None:
                    raise AnalyticsWorkspaceCacheSerializationError(
                        "plugin metadata references a missing v1 result"
                    )
                if value is not result.legacy_result:
                    raise AnalyticsWorkspaceCacheSerializationError(
                        "plugin metadata references a different v1 result"
                    )
                legacy_metadata_reference = True
                continue
            metadata.append(
                {
                    "name": key,
                    "value": _encode_value(
                        value,
                        path=f"plugins[{plugin_id}].metadata.{key}",
                    ),
                }
            )
        plugin_payload.append(
            {
                "plugin_id": plugin_id,
                "metrics": _encode_named_values(
                    plugin_result.metrics,
                    path=f"plugins[{plugin_id}].metrics",
                ),
                "tables": plugin_tables,
                "diagnostics": _encode_diagnostics(
                    plugin_result.diagnostics
                ),
                "metadata": metadata,
                "legacy_metadata_reference": legacy_metadata_reference,
            }
        )

    legacy_payload = _prepare_legacy_result(
        result.legacy_result,
        tables=tables,
    )
    payload = {
        "spec": _encode_spec(result.spec),
        "plugins": plugin_payload,
        "legacy_result": legacy_payload,
        "diagnostics": _encode_diagnostics(result.diagnostics),
    }
    digest = _json_digest(
        {
            "payload": payload,
            "tables": [table.identity() for table in tables],
        }
    )
    return _PreparedResult(
        payload=payload,
        tables=tables,
        result_digest=digest,
    )


def _prepare_legacy_result(
    result: AnalyticsResult | None,
    *,
    tables: list[_PreparedTable],
) -> dict[str, Any] | None:
    if result is None:
        return None
    if not isinstance(result, AnalyticsResult):
        raise AnalyticsWorkspaceCacheSerializationError(
            "legacy_result must be an AnalyticsResult or None"
        )
    fields_payload: dict[str, str] = {}
    for field_name in _LEGACY_FRAME_FIELDS:
        value = getattr(result, field_name)
        if not isinstance(value, pd.DataFrame):
            raise AnalyticsWorkspaceCacheSerializationError(
                f"v1 analytics field {field_name} is not a DataFrame"
            )
        logical_id = f"legacy:{field_name}"
        tables.append(
            _PreparedTable(
                logical_id=logical_id,
                value_kind="dataframe",
                frame=value.copy(deep=True),
            )
        )
        fields_payload[field_name] = logical_id

    if not isinstance(result.performance, pd.Series):
        raise AnalyticsWorkspaceCacheSerializationError(
            "v1 analytics performance is not a Series"
        )
    performance_id = "legacy:performance"
    tables.append(
        _PreparedTable(
            logical_id=performance_id,
            value_kind="series",
            frame=result.performance.to_frame(name=_SERIES_COLUMN),
            series_name=result.performance.name,
        )
    )
    brinson_payload = None
    if result.brinson is not None:
        if not isinstance(result.brinson, BrinsonAttribution):
            raise AnalyticsWorkspaceCacheSerializationError(
                "v1 analytics Brinson value has an invalid type"
            )
        brinson_payload = {}
        for field_name in ("detail", "totals"):
            logical_id = f"legacy:brinson:{field_name}"
            tables.append(
                _PreparedTable(
                    logical_id=logical_id,
                    value_kind="dataframe",
                    frame=getattr(result.brinson, field_name).copy(deep=True),
                )
            )
            brinson_payload[field_name] = logical_id
    return {
        "frames": fields_payload,
        "performance": performance_id,
        "brinson": brinson_payload,
        "diagnostics": _encode_diagnostics(result.diagnostics),
    }


def _restore_result(
    payload: object,
    tables: Mapping[str, pd.DataFrame | pd.Series],
) -> AnalyticsRunResult:
    if not isinstance(payload, Mapping):
        raise AnalyticsWorkspaceCacheIntegrityError(
            "analytics result payload must be an object"
        )
    spec = _decode_spec(payload["spec"])
    legacy_result = _restore_legacy_result(
        payload.get("legacy_result"),
        tables=tables,
    )
    plugin_entries = payload["plugins"]
    if not isinstance(plugin_entries, list):
        raise AnalyticsWorkspaceCacheIntegrityError(
            "analytics plugin entries must be a list"
        )
    plugin_results: dict[str, AnalyticsPluginResult] = {}
    for position, entry in enumerate(plugin_entries):
        if not isinstance(entry, Mapping):
            raise AnalyticsWorkspaceCacheIntegrityError(
                "analytics plugin entry must be an object"
            )
        plugin_id = entry["plugin_id"]
        if not isinstance(plugin_id, str) or not plugin_id:
            raise AnalyticsWorkspaceCacheIntegrityError(
                "analytics plugin ID must be a non-empty string"
            )
        if plugin_id in plugin_results:
            raise AnalyticsWorkspaceCacheIntegrityError(
                "analytics cache contains duplicate plugin results"
            )
        table_entries = entry["tables"]
        if not isinstance(table_entries, list):
            raise AnalyticsWorkspaceCacheIntegrityError(
                "analytics plugin tables must be a list"
            )
        restored_tables: dict[str, pd.DataFrame] = {}
        for table_entry in table_entries:
            if not isinstance(table_entry, Mapping):
                raise AnalyticsWorkspaceCacheIntegrityError(
                    "analytics plugin table reference must be an object"
                )
            table_name = table_entry["name"]
            logical_id = table_entry["logical_id"]
            value = tables[logical_id]
            if (
                not isinstance(table_name, str)
                or not isinstance(value, pd.DataFrame)
            ):
                raise AnalyticsWorkspaceCacheIntegrityError(
                    "analytics plugin table reference has an invalid schema"
                )
            restored_tables[table_name] = value
        metadata_payload = entry["metadata"]
        if not isinstance(metadata_payload, list):
            raise AnalyticsWorkspaceCacheIntegrityError(
                "analytics plugin metadata must be a list"
            )
        metadata = _decode_named_values(
            metadata_payload,
            path=f"plugins[{position}].metadata",
        )
        if entry.get("legacy_metadata_reference"):
            if legacy_result is None:
                raise AnalyticsWorkspaceCacheIntegrityError(
                    "analytics plugin metadata references a missing v1 result"
                )
            metadata["legacy_result"] = legacy_result
        metrics_payload = entry["metrics"]
        if not isinstance(metrics_payload, list):
            raise AnalyticsWorkspaceCacheIntegrityError(
                "analytics plugin metrics must be a list"
            )
        plugin_results[plugin_id] = AnalyticsPluginResult(
            metrics=_decode_named_values(
                metrics_payload,
                path=f"plugins[{position}].metrics",
            ),
            tables=restored_tables,
            diagnostics=_decode_diagnostics(entry["diagnostics"]),
            metadata=metadata,
        )
    return AnalyticsRunResult(
        spec=spec,
        plugin_results=plugin_results,
        legacy_result=legacy_result,
        diagnostics=_decode_diagnostics(payload["diagnostics"]),
    )


def _restore_legacy_result(
    payload: object,
    *,
    tables: Mapping[str, pd.DataFrame | pd.Series],
) -> AnalyticsResult | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise AnalyticsWorkspaceCacheIntegrityError(
            "v1 analytics payload must be an object"
        )
    frames_payload = payload["frames"]
    if not isinstance(frames_payload, Mapping):
        raise AnalyticsWorkspaceCacheIntegrityError(
            "v1 analytics frame references must be an object"
        )
    frames: dict[str, pd.DataFrame] = {}
    for field_name in _LEGACY_FRAME_FIELDS:
        value = tables[frames_payload[field_name]]
        if not isinstance(value, pd.DataFrame):
            raise AnalyticsWorkspaceCacheIntegrityError(
                f"v1 analytics field {field_name} is not a DataFrame"
            )
        frames[field_name] = value
    performance = tables[payload["performance"]]
    if not isinstance(performance, pd.Series):
        raise AnalyticsWorkspaceCacheIntegrityError(
            "v1 analytics performance is not a Series"
        )
    brinson_payload = payload.get("brinson")
    brinson = None
    if brinson_payload is not None:
        if not isinstance(brinson_payload, Mapping):
            raise AnalyticsWorkspaceCacheIntegrityError(
                "v1 Brinson references must be an object"
            )
        detail = tables[brinson_payload["detail"]]
        totals = tables[brinson_payload["totals"]]
        if not isinstance(detail, pd.DataFrame) or not isinstance(
            totals,
            pd.DataFrame,
        ):
            raise AnalyticsWorkspaceCacheIntegrityError(
                "v1 Brinson artifacts must be DataFrames"
            )
        brinson = BrinsonAttribution(detail=detail, totals=totals)
    return AnalyticsResult(
        **frames,
        performance=performance,
        brinson=brinson,
        diagnostics=_decode_diagnostics(payload["diagnostics"]),
    )


def _encode_spec(spec: AnalyticsSpec) -> dict[str, Any]:
    if not isinstance(spec, AnalyticsSpec):
        raise AnalyticsWorkspaceCacheSerializationError(
            "analytics result spec must be an AnalyticsSpec"
        )
    return {
        "profile": spec.profile,
        "plugins": [
            {
                "plugin_id": item.plugin_id,
                "version": item.version,
                "parameters": _encode_string_mapping(
                    item.parameters,
                    path=f"spec.plugins[{position}].parameters",
                ),
                "required": item.required,
            }
            for position, item in enumerate(spec.plugins)
        ],
        "return_series": spec.return_series.value,
        "annualization_factor": spec.annualization_factor,
        "weight_tolerance": spec.weight_tolerance,
        "missing_optional_input": spec.missing_optional_input.value,
    }


def _decode_spec(payload: object) -> AnalyticsSpec:
    if not isinstance(payload, Mapping):
        raise AnalyticsWorkspaceCacheIntegrityError(
            "analytics specification must be an object"
        )
    plugins_payload = payload["plugins"]
    if not isinstance(plugins_payload, list):
        raise AnalyticsWorkspaceCacheIntegrityError(
            "analytics plugin specification must be a list"
        )
    plugins = []
    for position, item in enumerate(plugins_payload):
        if not isinstance(item, Mapping):
            raise AnalyticsWorkspaceCacheIntegrityError(
                "analytics plugin specification must be an object"
            )
        parameters = item["parameters"]
        if not isinstance(parameters, Mapping):
            raise AnalyticsWorkspaceCacheIntegrityError(
                "analytics plugin parameters must be an object"
            )
        plugins.append(
            AnalyticsPluginSpec(
                plugin_id=item["plugin_id"],
                version=item["version"],
                parameters={
                    str(key): _decode_value(
                        value,
                        path=f"spec.plugins[{position}].parameters.{key}",
                    )
                    for key, value in parameters.items()
                },
                required=item["required"],
            )
        )
    return AnalyticsSpec(
        profile=payload["profile"],
        plugins=tuple(plugins),
        return_series=ReturnSeries(payload["return_series"]),
        annualization_factor=payload["annualization_factor"],
        weight_tolerance=payload["weight_tolerance"],
        missing_optional_input=MissingInputPolicy(
            payload["missing_optional_input"]
        ),
    )


def _encode_diagnostics(
    diagnostics: tuple[AnalyticsDiagnostic, ...],
) -> list[dict[str, str]]:
    result = []
    for item in diagnostics:
        if not isinstance(item, AnalyticsDiagnostic):
            raise AnalyticsWorkspaceCacheSerializationError(
                "analytics diagnostics must contain AnalyticsDiagnostic values"
            )
        result.append(
            {
                "level": item.level,
                "code": item.code,
                "message": item.message,
            }
        )
    return result


def _decode_diagnostics(payload: object) -> tuple[AnalyticsDiagnostic, ...]:
    if not isinstance(payload, list):
        raise AnalyticsWorkspaceCacheIntegrityError(
            "analytics diagnostics must be a list"
        )
    result = []
    for position, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise AnalyticsWorkspaceCacheIntegrityError(
                f"analytics diagnostic {position} must be an object"
            )
        result.append(
            AnalyticsDiagnostic(
                level=item["level"],
                code=item["code"],
                message=item["message"],
            )
        )
    return tuple(result)


def _encode_string_mapping(
    value: Mapping[str, Any],
    *,
    path: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalyticsWorkspaceCacheSerializationError(
            f"{path} must be a mapping"
        )
    encoded: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise AnalyticsWorkspaceCacheSerializationError(
                f"{path} contains a non-string key"
            )
        encoded[key] = _encode_value(item, path=f"{path}.{key}")
    return encoded


def _encode_named_values(
    value: Mapping[str, Any],
    *,
    path: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise AnalyticsWorkspaceCacheSerializationError(
            f"{path} must be a mapping"
        )
    result = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise AnalyticsWorkspaceCacheSerializationError(
                f"{path} contains a non-string key"
            )
        result.append(
            {
                "name": key,
                "value": _encode_value(item, path=f"{path}.{key}"),
            }
        )
    return result


def _decode_named_values(
    value: list[Any],
    *,
    path: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for position, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise AnalyticsWorkspaceCacheIntegrityError(
                f"{path}[{position}] must be an object"
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in result:
            raise AnalyticsWorkspaceCacheIntegrityError(
                f"{path}[{position}] has an invalid or duplicate name"
            )
        result[name] = _decode_value(
            entry.get("value"),
            path=f"{path}.{name}",
        )
    return result


def _encode_value(value: Any, *, path: str) -> Any:
    """Encode compact analytics metadata without lossy string coercion."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {_TAG: "float", "value": "nan"}
        if math.isinf(value):
            return {
                _TAG: "float",
                "value": "positive_infinity" if value > 0 else "negative_infinity",
            }
        return value
    if isinstance(value, Enum):
        return {
            _TAG: "enum_value",
            "value": _encode_value(value.value, path=f"{path}.value"),
        }
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return {_TAG: "timestamp", "value": "nat"}
        return {_TAG: "timestamp", "value": timestamp.isoformat()}
    if isinstance(value, np.generic):
        return _encode_value(value.item(), path=path)
    if isinstance(value, tuple):
        return {
            _TAG: "tuple",
            "items": [
                _encode_value(item, path=f"{path}[{position}]")
                for position, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return [
            _encode_value(item, path=f"{path}[{position}]")
            for position, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return {
            _TAG: "mapping",
            "items": _encode_named_values(value, path=path),
        }
    raise AnalyticsWorkspaceCacheSerializationError(
        f"{path} contains unsupported type {type(value).__qualname__}"
    )


def _decode_value(value: Any, *, path: str) -> Any:
    if isinstance(value, list):
        return [
            _decode_value(item, path=f"{path}[{position}]")
            for position, item in enumerate(value)
        ]
    if not isinstance(value, Mapping):
        return value
    tag = value.get(_TAG)
    if tag is None:
        return {
            str(key): _decode_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if tag == "float":
        selected = value.get("value")
        if selected == "nan":
            return float("nan")
        if selected == "positive_infinity":
            return float("inf")
        if selected == "negative_infinity":
            return float("-inf")
    elif tag == "timestamp":
        selected = value.get("value")
        return pd.NaT if selected == "nat" else pd.Timestamp(selected)
    elif tag == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise AnalyticsWorkspaceCacheIntegrityError(
                f"{path} contains an invalid tuple encoding"
            )
        return tuple(
            _decode_value(item, path=f"{path}[{position}]")
            for position, item in enumerate(items)
        )
    elif tag == "enum_value":
        return _decode_value(value.get("value"), path=f"{path}.value")
    elif tag == "mapping":
        items = value.get("items")
        if not isinstance(items, list):
            raise AnalyticsWorkspaceCacheIntegrityError(
                f"{path} contains an invalid mapping encoding"
            )
        return _decode_named_values(items, path=path)
    raise AnalyticsWorkspaceCacheIntegrityError(
        f"{path} contains an unsupported metadata tag"
    )


def _decode_commit(
    frame: pd.DataFrame,
    *,
    expected_key: str,
) -> dict[str, Any]:
    if list(frame.columns) != ["payload_json"] or len(frame) != 1:
        raise AnalyticsWorkspaceCacheIntegrityError(
            "analytics cache commit has an invalid Parquet schema"
        )
    encoded = frame.iloc[0]["payload_json"]
    if not isinstance(encoded, str):
        raise AnalyticsWorkspaceCacheIntegrityError(
            "analytics cache commit payload must be text"
        )
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise AnalyticsWorkspaceCacheIntegrityError(
            "analytics cache commit payload is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise AnalyticsWorkspaceCacheIntegrityError(
            "analytics cache commit payload must be an object"
        )
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise AnalyticsWorkspaceCacheIntegrityError(
            "analytics cache schema version is not supported"
        )
    if payload.get("cache_key") != expected_key:
        raise AnalyticsWorkspaceCacheIntegrityError(
            "analytics cache commit is bound to a different identity"
        )
    checksum = payload.get("metadata_checksum")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "metadata_checksum"
    }
    if not isinstance(checksum, str) or checksum != _json_digest(unsigned):
        raise AnalyticsWorkspaceCacheIntegrityError(
            "analytics cache commit checksum does not match its metadata"
        )
    required = {"result_digest", "result", "tables"}
    if not required.issubset(payload):
        raise AnalyticsWorkspaceCacheIntegrityError(
            "analytics cache commit is missing required fields"
        )
    return payload


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AnalyticsWorkspaceCacheSerializationError(
            "analytics cache metadata is not JSON-safe"
        ) from exc


def _json_digest(value: Any) -> str:
    return sha256(_json_bytes(value)).hexdigest()



__all__ = [
    "decode_commit",
    "decode_value",
    "json_bytes",
    "json_digest",
    "prepare_result",
    "restore_result",
]


decode_commit = _decode_commit
decode_value = _decode_value
json_bytes = _json_bytes
json_digest = _json_digest
prepare_result = _prepare_result
restore_result = _restore_result
