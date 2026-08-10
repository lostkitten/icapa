"""Atomic, multi-format report bundles for index research runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from datetime import date, datetime
from enum import Enum
import math
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd

from icapa.workspace.identity import secret_safe_canonicalize

from .bundle_constants import (
    _BEARER_CREDENTIAL_VALUE,
    _CONNECTION_ASSIGNMENT_VALUE,
    _CREDENTIAL_ASSIGNMENT_VALUE,
    _CREDENTIAL_URI_VALUE,
    _DATABASE_URI_VALUE,
    _FORMULA_PREFIXES,
    _KEY_COLUMN_NAMES,
    _PAYLOAD_FIELD_BY_SHEET,
    _PRIVATE_KEY_VALUE,
    _SQL_STATEMENT_VALUE,
)
from ..contracts import ReportBundleError, ReportPayload, _is_sensitive_key_name

def _workspace_reports_path(workspace: object) -> Path:
    raw = getattr(workspace, "reports_path", None)
    if raw is None:
        raise TypeError("workspace must expose reports_path")
    path = Path(os.fspath(raw)).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _workspace_name(workspace: object) -> str:
    value = getattr(workspace, "workspace_name", None)
    if value is None:
        value = getattr(workspace, "name", None)
    if not isinstance(value, str) or not value:
        raise TypeError("workspace must expose a name")
    return value


def _manifest_timestamp(manifest: Mapping[str, Any] | object | None) -> Any:
    for name in ("completed_at", "created_at"):
        value = _manifest_value(manifest, name)
        if value is not None:
            return value
    return None


def _manifest_value(manifest: Mapping[str, Any] | object | None, name: str) -> Any:
    if manifest is None:
        return None
    if isinstance(manifest, Mapping):
        return manifest.get(name)
    return getattr(manifest, name, None)


def _is_sensitive_identity_token(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"redacted", "identity_digest"}
        and value.get("redacted") is True
    )


def _sanitize(value: Any, *, key: str = "") -> Any:
    if _is_sensitive_key(key):
        return "[REDACTED]"
    if _is_sensitive_identity_token(value):
        return "[REDACTED]"
    if isinstance(value, str):
        return "[REDACTED]" if _is_sensitive_string_value(value) else value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return "[REDACTED]"
    if isinstance(value, Enum):
        return _sanitize(value.value, key=key)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if is_dataclass(value):
        return _sanitize(asdict(value), key=key)
    if isinstance(value, Mapping):
        return {
            str(name): _sanitize(item, key=str(name))
            for name, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_sanitize(item, key=key) for item in value]
    if hasattr(value, "item"):
        try:
            return _sanitize(value.item(), key=key)
        except (TypeError, ValueError):
            pass
    text = str(value)
    return "[REDACTED]" if _is_sensitive_string_value(text) else text


def _sanitize_report_payload(payload: ReportPayload) -> ReportPayload:
    """Apply the output sanitizer to every fixed-contract report table."""

    replacements = {
        field_name: _sanitize_report_table(getattr(payload, field_name))
        for field_name in _PAYLOAD_FIELD_BY_SHEET.values()
    }
    safe_index_id = _sanitize_output_value(payload.index_id)
    return replace(
        payload,
        index_id=str(safe_index_id),
        **replacements,
    )


def _sanitize_report_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a table whose keyed sensitive values cannot reach a report.

    Analytics plugins are extension points and therefore cannot be assumed to
    follow the manifest's safe-configuration contract. Sensitive column names,
    index names, and key/value rows are redacted before the table is identified
    or sent to Excel or Parquet. Object cells are recursively sanitized so a
    mapping such as ``{"token": "..."}`` is safe even in a generic column.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("report table must be a pandas DataFrame")
    result = frame.copy(deep=True)

    key_column_positions = [
        position
        for position, column in enumerate(result.columns)
        if _is_key_column_name(column)
    ]
    sensitive_rows = pd.Series(False, index=result.index, dtype=bool)
    for position in key_column_positions:
        sensitive_rows |= (
            result.iloc[:, position].map(_is_sensitive_key).fillna(False)
        )

    if isinstance(result.index, pd.MultiIndex):
        index_names = list(result.index.names)
        arrays = [
            result.index.get_level_values(position).tolist()
            for position in range(result.index.nlevels)
        ]
        for position, index_name in enumerate(index_names):
            if _is_sensitive_key(index_name):
                arrays[position] = ["[REDACTED]"] * len(result)
                index_names[position] = _redacted_label("index", position)
            elif _is_key_column_name(index_name):
                for row, value in enumerate(arrays[position]):
                    if _is_sensitive_key(value):
                        sensitive_rows.iloc[row] = True
                        arrays[position][row] = "[REDACTED]"
        result.index = pd.MultiIndex.from_arrays(
            arrays,
            names=index_names,
        )
    else:
        index_name = result.index.name
        if _is_sensitive_key(index_name):
            result.index = pd.Index(
                ["[REDACTED]"] * len(result),
                name=_redacted_label("index", 0),
            )
        elif _is_key_column_name(index_name):
            index_values = result.index.tolist()
            for row, value in enumerate(index_values):
                if _is_sensitive_key(value):
                    sensitive_rows.iloc[row] = True
                    index_values[row] = "[REDACTED]"
            result.index = pd.Index(index_values, name=index_name)

    if isinstance(result.index, pd.MultiIndex):
        result.index = pd.MultiIndex.from_arrays(
            [
                [
                    _sanitize_output_value(value)
                    if isinstance(value, str)
                    else value
                    for value in result.index.get_level_values(position)
                ]
                for position in range(result.index.nlevels)
            ],
            names=result.index.names,
        )
    elif any(isinstance(value, str) for value in result.index):
        result.index = pd.Index(
            [
                _sanitize_output_value(value)
                if isinstance(value, str)
                else value
                for value in result.index
            ],
            name=result.index.name,
        )

    renamed_columns: list[Any] = []
    occupied = {str(column) for column in result.columns}
    for position, column in enumerate(result.columns):
        if not (
            _is_sensitive_key(column)
            or (
                isinstance(column, str)
                and _is_sensitive_string_value(column)
            )
        ):
            renamed_columns.append(column)
            continue
        replacement = _redacted_label("column", position)
        sequence = 2
        while replacement in occupied:
            replacement = f"{_redacted_label('column', position)}_{sequence}"
            sequence += 1
        occupied.add(replacement)
        renamed_columns.append(replacement)
        result.iloc[:, position] = "[REDACTED]"
    result.columns = renamed_columns

    if bool(sensitive_rows.any()):
        result.iloc[sensitive_rows.to_numpy(dtype=bool), :] = "[REDACTED]"

    for position, column in enumerate(result.columns):
        if not (
            pd.api.types.is_object_dtype(result.dtypes.iloc[position])
            or isinstance(result.dtypes.iloc[position], pd.StringDtype)
            or isinstance(result.dtypes.iloc[position], pd.CategoricalDtype)
        ):
            continue
        result.iloc[:, position] = result.iloc[:, position].map(
            lambda value, key=str(column): _sanitize_output_value(
                value,
                key=key,
            )
        )
    return result


def _is_sensitive_key(value: Any) -> bool:
    return _is_sensitive_key_name(value)


def _is_key_column_name(value: Any) -> bool:
    folded = re.sub(r"[^a-z]", "", "" if value is None else str(value).casefold())
    return folded in _KEY_COLUMN_NAMES


def _is_sensitive_string_value(value: str) -> bool:
    """Whether free text contains credentials, a connection, or executable SQL."""

    if not isinstance(value, str) or not value.strip():
        return False
    if isinstance(secret_safe_canonicalize(value), Mapping):
        return True
    return any(
        pattern.search(value) is not None
        for pattern in (
            _DATABASE_URI_VALUE,
            _CREDENTIAL_URI_VALUE,
            _CREDENTIAL_ASSIGNMENT_VALUE,
            _CONNECTION_ASSIGNMENT_VALUE,
            _BEARER_CREDENTIAL_VALUE,
            _PRIVATE_KEY_VALUE,
            _SQL_STATEMENT_VALUE,
        )
    )


def _redacted_label(kind: str, position: int) -> str:
    return f"redacted_sensitive_{kind}_{position + 1}"


def _sanitize_output_value(value: Any, *, key: str = "") -> Any:
    """Sanitize a serialized value and hide sensitive mapping keys."""

    if _is_sensitive_key(key):
        return "[REDACTED]"
    if _is_sensitive_identity_token(value):
        return "[REDACTED]"
    if isinstance(value, str):
        return "[REDACTED]" if _is_sensitive_string_value(value) else value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return "[REDACTED]"
    if isinstance(value, Enum):
        return _sanitize_output_value(value.value, key=key)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if is_dataclass(value):
        return _sanitize_output_value(asdict(value), key=key)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        items = sorted(value.items(), key=lambda pair: str(pair[0]))
        for position, (raw_name, item) in enumerate(items):
            source_name = str(raw_name)
            output_name = (
                _redacted_label("field", position)
                if _is_sensitive_key(source_name)
                else source_name
            )
            sequence = 2
            candidate = output_name
            while candidate in result:
                candidate = f"{output_name}_{sequence}"
                sequence += 1
            result[candidate] = _sanitize_output_value(
                item,
                key=source_name,
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _sanitize_output_value(item, key=key)
            for item in value
        ]
    if hasattr(value, "item"):
        try:
            return _sanitize_output_value(value.item(), key=key)
        except (TypeError, ValueError):
            pass
    text = str(value)
    return "[REDACTED]" if _is_sensitive_string_value(text) else text


def _flatten_mapping(
    value: Any,
    prefix: str = "",
) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[dict[str, Any]] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            name = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_mapping(item, name))
        return rows
    if isinstance(value, list):
        rows = []
        for offset, item in enumerate(value):
            name = f"{prefix}[{offset}]"
            rows.extend(_flatten_mapping(item, name))
        return rows
    return [{"field": prefix or "value", "value": _sanitize(value, key=prefix)}]


def _safe_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        converted = pd.Timestamp(value)
        if converted.tzinfo is not None:
            converted = converted.tz_convert("UTC").tz_localize(None)
        return converted.to_pydatetime()
    if isinstance(value, date):
        return value
    if isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if hasattr(value, "item"):
        try:
            return _safe_cell(value.item())
        except (TypeError, ValueError):
            pass
    text = str(value)
    if _is_sensitive_string_value(text):
        return "[REDACTED]"
    if len(text) > 32_767:
        raise ReportBundleError("Excel cell text exceeds 32,767 characters")
    if text.startswith(_FORMULA_PREFIXES):
        return f"'{text}"
    return text


def _safe_file_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]", "_", value).strip("._")
    if not token:
        token = "table"
    return token[:120]



__all__ = [name for name in globals() if name.startswith("_")]
