"""Pandas table, index, and temporal codecs for review diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .diagnostic_types import ReviewDiagnosticSerializationError


_NUMPY_TEMPORAL_UNITS = {
    "Y",
    "M",
    "W",
    "D",
    "h",
    "m",
    "s",
    "ms",
    "us",
    "ns",
    "ps",
    "fs",
    "as",
}


def _diagnostic_table_frame(
    value: pd.DataFrame | pd.Series,
    *,
    path: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from .diagnostic_codec import _DiagnosticEncoder

    is_series = isinstance(value, pd.Series)
    frame = (
        value.to_frame(name="__icapa_series_value__")
        if is_series
        else value.copy(deep=True)
    )
    schema_encoder = _DiagnosticEncoder()
    column_kind, column_payload = _encode_column_index(
        frame.columns,
        encoder=schema_encoder,
        path=f"{path}.columns",
    )
    index_kind, index_payload = _encode_row_index(
        frame.index,
        encoder=schema_encoder,
        path=f"{path}.index",
    )
    if schema_encoder.tables:
        raise ReviewDiagnosticSerializationError(
            f"{path} contains tabular index or column labels"
        )

    index_count = frame.index.nlevels
    index_columns = [
        f"__icapa_index_{position:04d}__"
        for position in range(index_count)
    ]
    value_columns = [
        f"__icapa_value_{position:04d}__"
        for position in range(len(frame.columns))
    ]
    storage = pd.DataFrame(index=pd.RangeIndex(len(frame)))
    for position, storage_name in enumerate(index_columns):
        storage[storage_name] = pd.Series(
            frame.index.get_level_values(position).array,
            index=storage.index,
        )
    for position, storage_name in enumerate(value_columns):
        source = frame.iloc[:, position]
        storage[storage_name] = pd.Series(
            source.array,
            index=storage.index,
        )
    temporal_dtypes = [
        _temporal_dtype(storage[column].dtype)
        for column in storage.columns
    ]
    storage = _normalize_temporal_columns(storage)
    schema = {
        "index_count": index_count,
        "value_count": len(frame.columns),
        "storage_columns": list(storage.columns),
        "temporal_dtypes": temporal_dtypes,
        "index_kind": index_kind,
        "index": index_payload,
        "column_kind": column_kind,
        "columns": column_payload,
        "series_name": (
            schema_encoder.encode(
                value.name,
                path=f"{path}.name",
                allow_table=False,
            )
            if is_series
            else None
        ),
    }
    return storage, schema


def _restore_diagnostic_table(
    storage: pd.DataFrame,
    schema: Any,
    *,
    value_kind: str,
    decode: Any,
    path: str,
) -> pd.DataFrame | pd.Series:
    if not isinstance(schema, Mapping):
        raise ValueError(f"{path} pandas schema must be a mapping")
    storage_columns = schema.get("storage_columns")
    temporal_dtypes = schema.get("temporal_dtypes")
    if (
        not isinstance(storage_columns, list)
        or not all(isinstance(item, str) for item in storage_columns)
        or list(storage.columns) != storage_columns
        or not isinstance(temporal_dtypes, list)
        or len(temporal_dtypes) != len(storage_columns)
    ):
        raise ValueError(f"{path} stored table schema does not match its commit")
    working = storage.copy(deep=True)
    for position, descriptor in enumerate(temporal_dtypes):
        if descriptor is not None:
            working.isetitem(
                position,
                _restore_temporal_series(
                    working.iloc[:, position],
                    descriptor,
                    path=f"{path}.temporal_dtypes[{position}]",
                ),
            )
    index_count = schema.get("index_count")
    value_count = schema.get("value_count")
    if (
        not isinstance(index_count, int)
        or index_count < 1
        or not isinstance(value_count, int)
        or value_count < 0
        or index_count + value_count != len(storage_columns)
    ):
        raise ValueError(f"{path} stored table dimensions are invalid")
    index_arrays = [
        working.iloc[:, position].array
        for position in range(index_count)
    ]
    restored_index = _restore_row_index(
        index_arrays,
        kind=schema.get("index_kind"),
        payload=schema.get("index"),
        decode=decode,
        path=f"{path}.index",
    )
    values = working.iloc[:, index_count:].copy(deep=True)
    values.index = restored_index
    values.columns = _restore_column_index(
        kind=schema.get("column_kind"),
        payload=schema.get("columns"),
        decode=decode,
        path=f"{path}.columns",
    )
    if value_kind == "dataframe":
        return values
    if value_kind != "series" or value_count != 1:
        raise ValueError(f"{path} stored Series schema is invalid")
    result = values.iloc[:, 0].copy(deep=True)
    result.name = decode(schema.get("series_name"), f"{path}.name")
    return result


def _encode_row_index(
    index: pd.Index,
    *,
    encoder: Any,
    path: str,
) -> tuple[str, dict[str, Any]]:
    names = [
        encoder.encode(
            name,
            path=f"{path}.names[{position}]",
            allow_table=False,
        )
        for position, name in enumerate(index.names)
    ]
    if isinstance(index, pd.RangeIndex):
        return "range", {
            "names": names,
            "start": index.start,
            "stop": index.stop,
            "step": index.step,
        }
    return (
        "multi" if isinstance(index, pd.MultiIndex) else "index",
        {
            "names": names,
            "frequency": _diagnostic_index_frequency(index),
        },
    )


def _restore_row_index(
    arrays: list[Any],
    *,
    kind: Any,
    payload: Any,
    decode: Any,
    path: str,
) -> pd.Index:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} metadata must be a mapping")
    names_payload = payload.get("names")
    if not isinstance(names_payload, list) or len(names_payload) != len(arrays):
        raise ValueError(f"{path} names do not match its levels")
    names = [
        decode(item, f"{path}.names[{position}]")
        for position, item in enumerate(names_payload)
    ]
    if kind == "range":
        if len(arrays) != 1:
            raise ValueError(f"{path} RangeIndex has multiple levels")
        expected = pd.RangeIndex(
            start=payload.get("start"),
            stop=payload.get("stop"),
            step=payload.get("step"),
            name=names[0],
        )
        if not pd.Index(arrays[0]).equals(pd.Index(expected)):
            raise ValueError(f"{path} RangeIndex values do not match")
        return expected
    if kind == "multi":
        return pd.MultiIndex.from_arrays(arrays, names=names)
    if kind == "index" and len(arrays) == 1:
        return _restore_diagnostic_index_frequency(
            pd.Index(arrays[0], name=names[0]),
            payload.get("frequency"),
            path=path,
        )
    raise ValueError(f"{path} has an unsupported index kind")


def _encode_column_index(
    columns: pd.Index,
    *,
    encoder: Any,
    path: str,
) -> tuple[str, dict[str, Any]]:
    names = [
        encoder.encode(
            name,
            path=f"{path}.names[{position}]",
            allow_table=False,
        )
        for position, name in enumerate(columns.names)
    ]
    labels = [
        encoder.encode(
            label,
            path=f"{path}.labels[{position}]",
            allow_table=False,
        )
        for position, label in enumerate(columns)
    ]
    if isinstance(columns, pd.RangeIndex):
        kind = "range"
    elif isinstance(columns, pd.MultiIndex):
        kind = "multi"
    else:
        kind = "index"
    return kind, {
        "names": names,
        "labels": labels,
        "temporal_dtypes": [
            _temporal_dtype(
                columns.get_level_values(position).dtype
            )
            for position in range(columns.nlevels)
        ],
        "range": (
            {
                "start": columns.start,
                "stop": columns.stop,
                "step": columns.step,
            }
            if isinstance(columns, pd.RangeIndex)
            else None
        ),
        "frequency": _diagnostic_index_frequency(columns),
    }


def _restore_column_index(
    *,
    kind: Any,
    payload: Any,
    decode: Any,
    path: str,
) -> pd.Index:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} metadata must be a mapping")
    names_payload = payload.get("names")
    labels_payload = payload.get("labels")
    if not isinstance(names_payload, list) or not isinstance(
        labels_payload,
        list,
    ):
        raise ValueError(f"{path} metadata is incomplete")
    names = [
        decode(item, f"{path}.names[{position}]")
        for position, item in enumerate(names_payload)
    ]
    labels = [
        decode(item, f"{path}.labels[{position}]")
        for position, item in enumerate(labels_payload)
    ]
    temporal_dtypes = payload.get("temporal_dtypes")
    if temporal_dtypes is None:
        temporal_dtypes = [None] * len(names)
    if (
        not isinstance(temporal_dtypes, list)
        or len(temporal_dtypes) != len(names)
    ):
        raise ValueError(f"{path} temporal metadata does not match its levels")
    if kind == "range":
        range_payload = payload.get("range")
        if not isinstance(range_payload, Mapping):
            raise ValueError(f"{path} RangeIndex metadata is incomplete")
        result = pd.RangeIndex(
            start=range_payload.get("start"),
            stop=range_payload.get("stop"),
            step=range_payload.get("step"),
            name=names[0] if names else None,
        )
        if list(result) != labels:
            raise ValueError(f"{path} RangeIndex labels do not match")
        return result
    if kind == "multi":
        result = pd.MultiIndex.from_tuples(labels, names=names)
        levels = [
            _restore_temporal_index_values(
                result.get_level_values(position),
                descriptor,
                path=f"{path}.temporal_dtypes[{position}]",
            )
            if descriptor is not None
            else result.get_level_values(position)
            for position, descriptor in enumerate(temporal_dtypes)
        ]
        return pd.MultiIndex.from_arrays(levels, names=names)
    if kind == "index" and len(names) == 1:
        result = pd.Index(labels, name=names[0])
        if temporal_dtypes[0] is not None:
            result = _restore_temporal_index_values(
                result,
                temporal_dtypes[0],
                path=f"{path}.temporal_dtypes[0]",
            ).rename(names[0])
        return _restore_diagnostic_index_frequency(
            result,
            payload.get("frequency"),
            path=path,
        )
    raise ValueError(f"{path} has an unsupported column index kind")


def _diagnostic_index_frequency(index: pd.Index) -> str | None:
    if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        return index.freqstr
    return None


def _restore_temporal_index_values(
    values: pd.Index,
    descriptor: Any,
    *,
    path: str,
) -> pd.Index:
    restored = _restore_temporal_series(
        pd.Series(values.array),
        descriptor,
        path=path,
    )
    return pd.Index(restored.array, name=values.name)


def _restore_diagnostic_index_frequency(
    index: pd.Index,
    frequency: Any,
    *,
    path: str,
) -> pd.Index:
    if frequency is None:
        return index
    if not isinstance(frequency, str) or not frequency:
        raise ValueError(f"{path} frequency is invalid")
    try:
        if isinstance(index, pd.DatetimeIndex):
            return pd.DatetimeIndex(
                index.array,
                freq=frequency,
                name=index.name,
            )
        if isinstance(index, pd.TimedeltaIndex):
            return pd.TimedeltaIndex(
                index.array,
                freq=frequency,
                name=index.name,
            )
    except ValueError as exc:
        raise ValueError(
            f"{path} frequency does not match its labels"
        ) from exc
    raise ValueError(
        f"{path} frequency is only valid for a temporal index"
    )


def _temporal_dtype(dtype: object) -> dict[str, str] | None:
    if isinstance(dtype, pd.DatetimeTZDtype):
        return {"kind": "datetime", "unit": dtype.unit}
    if pd.api.types.is_datetime64_dtype(dtype):
        return {
            "kind": "datetime",
            "unit": np.datetime_data(np.dtype(dtype))[0],
        }
    if pd.api.types.is_timedelta64_dtype(dtype):
        return {
            "kind": "timedelta",
            "unit": np.datetime_data(np.dtype(dtype))[0],
        }
    return None


def _normalize_temporal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    for position in range(len(result.columns)):
        values = result.iloc[:, position]
        if isinstance(values.dtype, pd.DatetimeTZDtype):
            result.isetitem(
                position,
                pd.Series(
                    pd.DatetimeIndex(values.array).as_unit("ns"),
                    index=result.index,
                    name=values.name,
                ),
            )
        elif pd.api.types.is_datetime64_dtype(values.dtype):
            result.isetitem(position, values.astype("datetime64[ns]"))
        elif pd.api.types.is_timedelta64_dtype(values.dtype):
            result.isetitem(position, values.astype("timedelta64[ns]"))
    return result


def _restore_temporal_series(
    values: pd.Series,
    descriptor: Any,
    *,
    path: str,
) -> pd.Series:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{path} must be a mapping")
    kind = descriptor.get("kind")
    unit = _validated_temporal_unit(descriptor.get("unit"), path=path)
    if kind == "datetime":
        if isinstance(values.dtype, pd.DatetimeTZDtype):
            return pd.Series(
                pd.DatetimeIndex(values.array).as_unit(unit),
                index=values.index,
                name=values.name,
            )
        return values.astype(f"datetime64[{unit}]")
    if kind == "timedelta":
        return values.astype(f"timedelta64[{unit}]")
    raise ValueError(f"{path} has an unsupported temporal kind")


def _validated_temporal_unit(value: Any, *, path: str) -> str:
    if value not in {"s", "ms", "us", "ns"}:
        raise ValueError(f"{path} contains an unsupported temporal unit")
    return str(value)


def _validated_numpy_temporal_unit(value: Any, *, path: str) -> str:
    if value not in _NUMPY_TEMPORAL_UNITS:
        raise ValueError(f"{path} contains an unsupported NumPy temporal unit")
    return str(value)




__all__ = [
    "_diagnostic_table_frame",
    "_restore_diagnostic_table",
    "_validated_numpy_temporal_unit",
    "_validated_temporal_unit",
]
