"""DataFrame schema and digest helpers for analytics cache artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..identity import dataframe_content_digest
from ..manifests import ArtifactRef
from .analytics_contracts import AnalyticsWorkspaceCacheIntegrityError


def _analytics_table_digest(frame: pd.DataFrame) -> str:
    """Return a digest stable across a lossless Arrow/Pandas round trip."""

    working = _stable_parquet_frame(frame)
    if (
        not isinstance(working.index, pd.MultiIndex)
        and working.index.name is None
        and len(working.index) == len(working)
        and working.index.equals(pd.Index(range(len(working))))
    ):
        working.index = pd.RangeIndex(len(working))
    return dataframe_content_digest(working)


def _pandas_roundtrip_schema(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [
            _temporal_dtype(value.dtype)
            for _, value in frame.items()
        ],
        "index": [
            _temporal_dtype(
                frame.index.get_level_values(position).dtype
            )
            for position in range(frame.index.nlevels)
        ],
        "range_index": (
            {
                "start": frame.index.start,
                "stop": frame.index.stop,
                "step": frame.index.step,
            }
            if isinstance(frame.index, pd.RangeIndex)
            else None
        ),
    }


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


def _restore_pandas_schema(
    frame: pd.DataFrame,
    schema: object,
    *,
    path: str,
) -> pd.DataFrame:
    if not isinstance(schema, Mapping):
        raise AnalyticsWorkspaceCacheIntegrityError(
            f"{path} must be an object"
        )
    column_schema = schema.get("columns")
    index_schema = schema.get("index")
    if (
        not isinstance(column_schema, list)
        or len(column_schema) != len(frame.columns)
        or not isinstance(index_schema, list)
        or len(index_schema) != frame.index.nlevels
    ):
        raise AnalyticsWorkspaceCacheIntegrityError(
            f"{path} does not match the cached table shape"
        )
    working = frame.copy(deep=True)
    for position, descriptor in enumerate(column_schema):
        if descriptor is None:
            continue
        converted = _restore_temporal_values(
            working.iloc[:, position],
            descriptor,
            path=f"{path}.columns[{position}]",
        )
        working.isetitem(position, converted)

    restored_levels = [
        _restore_temporal_values(
            working.index.get_level_values(position),
            descriptor,
            path=f"{path}.index[{position}]",
        )
        if descriptor is not None
        else working.index.get_level_values(position)
        for position, descriptor in enumerate(index_schema)
    ]
    if isinstance(working.index, pd.MultiIndex):
        working.index = pd.MultiIndex.from_arrays(
            restored_levels,
            names=working.index.names,
        )
    else:
        working.index = pd.Index(
            restored_levels[0],
            name=working.index.name,
        )

    range_schema = schema.get("range_index")
    if range_schema is not None:
        if not isinstance(range_schema, Mapping):
            raise AnalyticsWorkspaceCacheIntegrityError(
                f"{path}.range_index must be an object"
            )
        expected = pd.RangeIndex(
            start=range_schema["start"],
            stop=range_schema["stop"],
            step=range_schema["step"],
            name=working.index.name,
        )
        if not working.index.equals(pd.Index(expected)):
            raise AnalyticsWorkspaceCacheIntegrityError(
                f"{path}.range_index does not match stored row labels"
            )
        working.index = expected
    return working


def _restore_temporal_values(
    values: pd.Series | pd.Index,
    descriptor: object,
    *,
    path: str,
) -> pd.Series | pd.Index:
    if not isinstance(descriptor, Mapping):
        raise AnalyticsWorkspaceCacheIntegrityError(
            f"{path} must be an object"
        )
    kind = descriptor.get("kind")
    unit = descriptor.get("unit")
    if kind not in {"datetime", "timedelta"} or unit not in {
        "s",
        "ms",
        "us",
        "ns",
    }:
        raise AnalyticsWorkspaceCacheIntegrityError(
            f"{path} contains an unsupported temporal dtype"
        )
    if isinstance(values, pd.Series):
        if kind == "datetime":
            if isinstance(values.dtype, pd.DatetimeTZDtype):
                converted = pd.DatetimeIndex(values.array).as_unit(unit)
                return pd.Series(
                    converted,
                    index=values.index,
                    name=values.name,
                )
            return values.astype(f"datetime64[{unit}]")
        return values.astype(f"timedelta64[{unit}]")
    if kind == "datetime":
        if not isinstance(values, pd.DatetimeIndex):
            raise AnalyticsWorkspaceCacheIntegrityError(
                f"{path} is not a DatetimeIndex"
            )
        return values.as_unit(unit)
    if not isinstance(values, pd.TimedeltaIndex):
        raise AnalyticsWorkspaceCacheIntegrityError(
            f"{path} is not a TimedeltaIndex"
        )
    return values.as_unit(unit)


def _stable_parquet_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize datetime resolution that Parquet may widen on read."""

    working = frame.copy(deep=True)
    for column in working.columns:
        values = working[column]
        if isinstance(values.dtype, pd.DatetimeTZDtype):
            converted = pd.DatetimeIndex(values.array).as_unit("ns")
            working[column] = pd.Series(
                converted,
                index=working.index,
                name=values.name,
            )
        elif pd.api.types.is_datetime64_dtype(values.dtype):
            working[column] = values.astype("datetime64[ns]")
        elif pd.api.types.is_timedelta64_dtype(values.dtype):
            working[column] = values.astype("timedelta64[ns]")

    if isinstance(working.index, pd.MultiIndex):
        levels = [
            _stable_datetime_index(
                working.index.get_level_values(position)
            )
            for position in range(working.index.nlevels)
        ]
        working.index = pd.MultiIndex.from_arrays(
            levels,
            names=working.index.names,
        )
    else:
        working.index = _stable_datetime_index(working.index)
    return working


def _stable_datetime_index(index: pd.Index) -> pd.Index:
    if isinstance(index, pd.DatetimeIndex):
        return index.as_unit("ns")
    if isinstance(index, pd.TimedeltaIndex):
        return index.as_unit("ns")
    return index.copy()


def _unique_artifacts(values: list[ArtifactRef]) -> tuple[ArtifactRef, ...]:
    result: list[ArtifactRef] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        key = (value.content_digest, value.file_checksum)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return tuple(result)


__all__ = [
    "analytics_table_digest",
    "pandas_roundtrip_schema",
    "restore_pandas_schema",
    "stable_parquet_frame",
    "unique_artifacts",
]


analytics_table_digest = _analytics_table_digest
pandas_roundtrip_schema = _pandas_roundtrip_schema
restore_pandas_schema = _restore_pandas_schema
stable_parquet_frame = _stable_parquet_frame
unique_artifacts = _unique_artifacts
