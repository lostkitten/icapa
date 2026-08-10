"""Content-addressed Parquet artifacts with v1 JSON read compatibility."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from .identity import dataframe_content_digest
from .manifests import ArtifactRef


PARQUET_ARTIFACT_SCHEMA_VERSION = 2
_ARTIFACT_TYPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PANDAS_ROUNDTRIP_METADATA_KEY = b"icapa.pandas_roundtrip"
_PANDAS_ROUNDTRIP_SCHEMA_VERSION = 1
_PANDAS_TEMPORAL_UNITS = {"s", "ms", "us", "ns"}


class ArtifactError(RuntimeError):
    """Base error for immutable workspace artifacts."""


class ParquetDependencyError(ArtifactError):
    """Raised when the optional PyArrow runtime is unavailable."""


class ArtifactSchemaError(ArtifactError):
    """Raised when a DataFrame cannot be represented without lossy coercion."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when file bytes or logical content do not match their reference."""


def physical_artifact_identity(
    artifact: ArtifactRef,
) -> tuple[str, str, str, str, int, int]:
    """Return immutable object identity independently of its logical role."""

    return (
        artifact.content_digest,
        artifact.file_checksum,
        artifact.relative_path,
        artifact.format,
        artifact.schema_version,
        artifact.size_bytes,
    )


class ParquetArtifactStore:
    """Store immutable DataFrames beneath one fixed research workspace."""

    def __init__(self, workspace_path: str | Path) -> None:
        self.workspace_path = Path(workspace_path).expanduser().resolve()
        self.objects_path = self.workspace_path.joinpath("objects", "sha256")

    def save_frame(
        self,
        artifact_type: str,
        frame: pd.DataFrame,
        *,
        sort_by: Sequence[str] | None = None,
    ) -> ArtifactRef:
        """Write one immutable Parquet object and return its automatic identity."""

        artifact_type = _validate_artifact_type(artifact_type)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        pa, pq = _load_pyarrow()
        working = _sorted_frame(frame, sort_by)
        try:
            table = pa.Table.from_pandas(
                working,
                preserve_index=True,
                safe=True,
            ).combine_chunks()
        except (TypeError, ValueError, pa.ArrowException) as exc:
            raise ArtifactSchemaError(
                f"{artifact_type} cannot be represented as a canonical Arrow table"
            ) from exc
        table = _attach_pandas_roundtrip_metadata(table, working)
        content_digest = _arrow_content_digest(table, pa)
        temporary_directory = self.workspace_path.joinpath("objects", ".tmp")
        temporary_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = temporary_directory.joinpath(f"{uuid4().hex}.parquet.tmp")
        try:
            pq.write_table(
                table,
                temporary_path,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
                version="2.6",
            )
            with temporary_path.open("rb") as stream:
                os.fsync(stream.fileno())
            raw_checksum = _file_checksum(temporary_path)
            destination = self.objects_path.joinpath(
                content_digest[:2],
                content_digest,
                f"{raw_checksum}.parquet",
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if _file_checksum(destination) != raw_checksum:
                    raise ArtifactIntegrityError(
                        f"existing content-addressed artifact is corrupt: {destination}"
                    )
                temporary_path.unlink()
            else:
                os.replace(temporary_path, destination)
            size_bytes = destination.stat().st_size
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return ArtifactRef(
            artifact_type=artifact_type,
            content_digest=content_digest,
            file_checksum=raw_checksum,
            relative_path=str(destination.relative_to(self.workspace_path)),
            format="parquet",
            schema_version=PARQUET_ARTIFACT_SCHEMA_VERSION,
            size_bytes=size_bytes,
        )

    def load_frame(self, reference: ArtifactRef) -> pd.DataFrame:
        """Read and verify a Parquet or v1 JSON frame artifact."""

        path = self._artifact_path(reference.relative_path)
        if reference.format in {"legacy_json", "json", "frame.json"}:
            return self._load_legacy_json(path, reference)
        if reference.format != "parquet":
            raise ArtifactSchemaError(
                f"unsupported frame artifact format: {reference.format!r}"
            )
        pa, pq = _load_pyarrow()
        observed_file_checksum = _file_checksum(path)
        if observed_file_checksum != reference.file_checksum:
            raise ArtifactIntegrityError(
                f"artifact file checksum does not match its reference: {path}"
            )
        try:
            table = pq.read_table(path).combine_chunks()
        except (OSError, pa.ArrowException) as exc:
            raise ArtifactIntegrityError(f"cannot read Parquet artifact: {path}") from exc
        observed_content_digest = _arrow_content_digest(table, pa)
        if observed_content_digest != reference.content_digest:
            raise ArtifactIntegrityError(
                f"artifact logical content does not match its reference: {path}"
            )
        try:
            return _table_to_pandas(table)
        except (ArtifactSchemaError, TypeError, ValueError, pa.ArrowException) as exc:
            raise ArtifactSchemaError(
                f"cannot restore pandas metadata from artifact: {path}"
            ) from exc

    def _load_legacy_json(
        self,
        path: Path,
        reference: ArtifactRef,
    ) -> pd.DataFrame:
        """Read the existing checksummed JSON frame format without rewriting it."""

        from .readers.json_v1 import _decode_envelope, _decode_frame

        raw = path.read_bytes()
        if sha256(raw).hexdigest() != reference.file_checksum:
            raise ArtifactIntegrityError(
                f"v1 artifact file checksum does not match its reference: {path}"
            )
        payload, payload_checksum = _decode_envelope(raw, path)
        if payload.get("kind") == "frame":
            encoded = payload.get("frame")
        else:
            encoded = payload.get("constituents")
        if encoded is None:
            raise ArtifactSchemaError(f"v1 artifact is not a frame: {path}")
        if (
            reference.content_digest
            and reference.content_digest not in {payload_checksum, reference.file_checksum}
        ):
            raise ArtifactIntegrityError(
                f"v1 artifact content digest does not match its reference: {path}"
            )
        return _decode_frame(encoded)

    def _artifact_path(self, relative_path: str) -> Path:
        if not isinstance(relative_path, str) or not relative_path:
            raise ArtifactIntegrityError("artifact relative_path must not be empty")
        candidate = self.workspace_path.joinpath(relative_path).resolve()
        if not candidate.is_relative_to(self.workspace_path):
            raise ArtifactIntegrityError("artifact path escapes its workspace")
        if not candidate.is_file():
            raise ArtifactIntegrityError(f"artifact file does not exist: {candidate}")
        return candidate


def _load_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ParquetDependencyError(
            "Parquet artifacts require the 'pyarrow' package"
        ) from exc
    return pa, pq


def _arrow_content_digest(table, pa) -> str:
    """Hash logical pandas content independently from Parquet encoding details.

    Parquet normalizes some Arrow child-field labels (for example ``item`` to
    ``element`` in list types).  Those physical schema labels must not change
    the identity of an otherwise equal research frame.
    """

    try:
        frame = _table_to_pandas(table)
    except Exception as exc:
        raise ArtifactSchemaError("cannot encode canonical Arrow content") from exc
    return dataframe_content_digest(frame)


def _attach_pandas_roundtrip_metadata(table, frame: pd.DataFrame):
    metadata = dict(table.schema.metadata or {})
    metadata[_PANDAS_ROUNDTRIP_METADATA_KEY] = json.dumps(
        _pandas_roundtrip_metadata(frame),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return table.replace_schema_metadata(metadata)


def _pandas_roundtrip_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": _PANDAS_ROUNDTRIP_SCHEMA_VERSION,
        "columns": [
            _temporal_dtype_descriptor(series.dtype)
            for _, series in frame.items()
        ],
        "index": [
            _temporal_dtype_descriptor(
                frame.index.get_level_values(position).dtype
            )
            for position in range(frame.index.nlevels)
        ],
        "index_frequency": _index_frequency(frame.index),
    }


def _table_to_pandas(table) -> pd.DataFrame:
    """Restore logical pandas temporal metadata before hashing or returning."""

    combined = table.combine_chunks()
    frame = combined.to_pandas()
    metadata = _read_pandas_roundtrip_metadata(combined, frame)
    return _restore_pandas_roundtrip_metadata(frame, metadata)


def _read_pandas_roundtrip_metadata(
    table,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    schema_metadata = table.schema.metadata or {}
    encoded = schema_metadata.get(_PANDAS_ROUNDTRIP_METADATA_KEY)
    if encoded is not None:
        try:
            decoded = json.loads(encoded.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactSchemaError(
                "artifact pandas round-trip metadata is invalid"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise ArtifactSchemaError(
                "artifact pandas round-trip metadata must be an object"
            )
        return dict(decoded)
    return _legacy_pandas_roundtrip_metadata(table, frame)


def _legacy_pandas_roundtrip_metadata(
    table,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Recover temporal units from Arrow's pandas metadata for older objects."""

    encoded = (table.schema.metadata or {}).get(b"pandas")
    if encoded is None:
        return {
            "schema_version": _PANDAS_ROUNDTRIP_SCHEMA_VERSION,
            "columns": [None] * len(frame.columns),
            "index": [None] * frame.index.nlevels,
            "index_frequency": None,
        }
    try:
        pandas_metadata = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactSchemaError("artifact pandas metadata is invalid") from exc
    if not isinstance(pandas_metadata, Mapping):
        raise ArtifactSchemaError("artifact pandas metadata must be an object")
    index_fields = {
        value
        for value in pandas_metadata.get("index_columns", [])
        if isinstance(value, str)
    }
    column_entries: list[Mapping[str, Any]] = []
    index_entries: list[Mapping[str, Any]] = []
    raw_entries = pandas_metadata.get("columns", [])
    if not isinstance(raw_entries, list):
        raise ArtifactSchemaError("artifact pandas columns metadata is invalid")
    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            raise ArtifactSchemaError(
                "artifact pandas column metadata is invalid"
            )
        if entry.get("field_name") in index_fields:
            index_entries.append(entry)
        else:
            column_entries.append(entry)
    if (
        len(column_entries) != len(frame.columns)
        or len(index_entries) != frame.index.nlevels
    ):
        raise ArtifactSchemaError(
            "artifact pandas metadata does not match its table shape"
        )
    return {
        "schema_version": _PANDAS_ROUNDTRIP_SCHEMA_VERSION,
        "columns": [
            _numpy_type_descriptor(entry.get("numpy_type"))
            for entry in column_entries
        ],
        "index": [
            _numpy_type_descriptor(entry.get("numpy_type"))
            for entry in index_entries
        ],
        "index_frequency": None,
    }


def _restore_pandas_roundtrip_metadata(
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    if metadata.get("schema_version") != _PANDAS_ROUNDTRIP_SCHEMA_VERSION:
        raise ArtifactSchemaError(
            "artifact pandas round-trip schema version is unsupported"
        )
    column_metadata = metadata.get("columns")
    index_metadata = metadata.get("index")
    if (
        not isinstance(column_metadata, list)
        or len(column_metadata) != len(frame.columns)
        or not isinstance(index_metadata, list)
        or len(index_metadata) != frame.index.nlevels
    ):
        raise ArtifactSchemaError(
            "artifact pandas round-trip metadata does not match its table shape"
        )
    restored = frame.copy(deep=True)
    for position, descriptor in enumerate(column_metadata):
        if descriptor is not None:
            restored.isetitem(
                position,
                _restore_temporal_values(
                    restored.iloc[:, position],
                    descriptor,
                    path=f"columns[{position}]",
                ),
            )

    index_levels = []
    restore_index = False
    for position, descriptor in enumerate(index_metadata):
        values = restored.index.get_level_values(position)
        if descriptor is not None:
            values = _restore_temporal_values(
                values,
                descriptor,
                path=f"index[{position}]",
            )
            restore_index = True
        index_levels.append(values)
    if restore_index:
        if isinstance(restored.index, pd.MultiIndex):
            restored.index = pd.MultiIndex.from_arrays(
                index_levels,
                names=restored.index.names,
            )
        else:
            restored.index = pd.Index(
                index_levels[0],
                name=restored.index.name,
            )

    frequency = metadata.get("index_frequency")
    if frequency is not None:
        restored.index = _restore_index_frequency(
            restored.index,
            frequency,
        )
    return restored


def _temporal_dtype_descriptor(dtype: object) -> dict[str, str] | None:
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


def _numpy_type_descriptor(value: Any) -> dict[str, str] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(datetime64|timedelta64)\[([A-Za-z]+)\]", value)
    if match is None or match.group(2) not in _PANDAS_TEMPORAL_UNITS:
        return None
    return {
        "kind": "datetime" if match.group(1) == "datetime64" else "timedelta",
        "unit": match.group(2),
    }


def _restore_temporal_values(
    values: pd.Series | pd.Index,
    descriptor: Any,
    *,
    path: str,
) -> pd.Series | pd.Index:
    if not isinstance(descriptor, Mapping):
        raise ArtifactSchemaError(f"{path} temporal metadata must be an object")
    kind = descriptor.get("kind")
    unit = descriptor.get("unit")
    if unit not in _PANDAS_TEMPORAL_UNITS:
        raise ArtifactSchemaError(f"{path} temporal unit is unsupported")
    if kind == "datetime":
        if not pd.api.types.is_datetime64_any_dtype(values.dtype):
            raise ArtifactSchemaError(
                f"{path} temporal metadata does not match stored values"
            )
        if isinstance(values, pd.Series):
            if isinstance(values.dtype, pd.DatetimeTZDtype):
                return pd.Series(
                    pd.DatetimeIndex(values.array).as_unit(str(unit)),
                    index=values.index,
                    name=values.name,
                )
            return values.astype(f"datetime64[{unit}]")
        return pd.DatetimeIndex(values).as_unit(str(unit))
    if kind == "timedelta":
        if not pd.api.types.is_timedelta64_dtype(values.dtype):
            raise ArtifactSchemaError(
                f"{path} temporal metadata does not match stored values"
            )
        if isinstance(values, pd.Series):
            return values.astype(f"timedelta64[{unit}]")
        return pd.TimedeltaIndex(values).as_unit(str(unit))
    raise ArtifactSchemaError(f"{path} temporal kind is unsupported")


def _index_frequency(index: pd.Index) -> str | None:
    if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        return index.freqstr
    return None


def _restore_index_frequency(
    index: pd.Index,
    frequency: Any,
) -> pd.Index:
    if not isinstance(frequency, str) or not frequency:
        raise ArtifactSchemaError("artifact index frequency is invalid")
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
        raise ArtifactSchemaError(
            "artifact index frequency does not match stored values"
        ) from exc
    raise ArtifactSchemaError(
        "artifact index frequency is only valid for temporal indexes"
    )


def _file_checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sorted_frame(
    frame: pd.DataFrame,
    sort_by: Sequence[str] | None,
) -> pd.DataFrame:
    working = frame.copy()
    if not sort_by:
        return working
    has_implicit_range_index = (
        isinstance(working.index, pd.RangeIndex)
        and working.index.name is None
        and working.index.equals(pd.RangeIndex(len(working)))
    )
    if has_implicit_range_index:
        missing = set(sort_by).difference(working.columns)
        if missing:
            raise ArtifactSchemaError(
                f"artifact sort columns are missing: {sorted(missing)}"
            )
        return working.sort_values(
            list(sort_by),
            kind="mergesort",
        ).reset_index(drop=True)
    original_names = list(working.index.names)
    safe_names = [
        name if name is not None else f"__index_level_{position}__"
        for position, name in enumerate(original_names)
    ]
    working.index = working.index.set_names(safe_names)
    flattened = working.reset_index()
    missing = set(sort_by).difference(flattened.columns)
    if missing:
        raise ArtifactSchemaError(
            f"artifact sort columns are missing: {sorted(missing)}"
        )
    flattened = flattened.sort_values(
        list(sort_by),
        kind="mergesort",
    )
    restored = flattened.set_index(safe_names, verify_integrity=True)
    restored.index = restored.index.set_names(original_names)
    return restored


def _validate_artifact_type(value: str) -> str:
    if not isinstance(value, str) or not _ARTIFACT_TYPE.fullmatch(value):
        raise ValueError(
            "artifact_type must use only letters, numbers, periods, underscores, "
            "and hyphens"
        )
    return value


__all__ = [
    "ArtifactError",
    "ArtifactIntegrityError",
    "ArtifactSchemaError",
    "PARQUET_ARTIFACT_SCHEMA_VERSION",
    "ParquetArtifactStore",
    "ParquetDependencyError",
]
