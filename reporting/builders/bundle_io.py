"""Atomic, multi-format report bundles for index research runs."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from ..contracts import ReportBundleError
from .security import (
    _safe_file_token,
    _sanitize_output_value,
    _sanitize_report_table,
)

def _publish_report_bundle(
    temporary: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    """Publish a completed bundle while preserving any prior bundle on error."""

    backup: Path | None = None
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"report bundle already exists: {destination.name}"
            )
        backup = destination.parent.joinpath(
            f".{destination.name}.replaced-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        )
        os.replace(destination, backup)

    try:
        os.replace(temporary, destination)
    except BaseException:
        if backup is not None:
            if destination.exists():
                shutil.rmtree(destination)
            try:
                os.replace(backup, destination)
            except BaseException as restore_error:
                raise ReportBundleError(
                    "report publication failed and the prior bundle could not "
                    f"be restored; its backup remains at {backup}"
                ) from restore_error
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup)



def _parquet_table_file_names(
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, str]:
    """Return stable filenames without merging normalized table names."""

    table_names = sorted(tables)
    base_tokens = {
        table_name: _safe_file_token(table_name)
        for table_name in table_names
    }
    token_counts: dict[str, int] = {}
    for token in base_tokens.values():
        token_counts[token] = token_counts.get(token, 0) + 1

    result: dict[str, str] = {}
    occupied: set[str] = set()
    for table_name in table_names:
        token = base_tokens[table_name]
        if token_counts[token] == 1:
            candidate = token
        else:
            candidate = _hashed_file_token(token, table_name)
        sequence = 2
        while candidate in occupied:
            candidate = _hashed_file_token(
                token,
                table_name,
                sequence=sequence,
            )
            sequence += 1
        occupied.add(candidate)
        result[table_name] = f"{candidate}.parquet"
    return result


def _hashed_file_token(
    normalized_token: str,
    source_name: str,
    *,
    sequence: int | None = None,
) -> str:
    digest = sha256(source_name.encode("utf-8")).hexdigest()
    suffix = f"--{digest}"
    if sequence is not None:
        suffix = f"{suffix}-{sequence}"
    prefix_length = max(1, 120 - len(suffix))
    prefix = normalized_token[:prefix_length].rstrip("._-") or "table"
    return f"{prefix}{suffix}"


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    serializable = _sanitize_report_table(frame)
    for column in serializable.select_dtypes(include=["object"]).columns:
        serializable[column] = serializable[column].map(_parquet_object_value)
    try:
        serializable.to_parquet(
            path,
            engine="pyarrow",
            compression="zstd",
            index=True,
        )
    except ImportError as error:
        raise ReportBundleError(
            "PyArrow is required for Parquet report tables"
        ) from error
    except Exception as error:
        raise ReportBundleError(
            f"report table cannot be serialized as Parquet: {path.stem}"
        ) from error


def _parquet_object_value(value: Any) -> str | None:
    if value is None:
        return None
    try:
        missing = pd.isna(value)
        if getattr(missing, "ndim", 0) == 0 and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    sanitized = _sanitize_output_value(value)
    if sanitized is None:
        return None
    if isinstance(sanitized, str):
        return sanitized
    if isinstance(sanitized, (bool, int, float)):
        return str(sanitized)
    return json.dumps(sanitized, sort_keys=True, ensure_ascii=True)


def _write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        _sanitize_output_value(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def _file_checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [name for name in globals() if name.startswith("_")]
