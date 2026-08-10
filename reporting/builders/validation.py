"""Safe, provider-neutral data adapter for index research reports."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime
from enum import Enum
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import ReportDataError, _SAFE_LABEL, _SENSITIVE_KEY_PARTS


def _validation_row(
    effective_date: object,
    check: str,
    passed: bool,
    value: Any,
    message: str,
) -> dict[str, Any]:
    return {
        "effective_date": effective_date,
        "check": check,
        "status": "PASS" if passed else "FAIL",
        "value": value,
        "message": message,
    }

def _normalize_date(value: object, label: str) -> pd.Timestamp:
    if value is None:
        raise ReportDataError(f"{label} is required")
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ReportDataError(f"{label} is not a valid date") from exc
    if pd.isna(result):
        raise ReportDataError(f"{label} is not a valid date")
    if result.tzinfo is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result.normalize()


def _with_named_index_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    index_names = [
        name
        for name in result.index.names
        if name is not None and name not in result.columns
    ]
    if index_names:
        result = result.reset_index()
    if result.columns.duplicated().any():
        raise ReportDataError("report input contains duplicate column names")
    for column in result.columns:
        _reject_sensitive_key(str(column))
    return result


def _first_present(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    label: str,
) -> str:
    for candidate in candidates:
        if candidate in frame:
            return candidate
    raise ReportDataError(f"could not identify {label}")


def _require_finite_numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    allow_missing: bool = False,
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        present = values.notna()
        if not allow_missing and not present.all():
            raise ReportDataError(f"{column} contains missing or non-numeric values")
        if present.any() and not np.isfinite(values[present].to_numpy(dtype=float)).all():
            raise ReportDataError(f"{column} contains non-finite values")


def _safe_label_value(value: object, label: str) -> str:
    if value is None:
        raise ReportDataError(f"{label} is required")
    result = str(value).strip()
    if (
        not _SAFE_LABEL.fullmatch(result)
        or ".." in result
        or "//" in result
    ):
        raise ReportDataError(f"unsafe {label}: {result!r}")
    return result


def _safe_parameter_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, Enum):
        return _safe_parameter_value(value.value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value) if isinstance(value, (float, np.floating)) else int(value)
        if isinstance(number, float) and not math.isfinite(number):
            raise ReportDataError("report parameters must be finite")
        return number
    if isinstance(value, str):
        if len(value) > 1_000:
            raise ReportDataError("report parameter strings are limited to 1,000 characters")
        return value
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return ", ".join(str(_safe_parameter_value(item)) for item in value)
    raise ReportDataError(
        f"unsupported report parameter value type: {type(value).__name__}"
    )


def _reject_sensitive_key(value: str) -> None:
    compact = re.sub(r"[^a-z0-9]", "", value.casefold())
    if any(part in compact for part in _SENSITIVE_KEY_PARTS):
        raise ReportDataError(f"sensitive field is not permitted in reports: {value!r}")


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))



__all__ = [name for name in globals() if name.startswith("_")]
