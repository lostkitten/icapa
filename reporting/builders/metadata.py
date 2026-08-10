"""Safe, provider-neutral data adapter for index research reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import (
    DATA_SOURCE_COLUMNS,
    METHODOLOGY_PARAMETER_COLUMNS,
    VALIDATION_COLUMNS,
    ReportDataError,
    _SAFE_FIELD,
)
from .validation import (
    _empty,
    _reject_sensitive_key,
    _safe_label_value,
    _safe_parameter_value,
    _validation_row,
    _with_named_index_columns,
)

def _build_methodology_parameters(
    values: Mapping[str, Any] | None,
) -> pd.DataFrame:
    if values is None:
        return _empty(METHODOLOGY_PARAMETER_COLUMNS)
    if not isinstance(values, Mapping):
        raise ReportDataError("methodology_parameters must be a mapping")
    rows: list[dict[str, Any]] = []
    _flatten_parameters(values, rows=rows)
    return pd.DataFrame.from_records(rows, columns=METHODOLOGY_PARAMETER_COLUMNS)


def _flatten_parameters(
    values: Mapping[str, Any],
    *,
    rows: list[dict[str, Any]],
    prefix: str = "",
) -> None:
    for raw_key, value in values.items():
        key = str(raw_key).strip()
        if not _SAFE_FIELD.fullmatch(key):
            raise ReportDataError(f"unsafe methodology parameter name: {key!r}")
        full_key = f"{prefix}.{key}" if prefix else key
        _reject_sensitive_key(full_key)
        if isinstance(value, Mapping):
            _flatten_parameters(value, rows=rows, prefix=full_key)
        else:
            rows.append(
                {
                    "parameter": full_key,
                    "value": _safe_parameter_value(value),
                }
            )


def _build_data_sources(
    values: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> pd.DataFrame:
    if values is None:
        return _empty(DATA_SOURCE_COLUMNS)
    records: list[Mapping[str, Any]]
    if isinstance(values, Mapping):
        if "capability" in values or "provider_name" in values:
            records = [values]
        else:
            records = []
            for capability, specification in values.items():
                if isinstance(specification, str):
                    records.append(
                        {
                            "capability": capability,
                            "provider_name": specification,
                            "data_type": "",
                            "fields": (),
                        }
                    )
                elif isinstance(specification, Mapping):
                    records.append({**specification, "capability": capability})
                else:
                    raise ReportDataError(
                        "data source mappings must contain labels or mappings"
                    )
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        records = list(values)
    else:
        raise ReportDataError("data_sources must be a mapping or sequence of mappings")

    rows: list[dict[str, Any]] = []
    allowed = set(DATA_SOURCE_COLUMNS)
    for record in records:
        if not isinstance(record, Mapping):
            raise ReportDataError("each data source must be a mapping")
        unknown = set(map(str, record)).difference(allowed)
        if unknown:
            raise ReportDataError(
                f"data source contains unsupported fields: {sorted(unknown)}"
            )
        capability = _safe_label_value(record.get("capability"), "capability")
        provider_name = _safe_label_value(
            record.get("provider_name"), "provider_name"
        )
        data_type_raw = record.get("data_type", "")
        data_type = (
            ""
            if data_type_raw in (None, "")
            else _safe_label_value(data_type_raw, "data_type")
        )
        fields_value = record.get("fields", ())
        if isinstance(fields_value, str):
            field_names = [fields_value]
        elif isinstance(fields_value, Iterable):
            field_names = list(fields_value)
        else:
            raise ReportDataError("data source fields must be a string or iterable")
        clean_fields: list[str] = []
        for raw_field in field_names:
            field_name = str(raw_field).strip()
            if not _SAFE_FIELD.fullmatch(field_name):
                raise ReportDataError(f"unsafe data source field: {field_name!r}")
            _reject_sensitive_key(field_name)
            clean_fields.append(field_name)
        rows.append(
            {
                "capability": capability,
                "provider_name": provider_name,
                "data_type": data_type,
                "fields": ", ".join(clean_fields),
            }
        )
    return pd.DataFrame.from_records(rows, columns=DATA_SOURCE_COLUMNS)


def _build_validation(
    reviews: Mapping[pd.Timestamp, object],
    analytics: object | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for effective_date, context in reviews.items():
        constituents = getattr(context, "cons")
        if "index_weight" not in constituents:
            rows.append(
                _validation_row(
                    effective_date,
                    "index_weight_present",
                    False,
                    "",
                    "The review does not contain index_weight.",
                )
            )
            continue
        weights = pd.to_numeric(constituents["index_weight"], errors="coerce")
        finite = bool(np.isfinite(weights.to_numpy(dtype=float)).all())
        non_negative = bool(finite and (weights >= 0).all())
        total = float(weights.sum()) if finite else math.nan
        sums_to_one = bool(finite and math.isclose(total, 1.0, abs_tol=1e-8))
        rows.extend(
            (
                _validation_row(
                    effective_date,
                    "finite_index_weights",
                    finite,
                    finite,
                    "",
                ),
                _validation_row(
                    effective_date,
                    "non_negative_index_weights",
                    non_negative,
                    float(weights.min()) if finite and len(weights) else "",
                    "",
                ),
                _validation_row(
                    effective_date,
                    "index_weights_sum_to_one",
                    sums_to_one,
                    total,
                    "",
                ),
            )
        )
    if analytics is not None:
        rows.extend(_adapt_analytics_validation(analytics))
    return pd.DataFrame.from_records(rows, columns=VALIDATION_COLUMNS)


def _adapt_analytics_validation(analytics: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw = getattr(analytics, "review_validation", None)
    if raw is not None:
        if not isinstance(raw, pd.DataFrame):
            raise ReportDataError("analytics.review_validation must be a DataFrame")
        if not raw.empty:
            frame = _with_named_index_columns(raw)
            if {"check", "status"}.issubset(frame.columns):
                for record in frame.to_dict(orient="records"):
                    rows.append(
                        {
                            "effective_date": record.get("effective_date", ""),
                            "check": str(record["check"]),
                            "status": str(record["status"]).upper(),
                            "value": _safe_parameter_value(record.get("value", "")),
                            "message": str(record.get("message", "")),
                        }
                    )
            else:
                date_column = next(
                    (
                        column
                        for column in ("effective_date", "reference_date")
                        if column in frame
                    ),
                    None,
                )
                for record in frame.to_dict(orient="records"):
                    for key, value in record.items():
                        if key == date_column:
                            continue
                        _reject_sensitive_key(str(key))
                        passed = bool(value) if isinstance(value, (bool, np.bool_)) else True
                        rows.append(
                            _validation_row(
                                record.get(date_column, ""),
                                f"analytics.{key}",
                                passed,
                                _safe_parameter_value(value),
                                "",
                            )
                        )
    diagnostics = getattr(analytics, "diagnostics", ()) or ()
    for diagnostic in diagnostics:
        level = str(getattr(diagnostic, "level", "info")).lower()
        code = str(getattr(diagnostic, "code", "analytics_diagnostic"))
        message = str(getattr(diagnostic, "message", ""))
        _reject_sensitive_key(code)
        rows.append(
            {
                "effective_date": "",
                "check": code,
                "status": "WARNING" if level == "warning" else "INFO",
                "value": "",
                "message": message,
            }
        )
    return rows



__all__ = [name for name in globals() if name.startswith("_")]
