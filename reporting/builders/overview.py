"""Safe, provider-neutral data adapter for index research reports."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..contracts import (
    OVERVIEW_COLUMNS,
    REPORT_CONTRACT,
    ReportDataError,
    _SAFE_FIELD,
)
from .validation import (
    _reject_sensitive_key,
    _safe_label_value,
    _safe_parameter_value,
)

def _build_overview(
    *,
    index_id: str,
    index_name: str | None,
    methodology_name: str | None,
    workspace_name: str | None,
    generated_at: object | None,
    schedule: pd.DataFrame,
    latest_holdings: pd.DataFrame,
    analytics: object | None,
) -> pd.DataFrame:
    timestamp = (
        pd.Timestamp.now(tz="UTC")
        if generated_at is None
        else pd.Timestamp(generated_at)
    )
    rows: list[dict[str, Any]] = [
        {"field": "report_contract", "value": REPORT_CONTRACT},
        {"field": "index_id", "value": index_id},
        {"field": "generated_at", "value": timestamp},
        {"field": "review_count", "value": int(len(schedule))},
        {
            "field": "first_reference_date",
            "value": schedule["reference_date"].min(),
        },
        {
            "field": "first_effective_date",
            "value": schedule["effective_date"].min(),
        },
        {
            "field": "latest_effective_date",
            "value": schedule["effective_date"].max(),
        },
        {
            "field": "latest_constituent_count",
            "value": int(len(latest_holdings)),
        },
    ]
    for field_name, value in (
        ("index_name", index_name),
        ("methodology_name", methodology_name),
        ("workspace_name", workspace_name),
    ):
        if value is not None:
            rows.append(
                {
                    "field": field_name,
                    "value": _safe_label_value(value, field_name),
                }
            )
    performance = getattr(analytics, "performance", None) if analytics else None
    if isinstance(performance, pd.Series):
        for name, value in performance.items():
            key = str(name).strip()
            if not _SAFE_FIELD.fullmatch(key):
                raise ReportDataError(f"unsafe analytics performance name: {key!r}")
            _reject_sensitive_key(key)
            rows.append(
                {
                    "field": f"performance.{key}",
                    "value": _safe_parameter_value(value),
                }
            )
    return pd.DataFrame.from_records(rows, columns=OVERVIEW_COLUMNS)


__all__ = ["_build_overview"]
