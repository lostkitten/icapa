"""Safe, provider-neutral data adapter for index research reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import (
    ALL_REVIEW_WEIGHTS_COLUMNS,
    LATEST_HOLDINGS_COLUMNS,
    PERFORMANCE_COLUMNS,
    REVIEW_SCHEDULE_COLUMNS,
    ReportDataError,
)
from .validation import (
    _empty,
    _normalize_date,
    _require_finite_numeric,
    _with_named_index_columns,
)

def _extract_reviews(backtest_result: object) -> dict[pd.Timestamp, object]:
    reviews = getattr(backtest_result, "reviews", None)
    if not isinstance(reviews, Mapping) or not reviews:
        raise ReportDataError("backtest_result.reviews must be a non-empty mapping")
    result: dict[pd.Timestamp, object] = {}
    for raw_date, context in reviews.items():
        effective_date = _normalize_date(raw_date, "review effective date")
        if effective_date in result:
            raise ReportDataError("backtest_result contains duplicate review dates")
        context_date = getattr(context, "effective_date", None)
        if context_date is None:
            raise ReportDataError("each review context must define effective_date")
        if _normalize_date(context_date, "context effective_date") != effective_date:
            raise ReportDataError(
                "review mapping key does not match the context effective_date"
            )
        result[effective_date] = context
    return dict(sorted(result.items()))


def _extract_weights(backtest_result: object) -> pd.DataFrame:
    frame = getattr(backtest_result, "weights", None)
    if not isinstance(frame, pd.DataFrame):
        raise ReportDataError("backtest_result.weights must be a pandas DataFrame")
    frame = _with_named_index_columns(frame)
    required = set(ALL_REVIEW_WEIGHTS_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ReportDataError(f"backtest weights are missing columns: {missing}")
    result = frame.loc[:, ALL_REVIEW_WEIGHTS_COLUMNS].copy()
    result["effective_date"] = pd.to_datetime(
        result["effective_date"], errors="raise"
    ).dt.normalize()
    result["index_weight"] = pd.to_numeric(
        result["index_weight"], errors="raise"
    )
    if result.duplicated(["effective_date", "instrument_id"]).any():
        raise ReportDataError(
            "backtest weights contain duplicate effective_date/instrument_id rows"
        )
    if not np.isfinite(result["index_weight"].to_numpy(dtype=float)).all():
        raise ReportDataError("backtest weights contain non-finite values")
    return result.sort_values(
        ["effective_date", "instrument_id"], kind="stable"
    ).reset_index(drop=True)


def _extract_index_id(reviews: Mapping[pd.Timestamp, object]) -> str:
    values = {
        str(getattr(context, "index_id", "")).strip()
        for context in reviews.values()
    }
    if "" in values or len(values) != 1:
        raise ReportDataError(
            "all review contexts must contain the same non-empty index_id"
        )
    return values.pop()


def _build_review_schedule(
    reviews: Mapping[pd.Timestamp, object],
    index_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for effective_date, context in reviews.items():
        reference_date = _normalize_date(
            getattr(context, "reference_date", None),
            "reference_date",
        )
        if reference_date > effective_date:
            raise ReportDataError("reference_date must not be after effective_date")
        constituents = getattr(context, "cons", None)
        if not isinstance(constituents, pd.DataFrame):
            raise ReportDataError("each review context must expose a constituent frame")
        rows.append(
            {
                "reference_date": reference_date,
                "effective_date": effective_date,
                "index_id": index_id,
                "universe_size": int(len(constituents)),
            }
        )
    return pd.DataFrame.from_records(rows, columns=REVIEW_SCHEDULE_COLUMNS)


def _validate_review_weight_dates(
    weights: pd.DataFrame,
    schedule: pd.DataFrame,
) -> None:
    actual = set(weights["effective_date"])
    expected = set(schedule["effective_date"])
    if actual != expected:
        raise ReportDataError(
            "backtest weights and review contexts contain different effective dates"
        )


def _build_latest_holdings(
    reviews: Mapping[pd.Timestamp, object],
) -> pd.DataFrame:
    effective_date = max(reviews)
    context = reviews[effective_date]
    raw = getattr(context, "cons", None)
    if not isinstance(raw, pd.DataFrame):
        raise ReportDataError("latest review context has no constituent frame")
    frame = _with_named_index_columns(raw)
    if "instrument_id" not in frame:
        raise ReportDataError(
            "latest holdings must use instrument_id as a column or named index"
        )
    if frame["instrument_id"].duplicated().any():
        raise ReportDataError("latest holdings contain duplicate instrument_id values")
    frame["reference_date"] = _normalize_date(
        getattr(context, "reference_date", None),
        "reference_date",
    )
    frame["effective_date"] = effective_date
    result = pd.DataFrame(index=frame.index)
    for column in LATEST_HOLDINGS_COLUMNS:
        result[column] = frame[column] if column in frame else pd.NA
    return result.reset_index(drop=True)


def _build_performance(simulation: object | None) -> pd.DataFrame:
    if simulation is None:
        return _empty(PERFORMANCE_COLUMNS)
    raw = getattr(simulation, "daily", None)
    if not isinstance(raw, pd.DataFrame):
        raise ReportDataError("simulation.daily must be a pandas DataFrame")
    if raw.empty:
        return _empty(PERFORMANCE_COLUMNS)
    frame = _with_named_index_columns(raw)
    if "business_date" not in frame:
        raise ReportDataError(
            "simulation.daily must use business_date as a column or named index"
        )
    core = {
        "index_price_return",
        "index_gross_total_return",
        "index_net_total_return",
        "benchmark_price_return",
        "benchmark_gross_total_return",
        "benchmark_net_total_return",
    }
    missing = sorted(core.difference(frame.columns))
    if missing:
        raise ReportDataError(f"simulation.daily is missing columns: {missing}")
    frame["business_date"] = pd.to_datetime(
        frame["business_date"], errors="raise"
    ).dt.normalize()
    for variant in ("price", "gross_total", "net_total"):
        active = f"active_{variant}_return"
        if active not in frame:
            frame[active] = (
                frame[f"index_{variant}_return"]
                - frame[f"benchmark_{variant}_return"]
            )
    result = pd.DataFrame(index=frame.index)
    for column in PERFORMANCE_COLUMNS:
        result[column] = frame[column] if column in frame else pd.NA
    _require_finite_numeric(
        result,
        [column for column in PERFORMANCE_COLUMNS if column != "business_date"],
        allow_missing=True,
    )
    return result.sort_values("business_date", kind="stable").reset_index(drop=True)



__all__ = [name for name in globals() if name.startswith("_")]
