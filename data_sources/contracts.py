"""Canonical data contracts shared by ICAPA data-source adapters.

Provider-specific names belong in the adapter/SQL boundary.  Portfolio rules
and analytics should only receive the names defined here.
"""

from __future__ import annotations

from collections.abc import Iterable
import math

import pandas as pd


UNIVERSE_COLUMNS = (
    "instrument_id",
    "name",
    "country",
    "industry",
    "shares",
    "free_float",
    "price",
    "currency",
    "base_currency",
    "fx_rate",
    "market_cap",
    "benchmark_weight",
    "reference_date",
    "effective_date",
)

DAILY_MARKET_COLUMNS = (
    "instrument_id",
    "business_date",
    "price_return",
    "gross_dividend",
    "net_dividend",
    "market_cap",
)

REVIEW_SCHEDULE_COLUMNS = (
    "reference_date",
    "effective_date",
)


def _available_columns(df: pd.DataFrame) -> set[str]:
    columns = set(df.columns)
    if df.index.name:
        columns.add(df.index.name)
    if isinstance(df.index, pd.MultiIndex):
        columns.update(name for name in df.index.names if name)
    return columns


def require_columns(df: pd.DataFrame, required: Iterable[str], contract: str) -> None:
    """Raise a readable error when a provider violates a canonical contract."""
    missing = set(required) - _available_columns(df)
    if missing:
        raise ValueError(f"{contract} is missing canonical columns: {sorted(missing)}")


def validate_review_dates(reference_date, effective_date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate the review cutoff/effective relationship and normalize values."""
    reference = pd.Timestamp(reference_date).normalize()
    effective = pd.Timestamp(effective_date).normalize()
    if pd.isna(reference) or pd.isna(effective):
        raise ValueError("reference_date and effective_date must not be null")
    if reference > effective:
        raise ValueError(
            f"reference_date ({reference.date()}) must not be after "
            f"effective_date ({effective.date()})"
        )
    return reference, effective


def validate_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a canonical point-in-time universe and return a safe copy."""
    require_columns(df, UNIVERSE_COLUMNS, "universe")
    result = df.copy()
    if "instrument_id" in result.columns:
        ids = result["instrument_id"]
    else:
        ids = pd.Series(result.index, index=result.index)
    if ids.isna().any():
        raise ValueError("universe contains null instrument_id values")
    if ids.duplicated().any():
        raise ValueError("universe contains duplicate instrument_id values")

    numeric_columns = (
        "shares",
        "free_float",
        "price",
        "fx_rate",
        "market_cap",
        "benchmark_weight",
    )
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not result[column].map(math.isfinite).all():
            raise ValueError(f"universe contains non-finite {column} values")
    if (result[["shares", "price", "market_cap", "benchmark_weight"]] < 0).any().any():
        raise ValueError("universe contains negative size, price, or weight values")
    if ((result["free_float"] < 0) | (result["free_float"] > 1)).any():
        raise ValueError("universe free_float values must be between 0 and 1")
    if (result["fx_rate"] <= 0).any():
        raise ValueError("universe fx_rate values must be positive")
    benchmark_total = float(result["benchmark_weight"].sum())
    if not math.isclose(benchmark_total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("universe benchmark_weight values must sum to 1")

    for column in ("reference_date", "effective_date"):
        result[column] = pd.to_datetime(result[column]).dt.normalize()
        if result[column].isna().any():
            raise ValueError(f"universe contains null {column} values")

    pairs = result[["reference_date", "effective_date"]].drop_duplicates()
    if len(pairs) != 1:
        raise ValueError("universe must contain exactly one review date pair")
    for row in pairs.itertuples(index=False):
        validate_review_dates(row.reference_date, row.effective_date)
    return result


def validate_daily_market_data(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """Validate instrument-by-business-day facts and optional cutoff compliance."""
    require_columns(df, DAILY_MARKET_COLUMNS, "daily market data")
    result = df.copy()
    if result["instrument_id"].isna().any():
        raise ValueError("daily market data contains null instrument_id values")
    result["business_date"] = pd.to_datetime(result["business_date"]).dt.normalize()
    if result["business_date"].isna().any():
        raise ValueError("daily market data contains null business_date values")
    if result.duplicated(["instrument_id", "business_date"]).any():
        raise ValueError(
            "daily market data contains duplicate instrument_id/business_date rows"
        )
    if reference_date is not None and (
        result["business_date"] > pd.Timestamp(reference_date).normalize()
    ).any():
        raise ValueError("daily market data contains business_date after reference_date")
    return result


def validate_review_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a canonical review schedule."""
    require_columns(df, REVIEW_SCHEDULE_COLUMNS, "review schedule")
    result = df.copy()
    if result.empty:
        raise ValueError("review schedule must contain at least one row")
    for column in REVIEW_SCHEDULE_COLUMNS:
        result[column] = pd.to_datetime(result[column]).dt.normalize()
        if result[column].isna().any():
            raise ValueError(f"review schedule contains null {column} values")
    if result["effective_date"].duplicated().any():
        raise ValueError("review schedule contains duplicate effective_date values")
    for row in result.itertuples(index=False):
        validate_review_dates(row.reference_date, row.effective_date)
    return result.sort_values("effective_date").reset_index(drop=True)
