"""Validation and numeric helpers for core analytics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..contracts import AnalyticsValidationError

def normalize_date(value: object, label: str) -> pd.Timestamp:
    try:
        date = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalyticsValidationError(f"{label} is invalid: {value!r}") from error
    if pd.isna(date):
        raise AnalyticsValidationError(f"{label} must not be missing")
    return date.normalize()

def use_instrument_index(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if isinstance(frame.index, pd.MultiIndex):
        raise AnalyticsValidationError(
            f"{label} must have a one-dimensional instrument_id index"
        )
    if frame.index.name == "instrument_id":
        if "instrument_id" in frame.columns:
            raise AnalyticsValidationError(
                f"{label} contains instrument_id as both index and column"
            )
        result = frame.copy(deep=True)
    elif "instrument_id" in frame.columns:
        result = frame.set_index("instrument_id", verify_integrity=True)
    else:
        raise AnalyticsValidationError(
            f"{label} must use instrument_id as its index or a column"
        )
    if result.index.hasnans:
        raise AnalyticsValidationError(f"{label} contains missing instrument_id")
    if result.index.has_duplicates:
        raise AnalyticsValidationError(
            f"{label} contains duplicate instrument_id values"
        )
    result.index.name = "instrument_id"
    return result

def validated_weight_series(series: pd.Series, label: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    if converted.isna().any() or not np.isfinite(
        converted.to_numpy(dtype=float)
    ).all():
        raise AnalyticsValidationError(f"{label} contains missing or non-finite values")
    if (converted < 0).any():
        raise AnalyticsValidationError(f"{label} contains negative values")
    return converted.astype(float)

def normalize_result_weights(weights: pd.DataFrame) -> pd.Series:
    frame = weights.copy(deep=True)
    if isinstance(frame.index, pd.MultiIndex):
        if set(frame.index.names) != {"effective_date", "instrument_id"}:
            raise AnalyticsValidationError(
                "backtest_result.weights index must be effective_date/instrument_id"
            )
        frame = frame.reset_index()
    required = {"effective_date", "instrument_id", "index_weight"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AnalyticsValidationError(
            f"backtest_result.weights is missing columns: {missing}"
        )
    if frame.empty:
        raise AnalyticsValidationError(
            "backtest_result.weights must contain at least one row"
        )
    dates = pd.to_datetime(frame["effective_date"], errors="coerce")
    if dates.isna().any():
        raise AnalyticsValidationError(
            "backtest_result.weights contains invalid effective_date values"
        )
    frame["effective_date"] = dates.map(
        lambda value: pd.Timestamp(value).normalize()
    )
    if frame["instrument_id"].isna().any():
        raise AnalyticsValidationError(
            "backtest_result.weights contains missing instrument_id"
        )
    if frame.duplicated(["effective_date", "instrument_id"]).any():
        raise AnalyticsValidationError(
            "backtest_result.weights contains duplicate date/instrument rows"
        )
    frame["index_weight"] = validated_weight_series(
        frame["index_weight"], "backtest_result.weights.index_weight"
    )
    return frame.set_index(
        ["effective_date", "instrument_id"], verify_integrity=True
    )["index_weight"]

def use_business_date_index(
    frame: pd.DataFrame, source_name: str
) -> pd.DataFrame:
    result = frame.copy(deep=True)
    if "business_date" in result.columns:
        dates = pd.to_datetime(result["business_date"], errors="coerce")
        if dates.isna().any():
            raise AnalyticsValidationError(
                f"{source_name}.business_date contains invalid dates"
            )
        result = result.drop(columns="business_date")
        result.index = pd.DatetimeIndex(dates, name="business_date")
    elif result.index.name == "business_date" or isinstance(
        result.index, pd.DatetimeIndex
    ):
        dates = pd.to_datetime(result.index, errors="coerce")
        if dates.isna().any():
            raise AnalyticsValidationError(
                f"{source_name} index contains invalid business_date values"
            )
        result.index = pd.DatetimeIndex(dates, name="business_date")
    else:
        raise AnalyticsValidationError(
            f"{source_name} must use business_date as its index or a column"
        )
    if result.index.has_duplicates:
        raise AnalyticsValidationError(
            f"{source_name} contains duplicate business_date rows"
        )
    return result.sort_index()

def annualized_return(
    total_return: float, observations: int, annualization_factor: int
) -> float:
    if observations <= 0:
        return np.nan
    if total_return <= -1.0:
        return -1.0
    return float(
        (1.0 + total_return) ** (annualization_factor / observations) - 1.0
    )

def annualized_volatility(
    returns: np.ndarray, annualization_factor: int
) -> float:
    if len(returns) < 2:
        return np.nan
    return float(np.std(returns, ddof=1) * np.sqrt(annualization_factor))

def level_and_drawdown(
    returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    levels = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], levels)))[1:]
    drawdowns = levels / peaks - 1.0
    return levels, drawdowns

def empty_performance() -> tuple[pd.Series, pd.DataFrame]:
    performance = pd.Series(dtype=float, name="value")
    drawdowns = pd.DataFrame(
        columns=[
            "index_level",
            "benchmark_level",
            "index_drawdown",
            "benchmark_drawdown",
            "active_return",
        ],
        index=pd.DatetimeIndex([], name="business_date"),
    )
    return performance, drawdowns

__all__ = [
    "normalize_date",
    "use_instrument_index",
    "validated_weight_series",
    "normalize_result_weights",
    "use_business_date_index",
    "annualized_return",
    "annualized_volatility",
    "level_and_drawdown",
    "empty_performance",
]
