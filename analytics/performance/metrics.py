"""Standalone performance calculations with explicit return inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Summary statistics and aligned daily diagnostics."""

    summary: pd.Series
    daily: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", self.summary.copy(deep=True))
        object.__setattr__(self, "daily", self.daily.copy(deep=True))


def calculate_performance_metrics(
    index_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    *,
    annualization_factor: int = 252,
) -> PerformanceMetrics:
    """Calculate deterministic performance metrics from simple daily returns."""

    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    index = _validated_returns(index_returns, "index_returns")
    if benchmark_returns is None:
        aligned = index.to_frame("index_return")
    else:
        benchmark = _validated_returns(benchmark_returns, "benchmark_returns")
        aligned = pd.concat(
            [
                index.rename("index_return"),
                benchmark.rename("benchmark_return"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if aligned.empty:
            raise ValueError(
                "index_returns and benchmark_returns have no aligned observations"
            )

    index = aligned["index_return"]
    daily = pd.DataFrame(index=index.index)
    daily["index_return"] = index
    daily["index_level"] = (1.0 + index).cumprod()
    daily["index_drawdown"] = drawdown_series(index)

    summary: dict[str, float] = _return_summary(
        index,
        annualization_factor=annualization_factor,
        prefix="",
    )
    if "benchmark_return" in aligned:
        benchmark = aligned["benchmark_return"]
        active = index - benchmark
        daily["benchmark_return"] = benchmark
        daily["benchmark_level"] = (1.0 + benchmark).cumprod()
        daily["benchmark_drawdown"] = drawdown_series(benchmark)
        daily["active_return"] = active
        summary.update(
            _return_summary(
                benchmark,
                annualization_factor=annualization_factor,
                prefix="benchmark_",
            )
        )
        tracking_error = float(active.std(ddof=1) * np.sqrt(annualization_factor))
        annualized_excess = (
            summary["annualized_return"]
            - summary["benchmark_annualized_return"]
        )
        summary["annualized_excess_return"] = annualized_excess
        summary["tracking_error"] = tracking_error
        summary["information_ratio"] = (
            annualized_excess / tracking_error
            if tracking_error > 0
            else np.nan
        )

    daily.index.name = "business_date"
    return PerformanceMetrics(
        summary=pd.Series(summary, dtype=float, name="value"),
        daily=daily,
    )


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Return the drawdown path of a simple-return series."""

    values = _validated_returns(returns, "returns")
    levels = (1.0 + values).cumprod()
    drawdowns = levels.div(levels.cummax()).sub(1.0)
    drawdowns.name = "drawdown"
    return drawdowns


def calculate_rolling_risk(
    index_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    *,
    windows: tuple[int, ...] = (21, 63, 252),
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """Calculate rolling volatility and optional active-risk statistics."""

    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    selected_windows = tuple(int(window) for window in windows)
    if not selected_windows or any(window <= 1 for window in selected_windows):
        raise ValueError("windows must contain integers greater than one")
    index = _validated_returns(index_returns, "index_returns")
    aligned = index.to_frame("index_return")
    if benchmark_returns is not None:
        benchmark = _validated_returns(benchmark_returns, "benchmark_returns")
        aligned = pd.concat(
            [
                index.rename("index_return"),
                benchmark.rename("benchmark_return"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if aligned.empty:
            raise ValueError(
                "index_returns and benchmark_returns have no aligned observations"
            )

    frames: list[pd.DataFrame] = []
    for window in selected_windows:
        frame = pd.DataFrame(index=aligned.index)
        frame["window"] = window
        frame["annualized_volatility"] = (
            aligned["index_return"].rolling(window).std(ddof=1)
            * np.sqrt(annualization_factor)
        )
        if "benchmark_return" in aligned:
            active = aligned["index_return"] - aligned["benchmark_return"]
            frame["tracking_error"] = (
                active.rolling(window).std(ddof=1)
                * np.sqrt(annualization_factor)
            )
            frame["correlation"] = aligned["index_return"].rolling(window).corr(
                aligned["benchmark_return"]
            )
        frames.append(frame.dropna(how="all", subset=["annualized_volatility"]))
    result = pd.concat(frames).reset_index(names="business_date")
    return result.sort_values(["window", "business_date"]).reset_index(drop=True)


def _validated_returns(value: pd.Series, label: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"{label} must be a pandas Series")
    if value.empty:
        raise ValueError(f"{label} must not be empty")
    result = pd.to_numeric(value.copy(deep=True), errors="coerce")
    if result.isna().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must contain finite numeric values")
    if (result < -1.0).any():
        raise ValueError(f"{label} cannot contain returns below -100%")
    if result.index.has_duplicates:
        raise ValueError(f"{label} index must be unique")
    if isinstance(result.index, pd.DatetimeIndex):
        result = result.sort_index()
    return result.astype(float)


def _return_summary(
    returns: pd.Series,
    *,
    annualization_factor: int,
    prefix: str,
) -> dict[str, float]:
    observations = len(returns)
    total_return = float((1.0 + returns).prod() - 1.0)
    annualized_return = float(
        (1.0 + total_return) ** (annualization_factor / observations) - 1.0
    )
    annualized_volatility = float(
        returns.std(ddof=1) * np.sqrt(annualization_factor)
    )
    return {
        f"{prefix}observations": float(observations),
        f"{prefix}total_return": total_return,
        f"{prefix}annualized_return": annualized_return,
        f"{prefix}annualized_volatility": annualized_volatility,
        f"{prefix}maximum_drawdown": float(drawdown_series(returns).min()),
    }
