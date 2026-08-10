"""Calendar-period, rolling-risk, and drawdown analytics plugins."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any
import numpy as np
import pandas as pd
from ..contracts import AnalyticsDiagnostic, AnalyticsValidationError
from .api import AnalyticsContext, AnalyticsPluginResult
from .support_common import episode_row, require_columns, simulation_daily


def _empty_annual_performance() -> pd.DataFrame:
    """Return the stable schema for unavailable calendar-period results."""

    return pd.DataFrame(
        {
            "period": pd.Series(dtype="object"),
            "start_date": pd.Series(dtype="datetime64[ns]"),
            "end_date": pd.Series(dtype="datetime64[ns]"),
            "observations": pd.Series(dtype="int64"),
            "index_return": pd.Series(dtype="float64"),
            "benchmark_return": pd.Series(dtype="float64"),
            "active_return": pd.Series(dtype="float64"),
        }
    )


@dataclass(frozen=True, slots=True)
class _CalendarPeriodPerformancePlugin:
    plugin_id: str = "calendar_period_performance"
    version: str = "1"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset(
        {"calendar_period_performance.annual"}
    )

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        del parameters
        daily = simulation_daily(context.simulation_result)
        index_column, benchmark_column = context.spec.return_series.columns
        require_columns(daily, (index_column, benchmark_column), self.plugin_id)
        working = daily[[index_column, benchmark_column]].copy()
        working.index = pd.to_datetime(working.index)
        if working.index.hasnans:
            raise AnalyticsValidationError(
                "simulation.daily index contains invalid business dates"
            )
        for column in working.columns:
            original = working[column]
            converted = pd.to_numeric(original, errors="coerce")
            if (original.notna() & converted.isna()).any():
                raise AnalyticsValidationError(
                    f"simulation.daily.{column} contains non-numeric values"
                )
            finite = converted.dropna().to_numpy(dtype=float)
            if not np.isfinite(finite).all():
                raise AnalyticsValidationError(
                    f"simulation.daily.{column} contains non-finite values"
                )
            working[column] = converted

        complete = working.dropna(how="any")
        dropped = len(working) - len(complete)
        diagnostics: list[AnalyticsDiagnostic] = []
        if dropped:
            diagnostics.append(
                AnalyticsDiagnostic(
                    level="warning",
                    code="calendar_period_incomplete_returns_dropped",
                    message=(
                        f"{dropped} daily rows without a complete index/benchmark "
                        "return pair were excluded from calendar-period performance."
                    ),
                )
            )
        if complete.empty:
            diagnostics.append(
                AnalyticsDiagnostic(
                    level="warning",
                    code="calendar_period_returns_unavailable",
                    message=(
                        "Daily simulation input contains no complete return pairs "
                        "for calendar-period performance."
                    ),
                )
            )
            return AnalyticsPluginResult(
                tables={"annual": _empty_annual_performance()},
                diagnostics=tuple(diagnostics),
            )
        if (complete < -1.0 - context.spec.weight_tolerance).any().any():
            raise AnalyticsValidationError("daily returns cannot be less than -100%")
        complete = complete.clip(lower=-1.0)

        rows: list[dict[str, Any]] = []
        for year, group in complete.groupby(complete.index.year, sort=True):
            index_return = float((1.0 + group[index_column]).prod() - 1.0)
            benchmark_return = float(
                (1.0 + group[benchmark_column]).prod() - 1.0
            )
            rows.append(
                {
                    "period": str(year),
                    "start_date": group.index.min().normalize(),
                    "end_date": group.index.max().normalize(),
                    "observations": len(group),
                    "index_return": index_return,
                    "benchmark_return": benchmark_return,
                    "active_return": index_return - benchmark_return,
                }
            )
        return AnalyticsPluginResult(
            tables={"annual": pd.DataFrame(rows)},
            diagnostics=tuple(diagnostics),
        )


@dataclass(frozen=True, slots=True)
class _RollingRiskPlugin:
    plugin_id: str = "rolling_risk"
    version: str = "1"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset({"rolling_risk.metrics"})

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        daily = simulation_daily(context.simulation_result)
        index_column, benchmark_column = context.spec.return_series.columns
        require_columns(daily, (index_column, benchmark_column), self.plugin_id)
        windows = tuple(int(value) for value in parameters.get("windows", (21, 63, 252)))
        if not windows or any(value <= 1 for value in windows):
            raise AnalyticsValidationError("rolling windows must be integers greater than one")
        index_return = pd.to_numeric(daily[index_column], errors="raise")
        benchmark_return = pd.to_numeric(daily[benchmark_column], errors="raise")
        active = index_return - benchmark_return
        rows: list[pd.DataFrame] = []
        for window in windows:
            minimum = max(2, math.ceil(window * 0.8))
            frame = pd.DataFrame(index=daily.index)
            frame["window"] = window
            frame["index_volatility"] = (
                index_return.rolling(window, min_periods=minimum).std(ddof=1)
                * math.sqrt(context.spec.annualization_factor)
            )
            frame["benchmark_volatility"] = (
                benchmark_return.rolling(window, min_periods=minimum).std(ddof=1)
                * math.sqrt(context.spec.annualization_factor)
            )
            frame["tracking_error"] = (
                active.rolling(window, min_periods=minimum).std(ddof=1)
                * math.sqrt(context.spec.annualization_factor)
            )
            frame["correlation"] = index_return.rolling(
                window,
                min_periods=minimum,
            ).corr(benchmark_return)
            rows.append(frame.dropna(how="all", subset=[
                "index_volatility",
                "benchmark_volatility",
                "tracking_error",
                "correlation",
            ]).reset_index(names="business_date"))
        return AnalyticsPluginResult(
            tables={"metrics": pd.concat(rows, ignore_index=True)}
        )


@dataclass(frozen=True, slots=True)
class _DrawdownEpisodesPlugin:
    plugin_id: str = "drawdown_episodes"
    version: str = "1"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset({"drawdown_episodes.episodes"})

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        del parameters
        daily = simulation_daily(context.simulation_result)
        index_column, _ = context.spec.return_series.columns
        require_columns(daily, (index_column,), self.plugin_id)
        returns = pd.to_numeric(daily[index_column], errors="raise")
        wealth = (1.0 + returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        episodes: list[dict[str, Any]] = []
        in_episode = False
        start: pd.Timestamp | None = None
        trough: pd.Timestamp | None = None
        minimum = 0.0
        for business_date, value in drawdown.items():
            date = pd.Timestamp(business_date).normalize()
            if value < 0 and not in_episode:
                in_episode = True
                start = date
                trough = date
                minimum = float(value)
            elif value < minimum:
                trough = date
                minimum = float(value)
            elif value >= -1e-15 and in_episode:
                episodes.append(
                    episode_row(
                        start,
                        trough,
                        date,
                        minimum,
                        recovered=True,
                    )
                )
                in_episode = False
                start = None
                trough = None
                minimum = 0.0
        if in_episode:
            episodes.append(
                episode_row(
                    start,
                    trough,
                    pd.Timestamp(drawdown.index[-1]).normalize(),
                    minimum,
                    recovered=False,
                )
            )
        return AnalyticsPluginResult(tables={"episodes": pd.DataFrame(episodes)})
