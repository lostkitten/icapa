"""Deterministic event studies using business-day positions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class EventStudySpec:
    """Business-day window and alignment policy for an event study."""

    pre_event_days: int = 5
    post_event_days: int = 5
    event_alignment: str = "next"
    require_complete_window: bool = True

    def __post_init__(self) -> None:
        if self.pre_event_days < 0 or self.post_event_days < 0:
            raise ValueError("event windows must be non-negative")
        if self.event_alignment not in {"exact", "next", "previous"}:
            raise ValueError(
                "event_alignment must be 'exact', 'next', or 'previous'"
            )


@dataclass(frozen=True, slots=True)
class EventStudyResult:
    """Per-event windows and cross-event summary statistics."""

    windows: pd.DataFrame
    summary: pd.DataFrame
    skipped_events: tuple[pd.Timestamp, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", self.windows.copy(deep=True))
        object.__setattr__(self, "summary", self.summary.copy(deep=True))
        object.__setattr__(
            self,
            "skipped_events",
            tuple(pd.Timestamp(value).normalize() for value in self.skipped_events),
        )


def run_event_study(
    index_returns: pd.Series,
    events: Iterable[object],
    *,
    benchmark_returns: pd.Series | None = None,
    spec: EventStudySpec | None = None,
) -> EventStudyResult:
    """Calculate event-window and cumulative abnormal returns."""

    selected = spec or EventStudySpec()
    index = _return_series(index_returns, "index_returns")
    aligned = index.to_frame("index_return")
    if benchmark_returns is not None:
        benchmark = _return_series(benchmark_returns, "benchmark_returns")
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
    else:
        aligned["benchmark_return"] = 0.0

    event_dates = tuple(
        sorted({pd.Timestamp(event).normalize() for event in events})
    )
    if not event_dates:
        raise ValueError("events must contain at least one date")
    rows: list[pd.DataFrame] = []
    skipped: list[pd.Timestamp] = []
    for requested_date in event_dates:
        position = _event_position(
            aligned.index,
            requested_date,
            selected.event_alignment,
        )
        if position is None:
            skipped.append(requested_date)
            continue
        start = position - selected.pre_event_days
        stop = position + selected.post_event_days
        if start < 0 or stop >= len(aligned):
            if selected.require_complete_window:
                skipped.append(requested_date)
                continue
            start = max(0, start)
            stop = min(len(aligned) - 1, stop)
        window = aligned.iloc[start : stop + 1].copy()
        relative = np.arange(start - position, stop - position + 1)
        window.insert(0, "relative_day", relative)
        window.insert(0, "business_date", window.index)
        window.insert(0, "event_date", aligned.index[position])
        window.insert(0, "requested_event_date", requested_date)
        window["abnormal_return"] = (
            window["index_return"] - window["benchmark_return"]
        )
        window["cumulative_abnormal_return"] = window[
            "abnormal_return"
        ].cumsum()
        rows.append(window.reset_index(drop=True))
    if not rows:
        raise ValueError("no events could be aligned to a valid event window")

    windows = pd.concat(rows, ignore_index=True)
    grouped = windows.groupby("relative_day", sort=True)
    summary = grouped["abnormal_return"].agg(
        event_count="count",
        mean_abnormal_return="mean",
        median_abnormal_return="median",
        standard_deviation="std",
    )
    summary["standard_error"] = (
        summary["standard_deviation"]
        / np.sqrt(summary["event_count"].astype(float))
    )
    summary["t_statistic"] = (
        summary["mean_abnormal_return"] / summary["standard_error"]
    )
    summary["mean_cumulative_abnormal_return"] = grouped[
        "cumulative_abnormal_return"
    ].mean()
    return EventStudyResult(
        windows=windows,
        summary=summary.reset_index(),
        skipped_events=tuple(skipped),
    )


def calculate_event_non_event_returns(
    returns: pd.Series,
    events: Iterable[object],
    *,
    exclusion_days: int = 0,
) -> pd.Series:
    """Compare compound returns on event and non-event business dates."""

    if exclusion_days < 0:
        raise ValueError("exclusion_days must be non-negative")
    values = _return_series(returns, "returns")
    event_positions: set[int] = set()
    for event in events:
        position = _event_position(
            values.index,
            pd.Timestamp(event).normalize(),
            "next",
        )
        if position is None:
            continue
        event_positions.update(
            range(
                max(0, position - exclusion_days),
                min(len(values), position + exclusion_days + 1),
            )
        )
    if not event_positions:
        raise ValueError("no event dates align to the return series")
    mask = np.zeros(len(values), dtype=bool)
    mask[list(event_positions)] = True
    event_return = float((1.0 + values.iloc[mask]).prod() - 1.0)
    non_event_return = float((1.0 + values.iloc[~mask]).prod() - 1.0)
    return pd.Series(
        {
            "event_observations": float(mask.sum()),
            "non_event_observations": float((~mask).sum()),
            "event_compound_return": event_return,
            "non_event_compound_return": non_event_return,
        },
        dtype=float,
        name="value",
    )


def _return_series(value: pd.Series, label: str) -> pd.Series:
    if not isinstance(value, pd.Series) or value.empty:
        raise ValueError(f"{label} must be a non-empty pandas Series")
    if not isinstance(value.index, pd.DatetimeIndex):
        raise TypeError(f"{label} must use a DatetimeIndex")
    result = pd.to_numeric(value.copy(deep=True), errors="coerce")
    if result.isna().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must contain finite numeric values")
    if (result < -1.0).any():
        raise ValueError(f"{label} cannot contain returns below -100%")
    result.index = pd.DatetimeIndex(result.index).normalize()
    if result.index.has_duplicates:
        raise ValueError(f"{label} business dates must be unique")
    return result.sort_index().astype(float)


def _event_position(
    index: pd.DatetimeIndex,
    event_date: pd.Timestamp,
    alignment: str,
) -> int | None:
    if event_date in index:
        return int(index.get_loc(event_date))
    if alignment == "exact":
        return None
    if alignment == "next":
        position = int(index.searchsorted(event_date, side="left"))
        return position if position < len(index) else None
    position = int(index.searchsorted(event_date, side="right")) - 1
    return position if position >= 0 else None
