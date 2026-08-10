"""Generic helpers shared by analytics plugins."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import math
import numpy as np
import pandas as pd
from .api import MissingAnalyticsInput

def simulation_daily(simulation_result: object | None) -> pd.DataFrame:
    if simulation_result is None:
        raise MissingAnalyticsInput("daily simulation input is unavailable")
    daily = getattr(simulation_result, "daily", None)
    if not isinstance(daily, pd.DataFrame) or daily.empty:
        raise MissingAnalyticsInput("daily simulation input is unavailable")
    result = daily.copy(deep=True)
    result.index = pd.DatetimeIndex(
        pd.to_datetime(result.index),
        name="business_date",
    )
    return result.sort_index()


def require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    plugin_id: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise MissingAnalyticsInput(
            f"{plugin_id} requires daily columns: {missing}"
        )


def episode_row(
    start: pd.Timestamp | None,
    trough: pd.Timestamp | None,
    end: pd.Timestamp,
    maximum_drawdown: float,
    *,
    recovered: bool,
) -> dict[str, Any]:
    assert start is not None
    assert trough is not None
    return {
        "start_date": start,
        "trough_date": trough,
        "end_date": end,
        "maximum_drawdown": maximum_drawdown,
        "duration_business_days": int(np.busday_count(start.date(), end.date())) + 1,
        "recovered": recovered,
    }


def json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).isoformat())
    if hasattr(value, "item"):
        try:
            return json_scalar(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return str(
            {
                str(key): json_scalar(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return str([json_scalar(item) for item in value])
    return str(value)
