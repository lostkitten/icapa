"""Optional notebook plots for completed research results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


DEFAULT_LEVEL_COLUMNS = (
    "index_net_total_level",
    "benchmark_net_total_level",
)


def plot_index_levels(
    result: object,
    *,
    columns: Sequence[str] = DEFAULT_LEVEL_COLUMNS,
    ax: Any | None = None,
) -> Any:
    """Plot stored index levels without recalculating a simulation.

    ``result`` may be a completed research run, a simulation result, or the
    simulation's daily DataFrame. Matplotlib remains optional and is imported
    only when this function is called.
    """

    daily = _daily_frame(result)
    selected = tuple(dict.fromkeys(map(str, columns)))
    if not selected:
        raise ValueError("columns must contain at least one level field")
    missing = [column for column in selected if column not in daily.columns]
    if missing:
        raise ValueError(
            f"simulation daily data is missing level columns: {missing}"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Notebook plots require the optional 'notebook' dependencies"
        ) from error

    if ax is None:
        _, ax = plt.subplots()
    business_dates = pd.to_datetime(daily.index)
    for column in selected:
        values = pd.to_numeric(daily[column], errors="raise")
        ax.plot(business_dates, values, label=column)
    ax.set_xlabel("Business date")
    ax.set_ylabel("Index level")
    ax.legend()
    return ax


def _daily_frame(result: object) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        daily = result
    else:
        simulation = getattr(result, "simulation", result)
        daily = getattr(simulation, "daily", None)
    if not isinstance(daily, pd.DataFrame) or daily.empty:
        raise ValueError("result must expose non-empty simulation daily data")
    return daily


__all__ = ["DEFAULT_LEVEL_COLUMNS", "plot_index_levels"]
