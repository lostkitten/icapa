"""Small notebook-facing views over completed research runs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def research_summary(run: object) -> pd.DataFrame:
    """Return a compact, display-ready summary without recalculating results."""

    definition = getattr(run, "definition", None)
    manifest = getattr(run, "manifest", None)
    simulation = getattr(run, "simulation", None)
    reviews = getattr(getattr(run, "backtest", None), "reviews", {}) or {}
    daily = getattr(simulation, "daily", None)
    row: dict[str, Any] = {
        "index_id": getattr(definition, "index_id", None),
        "name": getattr(definition, "display_name", None),
        "execution_id": getattr(manifest, "execution_id", None),
        "status": getattr(manifest, "status", None),
        "review_count": len(reviews),
        "simulation_start": None,
        "simulation_end": None,
        "business_days": 0,
        "report_available": getattr(run, "report", None) is not None,
    }
    if isinstance(daily, pd.DataFrame) and not daily.empty:
        dates = pd.to_datetime(daily.index)
        row["simulation_start"] = dates.min().normalize()
        row["simulation_end"] = dates.max().normalize()
        row["business_days"] = int(len(dates.unique()))
    return pd.DataFrame([row])


def display_research_summary(run: object) -> object:
    """Display a run summary in IPython, with a clear optional-dependency error."""

    try:
        from IPython.display import display
    except ImportError as error:
        raise ImportError(
            "Notebook display requires the optional 'notebook' dependencies"
        ) from error
    frame = research_summary(run)
    display(frame)
    return frame


__all__ = ["display_research_summary", "research_summary"]
