"""Provider-neutral boundaries for point-in-time research history."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ConstituentHistoryService(Protocol):
    """Load canonical universe snapshots across a requested date range."""

    def load_constituent_history(
        self,
        universe_id: str,
        start_date,
        end_date,
        **parameters,
    ) -> pd.DataFrame: ...


@runtime_checkable
class MarketHistoryService(Protocol):
    """Load canonical daily observations for an explicit instrument set."""

    def load_market_history(
        self,
        instrument_ids: Iterable,
        start_date,
        end_date,
        **parameters,
    ) -> pd.DataFrame: ...


__all__ = ["ConstituentHistoryService", "MarketHistoryService"]
