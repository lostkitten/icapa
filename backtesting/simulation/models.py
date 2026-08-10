"""Stable result and continuation-state contracts for index simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .rebalance import rebalance_weight_snapshot_frame, with_turnover_aliases


@dataclass(frozen=True)
class IndexSimulationResult:
    """Daily index series, holdings, rebalances, and source observations."""

    daily: pd.DataFrame
    holdings: pd.DataFrame
    rebalances: pd.DataFrame
    asset_returns: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    checkpoint: "SimulationCheckpoint | None" = field(
        default=None,
        repr=False,
        compare=False,
    )
    weight_snapshots: pd.DataFrame = field(
        default_factory=lambda: rebalance_weight_snapshot_frame([]),
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rebalances",
            with_turnover_aliases(self.rebalances),
        )

    @property
    def one_way_turnover(self) -> pd.DataFrame:
        """Return rebalance events with canonical one-way turnover columns."""

        return self.rebalances.copy()

    @property
    def formal_turnover(self) -> pd.DataFrame:
        """Return the historical turnover view for compatibility."""

        columns = [
            column
            for column in ("index_turnover", "benchmark_turnover")
            if column in self.rebalances
        ]
        return self.rebalances.loc[:, columns].copy()

    @property
    def rebalance_weight_snapshots(self) -> pd.DataFrame:
        """Return pre-rebalance, target, and end-of-day weight snapshots."""

        return self.weight_snapshots


@dataclass(frozen=True)
class SimulationCheckpoint:
    """State required to extend a completed simulation without replaying it."""

    business_date: pd.Timestamp
    index_weights: pd.Series
    benchmark_weights: pd.Series
    levels: dict[str, float]
    previous_observations: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "business_date",
            pd.Timestamp(self.business_date).normalize(),
        )
        object.__setattr__(
            self,
            "index_weights",
            self.index_weights.copy(deep=True),
        )
        object.__setattr__(
            self,
            "benchmark_weights",
            self.benchmark_weights.copy(deep=True),
        )
        object.__setattr__(self, "levels", dict(self.levels))
        object.__setattr__(
            self,
            "previous_observations",
            self.previous_observations.copy(deep=True),
        )


__all__ = ["IndexSimulationResult", "SimulationCheckpoint"]
