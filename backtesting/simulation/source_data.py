"""Market-data loading, coverage checks, and replay slicing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from icapa.data_sources.providers.registry import registry

from .assembly import slice_frame_from_date as _slice_frame_from_date
from .enums import WeightDrift
from .models import IndexSimulationResult, SimulationCheckpoint


class SimulationDataMixin:
    """Provide simulation source access and defensive date validation."""
    def _slice_replayed_state(
        self,
        result: IndexSimulationResult,
        *,
        calculation_start: pd.Timestamp,
    ) -> IndexSimulationResult:
        """Return the requested range after replaying from the prior review."""

        state_start = pd.Timestamp(calculation_start).normalize()
        if state_start >= self.start_date:
            return result
        daily = result.daily.loc[result.daily.index >= self.start_date].copy()
        if daily.empty:
            raise ValueError(
                "market data contains no observations in the requested range"
            )
        level_scales: dict[str, float] = {}
        for level_column in (
            column for column in daily.columns if column.endswith("_level")
        ):
            return_column = level_column.removesuffix("_level") + "_return"
            if return_column not in daily:
                raise ValueError(
                    f"simulation output is missing {return_column!r}"
                )
            observed = float(daily.iloc[0][level_column])
            target = float(self.params.base_value) * (
                1.0 + float(daily.iloc[0][return_column])
            )
            if not np.isfinite(observed) or observed <= 0.0:
                raise ValueError(
                    f"simulation produced an invalid {level_column!r}"
                )
            scale = target / observed
            daily.loc[:, level_column] = daily[level_column] * scale
            level_scales[level_column] = scale

        holdings = _slice_frame_from_date(
            result.holdings,
            self.start_date,
            level_name="business_date",
        )
        asset_returns = _slice_frame_from_date(
            result.asset_returns,
            self.start_date,
            level_name="business_date",
        )
        weight_snapshots = _slice_frame_from_date(
            result.weight_snapshots,
            self.start_date,
            level_name="applied_business_date",
        )
        rebalances = result.rebalances
        if not rebalances.empty:
            rebalances = rebalances.loc[
                pd.to_datetime(rebalances["applied_business_date"])
                >= self.start_date
            ].copy()
        checkpoint = result.checkpoint
        if checkpoint is not None:
            checkpoint = SimulationCheckpoint(
                business_date=checkpoint.business_date,
                index_weights=checkpoint.index_weights,
                benchmark_weights=checkpoint.benchmark_weights,
                levels={
                    name: value * level_scales.get(name, 1.0)
                    for name, value in checkpoint.levels.items()
                },
                previous_observations=checkpoint.previous_observations,
            )
        return IndexSimulationResult(
            daily=daily,
            holdings=holdings,
            rebalances=rebalances.reset_index(drop=True),
            asset_returns=asset_returns,
            metadata={
                **result.metadata,
                "start_date": str(self.start_date.date()),
                "state_replayed_from": str(state_start.date()),
            },
            checkpoint=checkpoint,
            weight_snapshots=weight_snapshots,
        )

    def _load_market_data(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        instrument_ids = sorted(
            {
                instrument_id
                for context in self.backtest_result.reviews.values()
                for instrument_id in context.cons.index
            },
            key=str,
        )
        if self.market_data is not None:
            available = self.market_data
            if "business_date" not in available.columns:
                raise ValueError("preloaded market_data is missing business_date")
            dates = pd.to_datetime(available["business_date"]).dt.normalize()
            available = available.loc[dates.between(start_date, end_date)].copy()
            if "instrument_id" not in available.columns:
                raise ValueError("preloaded market_data is missing instrument_id")
            return available.loc[
                available["instrument_id"].isin(instrument_ids)
            ].copy()
        if self.market_data_loader is not None:
            loaded = self.market_data_loader(
                instrument_ids=instrument_ids,
                start_date=start_date,
                end_date=end_date,
            )
            if not isinstance(loaded, pd.DataFrame):
                raise TypeError(
                    "market_data_loader must return a pandas DataFrame"
                )
            return loaded
        provider = registry.resolve(
            "load_daily_market_data",
            self.market_data_provider_name,
        )
        return provider.load_daily_market_data(
            instrument_ids=instrument_ids,
            start_date=start_date,
            end_date=end_date,
            **self.provider_parameters,
        )

    def _validate_business_date_coverage(
        self,
        market_data: pd.DataFrame,
        *,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> None:
        if self.business_days is None:
            return
        coverage_start = (
            self.start_date
            if start_date is None
            else pd.Timestamp(start_date).normalize()
        )
        coverage_end = (
            self.end_date
            if end_date is None
            else pd.Timestamp(end_date).normalize()
        )
        expected = self.business_days[
            (self.business_days >= coverage_start)
            & (self.business_days <= coverage_end)
        ]
        observed = pd.DatetimeIndex(
            pd.to_datetime(market_data["business_date"]).dt.normalize().unique()
        ).sort_values()
        missing = expected.difference(observed)
        unexpected = observed.difference(expected)
        if len(missing) or len(unexpected):
            raise ValueError(
                "market data business-date coverage does not match the supplied "
                f"calendar; missing={list(missing.date)}, "
                f"unexpected={list(unexpected.date)}"
            )

    def _uses_legacy_market_cap_drift(self) -> bool:
        """Return whether either portfolio still resolves through the v1 model."""

        return (
            self.params.weight_drift is WeightDrift.MARKET_CAP
            and (
                self.params.index_drift is None
                or self.params.benchmark_drift is None
            )
        )

    def _reject_legacy_market_cap_calculation(self) -> None:
        """Reject new calculations while allowing an exact v1 cache hit first."""

        if not self._uses_legacy_market_cap_drift():
            return
        raise ValueError(
            "WeightDrift.MARKET_CAP is available only for exact v1 cache "
            "replay. New simulations must select CapitalizationDrift or "
            "RelativeCapitalizationDrift explicitly for both the index and "
            "benchmark."
        )


__all__ = ["SimulationDataMixin"]
