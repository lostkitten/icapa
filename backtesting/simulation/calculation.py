"""Daily portfolio calculation and rebalance state transitions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from icapa.data_sources.contracts import validate_daily_market_data

from .assembly import (
    combine_simulation_results as _combine_simulation_results,
    empty_asset_return_frame as _empty_asset_return_frame,
    latest_observations_before as _latest_observations_before,
    legacy_simulation_params_payload as _legacy_simulation_params_payload,
    simulation_params_payload as _simulation_params_payload,
)
from .drift import CapitalizationDrift, RelativeCapitalizationDrift
from .enums import (
    RebalancePhase,
    RebalanceTiming,
    WeightSnapshotMode,
)
from .models import IndexSimulationResult, SimulationCheckpoint
from .rebalance import (
    append_holding_rows as _append_holding_rows,
    append_rebalance_weight_snapshot_rows as _append_rebalance_weight_snapshot_rows,
    holding_frame as _holding_frame,
    one_way_turnover as _one_way_turnover,
    rebalance_frame as _rebalance_frame,
    rebalance_weight_snapshot_frame as _rebalance_weight_snapshot_frame,
)
from .returns import (
    normalize_weights as _normalized_weights,
    portfolio_return_factors as _portfolio_factors,
)
from .segments import calendar_month_partitions as _calendar_month_partitions


class SimulationCalculationMixin:
    """Implement daily return calculation and effective-date rebalancing."""
    def _simulate_streaming(
        self,
        cache_key: str,
        *,
        calculation_start: pd.Timestamp,
        load_start: pd.Timestamp,
        initial_checkpoint: SimulationCheckpoint | None = None,
    ) -> IndexSimulationResult | None:
        """Calculate one calendar-month market-data partition at a time."""

        calculation_start = pd.Timestamp(calculation_start).normalize()
        load_start = pd.Timestamp(load_start).normalize()
        checkpoint = initial_checkpoint
        previous_observations: pd.DataFrame | None = None
        partition_results: list[IndexSimulationResult] = []
        loaded_partitions = 0
        calculated_partitions = 0

        for partition_start, partition_end in _calendar_month_partitions(
            load_start,
            self.end_date,
        ):
            raw = self._load_market_data(partition_start, partition_end)
            market_data = validate_daily_market_data(raw)
            loaded_partitions += 1

            if checkpoint is None:
                prior = _latest_observations_before(
                    market_data,
                    calculation_start,
                )
                if prior is not None:
                    previous_observations = prior

            simulation_start = max(calculation_start, partition_start)
            simulation_data = market_data.loc[
                market_data["business_date"].between(
                    simulation_start,
                    partition_end,
                )
            ].copy()
            self._validate_business_date_coverage(
                simulation_data,
                start_date=simulation_start,
                end_date=partition_end,
            )
            if simulation_data.empty:
                continue

            partition_result = self._simulate(
                simulation_data,
                cache_key,
                initial_checkpoint=checkpoint,
                initial_previous_observations=previous_observations,
            )
            calculated_partitions += 1
            checkpoint = partition_result.checkpoint
            partition_results.append(partition_result)

        if not partition_results:
            return None
        combined = _combine_simulation_results(
            partition_results,
            cache_key=cache_key,
        )
        return IndexSimulationResult(
            daily=combined.daily,
            holdings=combined.holdings,
            rebalances=combined.rebalances,
            asset_returns=combined.asset_returns,
            metadata={
                **combined.metadata,
                "cache_key": cache_key,
                "cache_source": "computed",
                "calculation_mode": "calendar_month_partitions",
                "market_data_partitions_loaded": loaded_partitions,
                "market_data_partitions_calculated": calculated_partitions,
                "start_date": str(self.start_date.date()),
                "end_date": str(self.end_date.date()),
            },
            checkpoint=combined.checkpoint,
            weight_snapshots=combined.weight_snapshots,
        )

    def _simulate(
        self,
        market_data: pd.DataFrame,
        cache_key: str,
        *,
        initial_checkpoint: SimulationCheckpoint | None = None,
        initial_previous_observations: pd.DataFrame | None = None,
    ) -> IndexSimulationResult:
        required = {
            "instrument_id",
            "business_date",
            "price_return",
            "gross_dividend",
            "net_dividend",
            "market_cap",
        }
        missing = sorted(required.difference(market_data.columns))
        if missing:
            raise ValueError(f"daily market data is missing columns: {missing}")
        numeric = ["price_return", "gross_dividend", "net_dividend", "market_cap"]
        for column in numeric:
            market_data[column] = pd.to_numeric(
                market_data[column],
                errors="raise",
            )
            if not np.isfinite(market_data[column]).all():
                raise ValueError(f"daily market data contains non-finite {column}")
        if (market_data["market_cap"] < 0).any():
            raise ValueError("daily market data contains negative market_cap")

        observations_by_date = {
            business_date: frame.set_index("instrument_id", verify_integrity=True)
            for business_date, frame in market_data.groupby(
                "business_date",
                sort=True,
            )
        }
        business_dates = pd.DatetimeIndex(sorted(observations_by_date))
        resume_after = (
            None if initial_checkpoint is None else initial_checkpoint.business_date
        )
        applications = self._review_applications(
            business_dates,
            resume_after=resume_after,
        )
        first_date = business_dates[0]
        if initial_checkpoint is None:
            active_applications = [
                applied
                for applied in applications
                if applied["applied_business_date"] <= first_date
            ]
            if not active_applications:
                first_effective = min(
                    pd.Timestamp(item).normalize()
                    for item in self.backtest_result.reviews
                )
                raise ValueError(
                    "simulation starts before the first available review becomes "
                    "effective; "
                    f"first effective date is {first_effective.date()}"
                )

        current_index: pd.Series | None = (
            None
            if initial_checkpoint is None
            else initial_checkpoint.index_weights.copy(deep=True)
        )
        current_benchmark: pd.Series | None = (
            None
            if initial_checkpoint is None
            else initial_checkpoint.benchmark_weights.copy(deep=True)
        )
        previous_observations = (
            initial_previous_observations
            if initial_checkpoint is None
            else initial_checkpoint.previous_observations.copy(deep=True)
        )
        daily_rows: list[dict[str, Any]] = []
        holding_rows: list[dict[str, Any]] = []
        rebalance_rows: list[dict[str, Any]] = []
        rebalance_weight_rows: list[dict[str, Any]] = []
        levels = (
            {
                "index_price_level": float(self.params.base_value),
                "index_gross_total_level": float(self.params.base_value),
                "index_net_total_level": float(self.params.base_value),
                "benchmark_price_level": float(self.params.base_value),
                "benchmark_gross_total_level": float(self.params.base_value),
                "benchmark_net_total_level": float(self.params.base_value),
            }
            if initial_checkpoint is None
            else dict(initial_checkpoint.levels)
        )
        applications_by_date: dict[pd.Timestamp, dict[str, Any]] = {
            item["applied_business_date"]: item for item in applications
        }

        for business_date in business_dates:
            observation = observations_by_date[business_date]
            application = applications_by_date.get(business_date)
            rebalance_snapshot: dict[str, Any] | None = None
            rebalance_before_return = (
                application is not None
                and (
                    self.params.rebalance_phase is RebalancePhase.OPEN
                    or current_index is None
                    or current_benchmark is None
                )
            )
            if rebalance_before_return:
                pre_rebalance_index = (
                    None if current_index is None else current_index.copy()
                )
                pre_rebalance_benchmark = (
                    None
                    if current_benchmark is None
                    else current_benchmark.copy()
                )
                (
                    current_index,
                    current_benchmark,
                    rebalance_row,
                ) = self._apply_rebalance(
                    application,
                    business_date,
                    current_index,
                    current_benchmark,
                    observation,
                    previous_observations,
                )
                rebalance_rows.append(rebalance_row)
                rebalance_snapshot = {
                    "application": application,
                    "phase": RebalancePhase.OPEN,
                    "pre_index": pre_rebalance_index,
                    "pre_benchmark": pre_rebalance_benchmark,
                    "target_index": current_index.copy(),
                    "target_benchmark": current_benchmark.copy(),
                }
            if current_index is None or current_benchmark is None:
                previous_observations = observation
                continue

            index_open = current_index.copy()
            benchmark_open = current_benchmark.copy()
            index_factors = _portfolio_factors(
                observation,
                index_open,
                self.params.dividend_treatment,
                business_date,
            )
            benchmark_factors = _portfolio_factors(
                observation,
                benchmark_open,
                self.params.dividend_treatment,
                business_date,
            )
            current_index = self.params.resolved_index_drift.drift(
                index_open,
                observation,
                business_date=business_date,
                previous_observations=previous_observations,
            )
            current_benchmark = self.params.resolved_benchmark_drift.drift(
                benchmark_open,
                observation,
                business_date=business_date,
                previous_observations=previous_observations,
            )
            returns = {
                "index_price_return": index_factors["price"] - 1.0,
                "index_gross_total_return": index_factors["gross"] - 1.0,
                "index_net_total_return": index_factors["net"] - 1.0,
                "benchmark_price_return": benchmark_factors["price"] - 1.0,
                "benchmark_gross_total_return": benchmark_factors["gross"] - 1.0,
                "benchmark_net_total_return": benchmark_factors["net"] - 1.0,
            }
            for prefix in ("index", "benchmark"):
                for return_name, level_name in (
                    ("price_return", "price_level"),
                    ("gross_total_return", "gross_total_level"),
                    ("net_total_return", "net_total_level"),
                ):
                    level_column = f"{prefix}_{level_name}"
                    levels[level_column] *= 1.0 + returns[
                        f"{prefix}_{return_name}"
                    ]
            daily_rows.append(
                {
                    "business_date": business_date,
                    **returns,
                    **levels,
                    "active_price_return": returns["index_price_return"]
                    - returns["benchmark_price_return"],
                    "active_gross_total_return": returns[
                        "index_gross_total_return"
                    ]
                    - returns["benchmark_gross_total_return"],
                    "active_net_total_return": returns["index_net_total_return"]
                    - returns["benchmark_net_total_return"],
                }
            )

            if (
                application is not None
                and not rebalance_before_return
                and self.params.rebalance_phase is RebalancePhase.CLOSE
            ):
                pre_rebalance_index = current_index.copy()
                pre_rebalance_benchmark = current_benchmark.copy()
                (
                    current_index,
                    current_benchmark,
                    rebalance_row,
                ) = self._apply_rebalance(
                    application,
                    business_date,
                    current_index,
                    current_benchmark,
                    observation,
                    previous_observations,
                )
                rebalance_rows.append(rebalance_row)
                rebalance_snapshot = {
                    "application": application,
                    "phase": RebalancePhase.CLOSE,
                    "pre_index": pre_rebalance_index,
                    "pre_benchmark": pre_rebalance_benchmark,
                    "target_index": current_index.copy(),
                    "target_benchmark": current_benchmark.copy(),
                }

            if (
                rebalance_snapshot is not None
                and self._should_materialize_rebalance_snapshots()
            ):
                _append_rebalance_weight_snapshot_rows(
                    rebalance_weight_rows,
                    business_date=business_date,
                    application=rebalance_snapshot["application"],
                    rebalance_phase=rebalance_snapshot["phase"],
                    pre_index=rebalance_snapshot["pre_index"],
                    target_index=rebalance_snapshot["target_index"],
                    end_index=current_index,
                    pre_benchmark=rebalance_snapshot["pre_benchmark"],
                    target_benchmark=rebalance_snapshot["target_benchmark"],
                    end_benchmark=current_benchmark,
                )

            if self._should_materialize_weights(application is not None):
                _append_holding_rows(
                    holding_rows,
                    business_date=business_date,
                    index_open=index_open,
                    index_close=current_index,
                    benchmark_open=benchmark_open,
                    benchmark_close=current_benchmark,
                )
            previous_observations = observation

        if not daily_rows:
            raise ValueError("simulation produced no daily index observations")
        daily = pd.DataFrame.from_records(daily_rows).set_index("business_date")
        holdings = _holding_frame(holding_rows)
        rebalances = _rebalance_frame(rebalance_rows)
        weight_snapshots = _rebalance_weight_snapshot_frame(
            rebalance_weight_rows
        )
        assets = (
            market_data.set_index(["business_date", "instrument_id"]).sort_index()
            if self.params.materialization.include_asset_returns
            else _empty_asset_return_frame(market_data.columns)
        )
        checkpoint = SimulationCheckpoint(
            business_date=business_dates[-1],
            index_weights=current_index,
            benchmark_weights=current_benchmark,
            levels=levels,
            previous_observations=observations_by_date[business_dates[-1]],
        )
        return IndexSimulationResult(
            daily=daily,
            holdings=holdings,
            rebalances=rebalances,
            asset_returns=assets,
            metadata={
                "cache_key": cache_key,
                "cache_source": "computed",
                "market_data_provider_name": self.market_data_provider_name,
                "data_revision": self.data_revision,
                "start_date": str(self.start_date.date()),
                "end_date": str(self.end_date.date()),
                "simulation_params": (
                    _legacy_simulation_params_payload(self.params)
                    if self._uses_legacy_cache_identity()
                    else _simulation_params_payload(self.params)
                ),
            },
            checkpoint=checkpoint,
            weight_snapshots=weight_snapshots,
        )

    def _review_applications(
        self,
        business_dates: pd.DatetimeIndex,
        *,
        resume_after: pd.Timestamp | None = None,
    ) -> list[dict[str, Any]]:
        review_items = sorted(
            (
                pd.Timestamp(effective_date).normalize(),
                context,
            )
            for effective_date, context in self.backtest_result.reviews.items()
        )
        first_business_date = business_dates[0]
        if resume_after is None:
            latest_prior = [
                item for item in review_items if item[0] <= first_business_date
            ][-1:]
            future = [
                item
                for item in review_items
                if first_business_date < item[0] <= business_dates[-1]
            ]
        else:
            latest_prior = []
            future = [
                item
                for item in review_items
                if resume_after < item[0] <= business_dates[-1]
            ]
        applications: list[dict[str, Any]] = []
        for scheduled, context in [*latest_prior, *future]:
            if self.params.rebalance_timing is RebalanceTiming.EXACT_DATE:
                if scheduled not in business_dates:
                    raise ValueError(
                        f"effective date is not a market-data business day: {scheduled.date()}"
                    )
                applied = scheduled
            else:
                position = int(business_dates.searchsorted(scheduled, side="left"))
                if position >= len(business_dates):
                    continue
                applied = business_dates[position]
            applications.append(
                {
                    "scheduled_effective_date": scheduled,
                    "applied_business_date": applied,
                    "reference_date": pd.Timestamp(context.reference_date).normalize(),
                }
            )
        applied_dates = [item["applied_business_date"] for item in applications]
        if len(applied_dates) != len(set(applied_dates)):
            raise ValueError("multiple reviews resolve to the same applied business date")
        return applications

    def _apply_rebalance(
        self,
        application: dict[str, Any],
        business_date: pd.Timestamp,
        current_index: pd.Series | None,
        current_benchmark: pd.Series | None,
        observations: pd.DataFrame,
        previous_observations: pd.DataFrame | None,
    ) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
        context = self.backtest_result.reviews[
            application["scheduled_effective_date"]
        ]
        target_index = _normalized_weights(
            context.cons["index_weight"],
            "index_weight",
        )
        target_benchmark = _normalized_weights(
            context.cons["benchmark_weight"],
            "benchmark_weight",
        )
        capitalization_observations = (
            previous_observations
            if self.params.rebalance_phase is RebalancePhase.OPEN
            else observations
        )
        self._validate_capitalization_target(
            strategy=self.params.resolved_index_drift,
            target_weights=target_index,
            capitalization_observations=capitalization_observations,
            business_date=business_date,
            portfolio_label="index",
        )
        self._validate_capitalization_target(
            strategy=self.params.resolved_benchmark_drift,
            target_weights=target_benchmark,
            capitalization_observations=capitalization_observations,
            business_date=business_date,
            portfolio_label="benchmark",
        )
        index_turnover = (
            np.nan
            if current_index is None
            else _one_way_turnover(current_index, target_index)
        )
        benchmark_turnover = (
            np.nan
            if current_benchmark is None
            else _one_way_turnover(current_benchmark, target_benchmark)
        )
        return (
            target_index,
            target_benchmark,
            {
                "scheduled_effective_date": application[
                    "scheduled_effective_date"
                ],
                "applied_business_date": business_date,
                "reference_date": application["reference_date"],
                "one_way_turnover": index_turnover,
                "benchmark_one_way_turnover": benchmark_turnover,
                "index_turnover": index_turnover,
                "benchmark_turnover": benchmark_turnover,
                "instrument_count": int(len(target_index)),
                # Cache provenance belongs in the run manifest, not numerical
                # simulation output. Keeping a stable value preserves the
                # existing column without making result identity operational.
                "review_source": "review_target",
            },
        )

    @staticmethod
    def _validate_capitalization_target(
        *,
        strategy,
        target_weights: pd.Series,
        capitalization_observations: pd.DataFrame | None,
        business_date: pd.Timestamp,
        portfolio_label: str,
    ) -> None:
        if isinstance(strategy, CapitalizationDrift):
            strategy.validate_target(
                target_weights,
                capitalization_observations,
                business_date=business_date,
                portfolio_label=portfolio_label,
            )

    def _should_materialize_weights(self, is_rebalance_date: bool) -> bool:
        mode = self.params.materialization.weight_snapshots
        return mode is WeightSnapshotMode.DAILY or (
            mode is WeightSnapshotMode.REBALANCE and is_rebalance_date
        )

    def _should_materialize_rebalance_snapshots(self) -> bool:
        return (
            self.params.materialization.weight_snapshots
            is not WeightSnapshotMode.NONE
        )

    def _state_seed_start(self) -> pd.Timestamp:
        """Return the authoritative review date needed to seed opening state."""

        prior_effective_dates = [
            pd.Timestamp(effective_date).normalize()
            for effective_date in self.backtest_result.reviews
            if pd.Timestamp(effective_date).normalize() <= self.start_date
        ]
        if not prior_effective_dates:
            return self.start_date
        if (
            not self._uses_legacy_cache_identity()
            and len(prior_effective_dates) > 1
        ):
            return sorted(prior_effective_dates)[-2]
        return max(prior_effective_dates)

    def _market_data_request_start(
        self,
        calculation_start: pd.Timestamp | None = None,
    ) -> pd.Timestamp:
        state_start = (
            self._state_seed_start()
            if calculation_start is None
            else pd.Timestamp(calculation_start).normalize()
        )
        lookbacks = []
        for strategy in (
            self.params.resolved_index_drift,
            self.params.resolved_benchmark_drift,
        ):
            needs_prior = isinstance(strategy, RelativeCapitalizationDrift) or (
                isinstance(strategy, CapitalizationDrift)
                and self.params.rebalance_phase is RebalancePhase.OPEN
            )
            if needs_prior:
                lookbacks.append(strategy.lookback_calendar_days)
        if not lookbacks:
            return state_start
        return state_start - pd.Timedelta(days=max(lookbacks))


__all__ = ["SimulationCalculationMixin"]
