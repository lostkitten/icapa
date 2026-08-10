"""Stateful execution engine for review-based daily index simulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from icapa.backtesting.reviews import BacktestResult
from icapa.backtesting.simulation.config import SimulationParams
from icapa.data_sources.contracts import validate_daily_market_data

from .assembly import latest_observations_before as _latest_observations_before
from .cache_contracts import (
    DefaultSimulationIdentityService,
    SimulationCacheMissError as CacheMissError,
    SimulationCachePolicy as CachePolicy,
    SimulationCacheStore,
    SimulationIdentityService,
)
from .calculation import SimulationCalculationMixin
from .models import IndexSimulationResult, SimulationCheckpoint
from .range_cache import SimulationRangeCacheMixin
from .result_cache import SimulationResultCacheMixin
from .segment_execution import ImmutableSegmentExecutionMixin
from .segment_identity import ImmutableSegmentIdentityMixin
from .segments import calendar_month_partitions as _calendar_month_partitions
from .source_data import SimulationDataMixin


@dataclass
class IndexSimulator(
    SimulationCalculationMixin,
    SimulationDataMixin,
    ImmutableSegmentExecutionMixin,
    ImmutableSegmentIdentityMixin,
    SimulationRangeCacheMixin,
    SimulationResultCacheMixin,
):
    """Simulate daily index levels from cached or freshly generated review weights."""

    backtest_result: BacktestResult
    market_data_provider_name: str
    start_date: object
    end_date: object
    provider_parameters: dict[str, Any] = field(default_factory=dict)
    params: SimulationParams = field(default_factory=SimulationParams)
    data_revision: str | None = None
    workspace: SimulationCacheStore | None = None
    cache_policy: CachePolicy | str = CachePolicy.REUSE
    segmented_cache: bool = False
    market_data: pd.DataFrame | None = field(default=None, repr=False)
    market_data_loader: Callable[..., pd.DataFrame] | None = field(
        default=None,
        repr=False,
    )
    business_days: pd.DatetimeIndex | None = field(default=None, repr=False)
    market_data_lineage: tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple,
        repr=False,
    )
    identity_service: SimulationIdentityService = field(
        default_factory=DefaultSimulationIdentityService,
        repr=False,
    )
    streaming: bool = False

    def __post_init__(self) -> None:
        if not self.market_data_provider_name:
            raise ValueError("market_data_provider_name is required")
        self.start_date = pd.Timestamp(self.start_date).normalize()
        self.end_date = pd.Timestamp(self.end_date).normalize()
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        if not self.backtest_result.reviews:
            raise ValueError("backtest_result must contain at least one review")
        self.cache_policy = CachePolicy(self.cache_policy)
        if not isinstance(self.segmented_cache, bool):
            raise TypeError("segmented_cache must be a bool")
        if not isinstance(self.streaming, bool):
            raise TypeError("streaming must be a bool")
        if self.market_data is not None:
            if not isinstance(self.market_data, pd.DataFrame):
                raise TypeError("market_data must be a pandas DataFrame or None")
            self.market_data = self.market_data.copy(deep=True)
        if self.market_data_loader is not None and not callable(
            self.market_data_loader
        ):
            raise TypeError("market_data_loader must be callable or None")
        if self.market_data is not None and self.market_data_loader is not None:
            raise ValueError(
                "market_data and market_data_loader cannot both be supplied"
            )
        if self.business_days is not None:
            days = pd.DatetimeIndex(
                pd.to_datetime(list(self.business_days))
            ).normalize()
            if days.has_duplicates:
                raise ValueError("business_days must not contain duplicates")
            self.business_days = days.sort_values()
        try:
            self.market_data_lineage = tuple(
                dict(record) for record in self.market_data_lineage
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "market_data_lineage must contain mappings"
            ) from exc
        if self.cache_policy is CachePolicy.READ_ONLY and self.workspace is None:
            raise ValueError("READ_ONLY simulation cache requires a workspace")
        required_identity_methods = (
            "digest",
            "safe_parameters",
            "source_identity",
            "runtime_identity",
            "dataframe_digest",
        )
        if any(
            not callable(getattr(self.identity_service, name, None))
            for name in required_identity_methods
        ):
            raise TypeError(
                "identity_service does not implement SimulationIdentityService"
            )

    def run(self) -> IndexSimulationResult:
        """Load market data and run a stateful daily index simulation."""

        if (
            self._uses_legacy_market_cap_drift()
            and not self._uses_legacy_cache_identity()
        ):
            self._reject_legacy_market_cap_calculation()
        cache_key = self._cache_key()
        cached = self._load_cached(
            cache_key,
            allow_miss=self.segmented_cache,
        )
        if cached is not None:
            return cached
        self._reject_legacy_market_cap_calculation()
        if self._supports_immutable_segments():
            segmented = self._run_immutable_segments(cache_key)
            if segmented is not None:
                return segmented
        elif self.segmented_cache and self.cache_policy is not CachePolicy.REFRESH:
            segmented = self._reuse_segment(cache_key)
            if segmented is not None:
                return segmented
        if self.cache_policy is CachePolicy.READ_ONLY:
            raise CacheMissError(
                "no exact or safely reusable simulation segment is cached for "
                f"{self.start_date.date()} through {self.end_date.date()}"
            )

        calculation_start = self._state_seed_start()
        if self.streaming:
            result = self._simulate_streaming(
                cache_key,
                calculation_start=calculation_start,
                load_start=self._market_data_request_start(calculation_start),
            )
            if result is None:
                raise ValueError(
                    "market data contains no observations in the requested range"
                )
        else:
            request_start = self._market_data_request_start(calculation_start)
            raw = self._load_market_data(request_start, self.end_date)
            market_data = validate_daily_market_data(raw)
            prior_observations = _latest_observations_before(
                market_data,
                calculation_start,
            )
            market_data = market_data.loc[
                market_data["business_date"].between(
                    calculation_start,
                    self.end_date,
                )
            ].copy()
            if market_data.empty:
                raise ValueError(
                    "market data contains no observations in the requested range"
                )
            self._validate_business_date_coverage(
                market_data,
                start_date=calculation_start,
                end_date=self.end_date,
            )
            result = self._simulate(
                market_data,
                cache_key,
                initial_previous_observations=prior_observations,
            )
        result = self._slice_replayed_state(
            result,
            calculation_start=calculation_start,
        )
        result = self._stabilize_reusable_levels(result)
        result = self._attach_market_data_lineage(result)
        self._save_cached(cache_key, result)
        self._record_segment(cache_key)
        return result

    def load_required_market_data(self) -> pd.DataFrame:
        """Load the exact source range needed to verify a simulation request.

        This performs no index calculation. Research workspaces use it when
        source content must be verified before a downstream cache lookup, such
        as providers without a snapshot token or strict read-only execution.
        """

        calculation_start = self._state_seed_start()
        request_start = self._market_data_request_start(calculation_start)
        if self.streaming:
            partitions = [
                self._load_market_data(partition_start, partition_end)
                for partition_start, partition_end in _calendar_month_partitions(
                    request_start,
                    self.end_date,
                )
            ]
            nonempty = [frame for frame in partitions if not frame.empty]
            if not nonempty:
                raise ValueError(
                    "market data contains no observations in the required range"
                )
            return pd.concat(nonempty, ignore_index=True)
        frame = self._load_market_data(request_start, self.end_date)
        if frame.empty:
            raise ValueError(
                "market data contains no observations in the required range"
            )
        return frame.reset_index(drop=True)

    def required_market_data_start(self) -> pd.Timestamp:
        """Return the earliest observation date needed for this request.

        This includes state replay from the prior effective date and any
        explicit drift-model lookback. Orchestrators use this same calculation
        when requesting a defensive business-day calendar.
        """

        return self._market_data_request_start(self._state_seed_start())

    def verify_required_market_data(self) -> int:
        """Verify required source partitions without retaining a full-history frame.

        This is used by high-level no-snapshot preflight. Streaming requests
        are validated one month at a time, allowing a source loader to persist
        and later reload verified Parquet partitions without a second provider
        call or one large in-memory concatenation.
        """

        calculation_start = self._state_seed_start()
        request_start = self._market_data_request_start(calculation_start)
        partitions = (
            _calendar_month_partitions(request_start, self.end_date)
            if self.streaming
            else ((request_start, self.end_date),)
        )
        rows = 0
        observed_any = False
        for partition_start, partition_end in partitions:
            raw = self._load_market_data(partition_start, partition_end)
            frame = validate_daily_market_data(raw)
            rows += len(frame)
            observed_any = observed_any or not frame.empty
            simulation_start = max(
                calculation_start,
                partition_start,
            )
            simulation_data = frame.loc[
                frame["business_date"].between(
                    simulation_start,
                    partition_end,
                )
            ]
            self._validate_business_date_coverage(
                simulation_data,
                start_date=simulation_start,
                end_date=partition_end,
            )
        if not observed_any:
            raise ValueError(
                "market data contains no observations in the required range"
            )
        return rows

__all__ = [
    "IndexSimulationResult",
    "IndexSimulator",
    "SimulationCheckpoint",
]
