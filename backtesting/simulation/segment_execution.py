"""Planning, calculation, and persistence of immutable simulation segments."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd

from icapa.data_sources.contracts import validate_daily_market_data
from .cache_contracts import (
    SimulationCacheMissError as CacheMissError,
    SimulationCachePolicy as CachePolicy,
)

from .assembly import (
    checkpoint_with_levels as _checkpoint_with_levels,
    combine_simulation_results as _combine_simulation_results,
    latest_observations_before as _latest_observations_before,
    merged_source_data_records as _merged_source_data_records,
    rescale_simulation_levels as _rescale_simulation_levels,
)
from .models import IndexSimulationResult, SimulationCheckpoint
from .segments import (
    ImmutableSimulationSegment as _ImmutableSimulationSegment,
    calendar_month_partitions as _calendar_month_partitions,
)


class ImmutableSegmentExecutionMixin:
    """Plan, calculate, save, and assemble immutable simulation segments."""

    def _supports_immutable_segments(self) -> bool:
        """Return whether the additive v2 segment store is available."""

        return (
            self.segmented_cache
            and self.workspace is not None
            and self.data_revision is not None
            and not self._uses_legacy_cache_identity()
        )

    def _run_immutable_segments(
        self,
        cache_key: str,
    ) -> IndexSimulationResult | None:
        """Assemble immutable effective-date and open-tail segment artifacts.

        Closed holding periods use ``[effective_i, effective_i+1)`` calendar
        bounds. The final open holding period is checkpointed by calendar
        month. Every stored piece starts its level columns at the configured
        base and therefore represents return factors rather than an absolute
        index level. Its ending weights and prior observations are sufficient
        to continue the next piece without replaying an earlier month.
        """

        plan = self._immutable_segment_plan()
        if not plan:
            return None
        catalog = (
            []
            if self.cache_policy is CachePolicy.REFRESH
            else self._load_immutable_segment_catalog()
        )
        rejected_keys: set[str] = set()
        assembled: list[IndexSimulationResult] = []
        opening_checkpoint: SimulationCheckpoint | None = None
        reused_segments = 0
        computed_segments = 0
        reused_after_computation = False
        calculation_start = plan[0].start_date

        for desired in plan:
            cursor = desired.start_date
            while cursor <= desired.end_date:
                candidate = self._best_immutable_segment_candidate(
                    catalog,
                    desired=desired,
                    cursor=cursor,
                    opening_checkpoint=opening_checkpoint,
                    rejected_keys=rejected_keys,
                )
                if candidate is not None:
                    candidate_key = str(candidate["cache_key"])
                    local = self._load_cached(
                        candidate_key,
                        allow_miss=True,
                    )
                    if local is None or local.checkpoint is None:
                        rejected_keys.add(candidate_key)
                        continue
                    global_piece = self._scale_segment_from_opening_state(
                        local,
                        opening_checkpoint,
                    )
                    assembled.append(global_piece)
                    opening_checkpoint = global_piece.checkpoint
                    cursor = pd.Timestamp(
                        candidate["end_date"]
                    ).normalize() + pd.Timedelta(days=1)
                    reused_after_computation = (
                        reused_after_computation or computed_segments > 0
                    )
                    reused_segments += 1
                    continue

                if self.cache_policy is CachePolicy.READ_ONLY:
                    raise CacheMissError(
                        "READ_ONLY simulation cache is missing immutable "
                        f"segment coverage beginning {cursor.date()}"
                    )
                piece = _ImmutableSimulationSegment(
                    start_date=cursor,
                    end_date=desired.end_date,
                    effective_date=desired.effective_date,
                    next_effective_date=desired.next_effective_date,
                    kind=(
                        desired.kind
                        if cursor == desired.start_date
                        else f"{desired.kind}_remainder"
                    ),
                    target_checksum=desired.target_checksum,
                    previous_target_checksum=(desired.previous_target_checksum),
                )
                piece_key = self._immutable_segment_key(
                    piece,
                    opening_checkpoint=opening_checkpoint,
                )
                local, partition_digest = self._calculate_immutable_segment(
                    piece,
                    piece_key,
                    opening_checkpoint=opening_checkpoint,
                )
                if local is None:
                    cursor = piece.end_date + pd.Timedelta(days=1)
                    continue
                local = self._attach_market_data_lineage(local)
                local = self._with_segment_metadata(
                    local,
                    segment=piece,
                    cache_key=piece_key,
                    partition_digest=partition_digest,
                    cache_source="computed_immutable_segment",
                    opening_checkpoint=opening_checkpoint,
                )
                self._save_cached(piece_key, local)
                persisted_local = self._load_cached(
                    piece_key,
                    allow_miss=True,
                    after_write=True,
                )
                if persisted_local is not None:
                    # Continue from the canonical persisted representation.
                    # This keeps chained state identities stable even for the
                    # v1 JSON store, while Parquet remains float64 exact.
                    local = persisted_local
                self._record_immutable_segment(
                    piece,
                    cache_key=piece_key,
                    partition_digest=partition_digest,
                    opening_checkpoint=opening_checkpoint,
                )
                global_piece = self._scale_segment_from_opening_state(
                    local,
                    opening_checkpoint,
                )
                assembled.append(global_piece)
                opening_checkpoint = global_piece.checkpoint
                cursor = piece.end_date + pd.Timedelta(days=1)
                computed_segments += 1

        if not assembled:
            raise ValueError(
                "market data contains no observations in the requested range"
            )
        result = _combine_simulation_results(
            assembled,
            cache_key=cache_key,
        )
        source_records = _merged_source_data_records(assembled)
        if source_records:
            result = IndexSimulationResult(
                daily=result.daily,
                holdings=result.holdings,
                rebalances=result.rebalances,
                asset_returns=result.asset_returns,
                metadata={
                    **result.metadata,
                    "source_data_records": source_records,
                },
                checkpoint=result.checkpoint,
                weight_snapshots=result.weight_snapshots,
            )
        result = self._slice_replayed_state(
            result,
            calculation_start=calculation_start,
        )
        result = self._reconstruct_levels_from_returns(result)
        result = self._stabilize_reusable_levels(result)
        result = self._attach_market_data_lineage(
            IndexSimulationResult(
                daily=result.daily,
                holdings=result.holdings,
                rebalances=result.rebalances,
                asset_returns=result.asset_returns,
                metadata={
                    **result.metadata,
                    "cache_key": cache_key,
                    "cache_source": (
                        "workspace" if computed_segments == 0 else "computed"
                    ),
                    "segment_cache_source": (
                        "workspace_immutable_segments"
                        if computed_segments == 0
                        else "computed_immutable_segments"
                    ),
                    "start_date": str(self.start_date.date()),
                    "end_date": str(self.end_date.date()),
                    "segment_schema_version": 2,
                    "immutable_segments_reused": reused_segments,
                    "immutable_segments_computed": computed_segments,
                    "state_replayed_from": str(calculation_start.date()),
                    **(
                        {"segment_reuse": "extended_prefix"}
                        if (
                            reused_segments > 0
                            and computed_segments > 0
                            and not reused_after_computation
                        )
                        else {}
                    ),
                },
                checkpoint=result.checkpoint,
                weight_snapshots=result.weight_snapshots,
            )
        )
        return result

    def _immutable_segment_plan(
        self,
    ) -> list[_ImmutableSimulationSegment]:
        effective_dates = sorted(
            pd.Timestamp(value).normalize() for value in self.backtest_result.reviews
        )
        active = [value for value in effective_dates if value <= self.start_date]
        if not active:
            return []
        # One prior target is replayed when available so a mid-period request
        # uses the real state entering its nearest effective date. This is
        # required for CLOSE-phase returns and produces correct turnover for
        # either phase without treating ``start_date`` as a new rebalance.
        seed = active[-2] if len(active) > 1 else active[-1]
        selected = [
            value for value in effective_dates if seed <= value <= self.end_date
        ]
        if not selected:
            return []
        checksums = {
            value: self._review_target_checksum(value) for value in effective_dates
        }
        positions = {value: position for position, value in enumerate(effective_dates)}
        result: list[_ImmutableSimulationSegment] = []
        for effective_date in selected:
            position = positions[effective_date]
            previous_effective = effective_dates[position - 1] if position > 0 else None
            next_effective = (
                effective_dates[position + 1]
                if position + 1 < len(effective_dates)
                else None
            )
            previous_checksum = (
                None if previous_effective is None else checksums[previous_effective]
            )
            if next_effective is not None and next_effective <= self.end_date:
                result.append(
                    _ImmutableSimulationSegment(
                        start_date=effective_date,
                        end_date=next_effective - pd.Timedelta(days=1),
                        effective_date=effective_date,
                        next_effective_date=next_effective,
                        kind="closed_effective_period",
                        target_checksum=checksums[effective_date],
                        previous_target_checksum=previous_checksum,
                    )
                )
                continue
            interval_end = (
                self.end_date
                if next_effective is None
                else min(
                    self.end_date,
                    next_effective - pd.Timedelta(days=1),
                )
            )
            for month_start, month_end in _calendar_month_partitions(
                effective_date,
                interval_end,
            ):
                result.append(
                    _ImmutableSimulationSegment(
                        start_date=month_start,
                        end_date=month_end,
                        effective_date=effective_date,
                        next_effective_date=next_effective,
                        kind="open_tail_month",
                        target_checksum=checksums[effective_date],
                        previous_target_checksum=previous_checksum,
                    )
                )
        return result

    def _review_target_checksum(
        self,
        effective_date: pd.Timestamp,
    ) -> str:
        context = self.backtest_result.reviews[effective_date]
        required = ["index_weight", "benchmark_weight"]
        missing = [name for name in required if name not in context.cons]
        if missing:
            raise ValueError(f"review target is missing columns: {missing}")
        frame = context.cons.loc[:, required].reset_index()
        if "instrument_id" not in frame:
            frame = frame.rename(columns={frame.columns[0]: "instrument_id"})
        return self.identity_service.digest(
            {
                "effective_date": effective_date,
                "reference_date": pd.Timestamp(context.reference_date).normalize(),
                "weights": self.identity_service.dataframe_digest(
                    frame,
                    sort_by=["instrument_id"],
                ),
            }
        )

    def _calculate_immutable_segment(
        self,
        segment: _ImmutableSimulationSegment,
        cache_key: str,
        *,
        opening_checkpoint: SimulationCheckpoint | None,
    ) -> tuple[IndexSimulationResult | None, str | None]:
        request_start = (
            segment.start_date
            if opening_checkpoint is not None
            else self._market_data_request_start(segment.start_date)
        )
        if self.streaming:
            partitions = [
                self._load_market_data(partition_start, partition_end)
                for partition_start, partition_end in _calendar_month_partitions(
                    request_start,
                    segment.end_date,
                )
            ]
            raw = (
                pd.concat(
                    [frame for frame in partitions if not frame.empty],
                    ignore_index=True,
                )
                if any(not frame.empty for frame in partitions)
                else pd.DataFrame()
            )
        else:
            raw = self._load_market_data(request_start, segment.end_date)
        market_data = validate_daily_market_data(raw)
        partition_digest = (
            None
            if market_data.empty
            else self.identity_service.dataframe_digest(
                market_data,
                sort_by=["business_date", "instrument_id"],
            )
        )
        prior_observations = (
            None
            if opening_checkpoint is not None
            else _latest_observations_before(
                market_data,
                segment.start_date,
            )
        )
        simulation_data = market_data.loc[
            market_data["business_date"].between(
                segment.start_date,
                segment.end_date,
            )
        ].copy()
        self._validate_business_date_coverage(
            simulation_data,
            start_date=segment.start_date,
            end_date=segment.end_date,
        )
        if simulation_data.empty:
            return None, partition_digest
        local_checkpoint = (
            None
            if opening_checkpoint is None
            else _checkpoint_with_levels(
                opening_checkpoint,
                {
                    name: float(self.params.base_value)
                    for name in opening_checkpoint.levels
                },
            )
        )
        return (
            self._simulate(
                simulation_data,
                cache_key,
                initial_checkpoint=local_checkpoint,
                initial_previous_observations=prior_observations,
            ),
            partition_digest,
        )

    def _scale_segment_from_opening_state(
        self,
        local: IndexSimulationResult,
        opening_checkpoint: SimulationCheckpoint | None,
    ) -> IndexSimulationResult:
        desired_opening = (
            {
                name: float(self.params.base_value)
                for name in (
                    column for column in local.daily if column.endswith("_level")
                )
            }
            if opening_checkpoint is None
            else dict(opening_checkpoint.levels)
        )
        return _rescale_simulation_levels(
            local,
            desired_opening=desired_opening,
        )

    def _with_segment_metadata(
        self,
        result: IndexSimulationResult,
        *,
        segment: _ImmutableSimulationSegment,
        cache_key: str,
        partition_digest: str | None,
        cache_source: str,
        opening_checkpoint: SimulationCheckpoint | None,
    ) -> IndexSimulationResult:
        source_lineage_digest = self._segment_market_data_lineage_digest(
            segment,
            opening_checkpoint=opening_checkpoint,
        )
        metadata = {
            **result.metadata,
            "cache_key": cache_key,
            "cache_source": cache_source,
            "segment_schema_version": 2,
            "segment_kind": segment.kind,
            "segment_start_date": str(segment.start_date.date()),
            "segment_end_date": str(segment.end_date.date()),
            "segment_end_exclusive": str(
                (segment.end_date + pd.Timedelta(days=1)).date()
            ),
            "effective_date": str(segment.effective_date.date()),
            "review_target_checksum": segment.target_checksum,
            "previous_review_target_checksum": (segment.previous_target_checksum),
            "market_data_partition_digest": partition_digest,
            "market_data_partition_lineage_digest": (source_lineage_digest),
            "business_day_partition_digest": (
                self._business_day_segment_digest(segment)
            ),
            "opening_state_digest": self._checkpoint_state_digest(opening_checkpoint),
        }
        return IndexSimulationResult(
            daily=result.daily,
            holdings=result.holdings,
            rebalances=result.rebalances,
            asset_returns=result.asset_returns,
            metadata=metadata,
            checkpoint=result.checkpoint,
            weight_snapshots=result.weight_snapshots,
        )

    def _record_immutable_segment(
        self,
        segment: _ImmutableSimulationSegment,
        *,
        cache_key: str,
        partition_digest: str | None,
        opening_checkpoint: SimulationCheckpoint | None,
    ) -> None:
        if self.cache_policy is CachePolicy.READ_ONLY:
            return
        save_json = getattr(self.workspace, "save_json", None)
        if not callable(save_json):
            return
        namespace = self._immutable_segment_namespace()
        with self._immutable_segment_catalog_lock(namespace):
            coverage = self._load_immutable_segment_catalog()
            record = {
                "start_date": str(segment.start_date.date()),
                "end_date": str(segment.end_date.date()),
                "effective_date": str(segment.effective_date.date()),
                "target_checksum": segment.target_checksum,
                "previous_target_checksum": (segment.previous_target_checksum),
                "opening_state_digest": self._checkpoint_state_digest(
                    opening_checkpoint
                ),
                "market_data_partition_lineage_digest": (
                    self._segment_market_data_lineage_digest(
                        segment,
                        opening_checkpoint=opening_checkpoint,
                    )
                ),
                "business_day_partition_digest": (
                    self._business_day_segment_digest(segment)
                ),
                "cache_key": cache_key,
                "kind": segment.kind,
                "partition_digest": partition_digest,
            }
            unique = {
                (
                    item.get("start_date"),
                    item.get("end_date"),
                    item.get("effective_date"),
                    item.get("target_checksum"),
                    item.get("previous_target_checksum"),
                    item.get("opening_state_digest"),
                    item.get("cache_key"),
                ): dict(item)
                for item in [*coverage, record]
            }
            ordered = sorted(
                unique.values(),
                key=lambda item: (
                    str(item.get("start_date")),
                    str(item.get("end_date")),
                    str(item.get("cache_key")),
                ),
            )
            save_json(
                "simulation_segments",
                namespace,
                "coverage",
                {
                    "schema_version": 2,
                    "segments": ordered,
                },
            )

    @contextmanager
    def _immutable_segment_catalog_lock(
        self,
        namespace: str,
    ) -> Iterator[None]:
        """Serialize one segment coverage merge across local processes."""

        lock = getattr(self.workspace, "simulation_catalog_lock", None)
        if not callable(lock):
            yield
            return
        with lock(namespace):
            yield


__all__ = ["ImmutableSegmentExecutionMixin"]
