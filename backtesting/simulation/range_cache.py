"""Range reuse and extension for coarse simulation cache artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

import pandas as pd

from icapa.data_sources.contracts import validate_daily_market_data

from .assembly import (
    checkpoint_from_result as _checkpoint_from_result,
    concat_frames as _concat_frames,
    concat_rebalances as _concat_rebalances,
)
from .cache_contracts import (
    SimulationCacheMissError as CacheMissError,
    SimulationCachePolicy as CachePolicy,
)
from .enums import WeightSnapshotMode
from .models import IndexSimulationResult


class SimulationRangeCacheMixin:
    """Reuse, slice, and extend cached simulation date ranges."""
    def _reuse_segment(self, cache_key: str) -> IndexSimulationResult | None:
        coverage = self._load_segment_coverage()
        same_start = [
            item
            for item in coverage
            if pd.Timestamp(item["start_date"]).normalize() == self.start_date
        ]
        containing = sorted(
            (
                item
                for item in same_start
                if pd.Timestamp(item["end_date"]).normalize() >= self.end_date
            ),
            key=lambda item: pd.Timestamp(item["end_date"]),
        )
        for item in containing:
            candidate = self._load_cached(
                str(item["cache_key"]),
                allow_miss=True,
            )
            if candidate is None:
                continue
            sliced = self._slice_cached_segment(candidate, cache_key)
            if sliced is not None:
                self._save_cached(cache_key, sliced)
                self._record_segment(cache_key)
                return sliced

        if self.cache_policy is CachePolicy.READ_ONLY:
            return None
        prefixes = sorted(
            (
                item
                for item in same_start
                if pd.Timestamp(item["end_date"]).normalize() < self.end_date
            ),
            key=lambda item: pd.Timestamp(item["end_date"]),
            reverse=True,
        )
        for item in prefixes:
            prefix = self._load_cached(
                str(item["cache_key"]),
                allow_miss=True,
            )
            if prefix is None or prefix.checkpoint is None:
                continue
            extended = self._extend_cached_segment(prefix, cache_key)
            if extended is None:
                continue
            self._save_cached(cache_key, extended)
            self._record_segment(cache_key)
            return extended
        return None

    def _slice_cached_segment(
        self,
        result: IndexSimulationResult,
        cache_key: str,
    ) -> IndexSimulationResult | None:
        selected_daily = result.daily.loc[result.daily.index <= self.end_date]
        if selected_daily.empty:
            return None
        final_date = pd.Timestamp(selected_daily.index[-1]).normalize()
        checkpoint = _checkpoint_from_result(result, final_date)
        if (
            checkpoint is None
            and self.params.materialization.weight_snapshots
            is WeightSnapshotMode.DAILY
        ):
            return None
        holdings = result.holdings
        if not holdings.empty:
            holding_dates = holdings.index.get_level_values("business_date")
            holdings = holdings.loc[holding_dates <= final_date]
        rebalances = result.rebalances
        if not rebalances.empty:
            rebalances = rebalances.loc[
                pd.to_datetime(rebalances["applied_business_date"])
                <= final_date
            ]
        assets = result.asset_returns
        if not assets.empty:
            asset_dates = assets.index.get_level_values("business_date")
            assets = assets.loc[asset_dates <= final_date]
        weight_snapshots = result.weight_snapshots
        if not weight_snapshots.empty:
            snapshot_dates = weight_snapshots.index.get_level_values(
                "applied_business_date"
            )
            weight_snapshots = weight_snapshots.loc[
                snapshot_dates <= final_date
            ]
        return IndexSimulationResult(
            daily=selected_daily.copy(),
            holdings=holdings.copy(),
            rebalances=rebalances.copy(),
            asset_returns=assets.copy(),
            metadata={
                **result.metadata,
                "cache_key": cache_key,
                "cache_source": "workspace_segment",
                "start_date": str(self.start_date.date()),
                "end_date": str(self.end_date.date()),
                "segment_reuse": "containing_prefix",
            },
            checkpoint=checkpoint,
            weight_snapshots=weight_snapshots.copy(),
        )

    def _extend_cached_segment(
        self,
        prefix: IndexSimulationResult,
        cache_key: str,
    ) -> IndexSimulationResult | None:
        checkpoint = prefix.checkpoint
        if checkpoint is None:
            return None
        request_start = checkpoint.business_date + pd.Timedelta(days=1)
        if self.streaming:
            tail = self._simulate_streaming(
                cache_key,
                calculation_start=request_start,
                load_start=request_start,
                initial_checkpoint=checkpoint,
            )
        else:
            raw = self._load_market_data(request_start, self.end_date)
            market_data = validate_daily_market_data(raw)
            market_data = market_data.loc[
                market_data["business_date"].between(
                    request_start,
                    self.end_date,
                )
            ].copy()
            if not market_data.empty:
                self._validate_business_date_coverage(
                    market_data,
                    start_date=request_start,
                    end_date=self.end_date,
                )
                tail = self._simulate(
                    market_data,
                    cache_key,
                    initial_checkpoint=checkpoint,
                )
            else:
                tail = None
        if tail is None:
            return self._attach_market_data_lineage(IndexSimulationResult(
                daily=prefix.daily.copy(),
                holdings=prefix.holdings.copy(),
                rebalances=prefix.rebalances.copy(),
                asset_returns=prefix.asset_returns.copy(),
                metadata={
                    **prefix.metadata,
                    "cache_key": cache_key,
                    "cache_source": "workspace_segment",
                    "end_date": str(self.end_date.date()),
                    "segment_reuse": "calendar_only_extension",
                },
                checkpoint=checkpoint,
                weight_snapshots=prefix.weight_snapshots.copy(),
            ))
        extended = self._attach_market_data_lineage(IndexSimulationResult(
            daily=_concat_frames(prefix.daily, tail.daily),
            holdings=_concat_frames(prefix.holdings, tail.holdings),
            rebalances=_concat_rebalances(
                prefix.rebalances,
                tail.rebalances,
            ),
            asset_returns=_concat_frames(
                prefix.asset_returns,
                tail.asset_returns,
            ),
            metadata={
                **tail.metadata,
                "cache_key": cache_key,
                "cache_source": "computed_segment_extension",
                "start_date": str(self.start_date.date()),
                "end_date": str(self.end_date.date()),
                "segment_reuse": "extended_prefix",
                "reused_through": str(checkpoint.business_date.date()),
            },
            checkpoint=tail.checkpoint,
            weight_snapshots=_concat_frames(
                prefix.weight_snapshots,
                tail.weight_snapshots,
            ),
        ))
        return self._stabilize_reusable_levels(extended)

    def _attach_market_data_lineage(
        self,
        result: IndexSimulationResult,
    ) -> IndexSimulationResult:
        """Persist sanitized source-partition identities with cached results."""

        owner = getattr(self.market_data_loader, "__self__", None)
        current = (
            *self.market_data_lineage,
            *(tuple(getattr(owner, "records", ()) or ())),
        )
        previous = tuple(
            result.metadata.get("source_data_records", ()) or ()
        )
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in (*previous, *current):
            if not isinstance(record, dict):
                record = dict(record)
            marker = json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            if marker in seen:
                continue
            seen.add(marker)
            records.append(dict(record))
        if not records:
            return result
        return IndexSimulationResult(
            daily=result.daily,
            holdings=result.holdings,
            rebalances=result.rebalances,
            asset_returns=result.asset_returns,
            metadata={
                **result.metadata,
                "source_data_records": records,
            },
            checkpoint=result.checkpoint,
            weight_snapshots=result.weight_snapshots,
        )

    def _segment_identity(self) -> str:
        payload = {
            "schema": 1,
            "cache_identity_without_range": self._cache_identity_without_range(),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def _cache_identity_without_range(self) -> dict[str, Any]:
        cache_key_payload = self._cache_key_payload()
        cache_key_payload.pop("start_date", None)
        cache_key_payload.pop("end_date", None)
        return cache_key_payload

    def _load_segment_coverage(self) -> list[dict[str, str]]:
        if self.workspace is None or self.data_revision is None:
            return []
        load_json = getattr(self.workspace, "load_json", None)
        if not callable(load_json):
            return []
        try:
            value = load_json(
                "simulation_segments",
                self._segment_identity(),
                "coverage",
            )
        except CacheMissError:
            return []
        segments = value.get("segments", []) if isinstance(value, dict) else []
        return [
            dict(item)
            for item in segments
            if isinstance(item, dict)
            and {"start_date", "end_date", "cache_key"}.issubset(item)
        ]

    def _record_segment(self, cache_key: str) -> None:
        if (
            not self.segmented_cache
            or self.workspace is None
            or self.data_revision is None
            or self.cache_policy is CachePolicy.READ_ONLY
        ):
            return
        save_json = getattr(self.workspace, "save_json", None)
        if not callable(save_json):
            return
        coverage = self._load_segment_coverage()
        record = {
            "start_date": str(self.start_date.date()),
            "end_date": str(self.end_date.date()),
            "cache_key": cache_key,
        }
        unique = {
            (
                item["start_date"],
                item["end_date"],
                item["cache_key"],
            ): item
            for item in [*coverage, record]
        }
        ordered = sorted(
            unique.values(),
            key=lambda item: (
                item["start_date"],
                item["end_date"],
                item["cache_key"],
            ),
        )
        save_json(
            "simulation_segments",
            self._segment_identity(),
            "coverage",
            {"schema_version": 1, "segments": ordered},
        )


__all__ = ["SimulationRangeCacheMixin"]
