"""Identity, lineage, and catalog selection for immutable segments."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

import pandas as pd

from .assembly import (
    configuration_value as _configuration_value,
    simulation_params_payload as _simulation_params_payload,
)
from .cache_contracts import SimulationCacheMissError as CacheMissError
from .models import SimulationCheckpoint
from .segments import ImmutableSimulationSegment as _ImmutableSimulationSegment


class ImmutableSegmentIdentityMixin:
    """Build verified identities and select reusable segment artifacts."""
    def _immutable_segment_base_identity(self) -> dict[str, Any]:
        parameters = _simulation_params_payload(self.params)
        parameters.pop("base_value", None)
        verified_partition_lineage = bool(
            self._market_data_lineage_records()
        )
        return {
            "schema_version": 2,
            "kind": "immutable_effective_date_simulation_segments",
            "simulator": self.identity_service.source_identity(type(self)),
            "runtime": self.identity_service.runtime_identity(),
            "index_drift": self._drift_cache_identity(
                self.params.resolved_index_drift
            ),
            "benchmark_drift": self._drift_cache_identity(
                self.params.resolved_benchmark_drift
            ),
            "provider_name": self.market_data_provider_name.strip().lower(),
            "provider_parameters": self.identity_service.safe_parameters(
                self.provider_parameters
            ),
            "market_data_identity_mode": (
                "verified_partition_content"
                if verified_partition_lineage
                else "provider_snapshot"
            ),
            "data_revision": (
                None
                if verified_partition_lineage
                else _configuration_value(self.data_revision)
            ),
            "simulation_parameters": parameters,
        }

    def _drift_cache_identity(self, strategy: object) -> dict[str, Any]:
        return {
            "implementation": self.identity_service.source_identity(strategy),
            "configuration": _configuration_value(strategy),
        }

    def _immutable_segment_namespace(self) -> str:
        return self.identity_service.digest(self._immutable_segment_base_identity())

    def _business_day_digest(self) -> str | None:
        if self.business_days is None:
            return None
        return self.identity_service.digest(
            [
                str(pd.Timestamp(value).normalize().date())
                for value in self.business_days
            ]
        )

    def _business_day_segment_digest(
        self,
        segment: _ImmutableSimulationSegment,
    ) -> str | None:
        if self.business_days is None:
            return None
        selected = self.business_days[
            (self.business_days >= segment.start_date)
            & (self.business_days <= segment.end_date)
        ]
        return self.identity_service.digest(
            [
                str(pd.Timestamp(value).normalize().date())
                for value in selected
            ]
        )

    def _market_data_lineage_records(
        self,
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            record
            for record in self.market_data_lineage
            if record.get("input_type") == "source_daily_market_data"
        )

    def _segment_market_data_lineage_digest(
        self,
        segment: _ImmutableSimulationSegment,
        *,
        opening_checkpoint: SimulationCheckpoint | None,
    ) -> str:
        records = self._market_data_lineage_records()
        if not records:
            return self.identity_service.digest(
                {
                    "identity_mode": "provider_snapshot",
                    "data_revision": _configuration_value(
                        self.data_revision
                    ),
                }
            )

        request_start = (
            segment.start_date
            if opening_checkpoint is not None
            else self._market_data_request_start(segment.start_date)
        )
        selected: list[dict[str, Any]] = []
        coverage: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        stable_fields = (
            "provider_name",
            "capability",
            "request_digest",
            "content_digest",
            "snapshot_digest",
            "snapshot_protocol",
            "rows",
            "start_date",
            "end_date",
            "instrument_set_digest",
        )
        for record in records:
            try:
                start = pd.Timestamp(record["start_date"]).normalize()
                end = pd.Timestamp(record["end_date"]).normalize()
            except (KeyError, TypeError, ValueError):
                continue
            if end < request_start or start > segment.end_date:
                continue
            selected.append(
                {
                    name: record.get(name)
                    for name in stable_fields
                    if record.get(name) is not None
                }
            )
            coverage.append(
                (
                    max(start, request_start),
                    min(end, segment.end_date),
                )
            )

        cursor = request_start
        for start, end in sorted(coverage):
            if start > cursor:
                break
            if end >= cursor:
                cursor = max(cursor, end + pd.Timedelta(days=1))
            if cursor > segment.end_date:
                break
        if cursor <= segment.end_date:
            raise ValueError(
                "verified market-data lineage does not cover simulation "
                f"segment {request_start.date()} through "
                f"{segment.end_date.date()}"
            )

        unique = {
            canonical: record
            for record in selected
            for canonical in (
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ),
            )
        }
        return self.identity_service.digest(
            {
                "identity_mode": "verified_partition_content",
                "partitions": [
                    unique[key] for key in sorted(unique)
                ],
            }
        )

    def _checkpoint_state_digest(
        self,
        checkpoint: SimulationCheckpoint | None,
    ) -> str | None:
        if checkpoint is None:
            return None

        def series_digest(series: pd.Series, name: str) -> str:
            frame = series.rename(name).rename_axis(
                "instrument_id"
            ).reset_index()
            return self.identity_service.dataframe_digest(
                frame,
                sort_by=["instrument_id"],
            )

        observations = checkpoint.previous_observations.reset_index()
        observation_sort = (
            ["instrument_id"]
            if "instrument_id" in observations
            else None
        )
        return self.identity_service.digest(
            {
                "business_date": checkpoint.business_date,
                "index_weights": series_digest(
                    checkpoint.index_weights,
                    "index_weight",
                ),
                "benchmark_weights": series_digest(
                    checkpoint.benchmark_weights,
                    "benchmark_weight",
                ),
                "previous_observations": self.identity_service.dataframe_digest(
                    observations,
                    sort_by=observation_sort,
                ),
            }
        )

    def _immutable_segment_key(
        self,
        segment: _ImmutableSimulationSegment,
        *,
        opening_checkpoint: SimulationCheckpoint | None,
    ) -> str:
        # ``kind`` and ``next_effective_date`` are intentionally excluded.
        # A month written while the tail was open remains valid if a later
        # review subsequently closes that holding period.
        return self.identity_service.digest(
            {
                "namespace": self._immutable_segment_namespace(),
                "start_date": segment.start_date,
                "end_date": segment.end_date,
                "effective_date": segment.effective_date,
                "target_checksum": segment.target_checksum,
                "previous_target_checksum": (
                    segment.previous_target_checksum
                ),
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
            }
        )

    def _load_immutable_segment_catalog(
        self,
    ) -> list[dict[str, Any]]:
        load_json = getattr(self.workspace, "load_json", None)
        if not callable(load_json):
            return []
        try:
            value = load_json(
                "simulation_segments",
                self._immutable_segment_namespace(),
                "coverage",
            )
        except CacheMissError:
            return []
        records = value.get("segments", []) if isinstance(value, dict) else []
        return [
            dict(item)
            for item in records
            if isinstance(item, dict)
            and {
                "start_date",
                "end_date",
                "effective_date",
                "target_checksum",
                "cache_key",
                "opening_state_digest",
            }.issubset(item)
        ]

    def _best_immutable_segment_candidate(
        self,
        catalog: list[dict[str, Any]],
        *,
        desired: _ImmutableSimulationSegment,
        cursor: pd.Timestamp,
        opening_checkpoint: SimulationCheckpoint | None,
        rejected_keys: set[str],
    ) -> dict[str, Any] | None:
        candidates = []
        for item in catalog:
            cache_key = str(item.get("cache_key", ""))
            if cache_key in rejected_keys:
                continue
            try:
                start = pd.Timestamp(item["start_date"]).normalize()
                end = pd.Timestamp(item["end_date"]).normalize()
                effective = pd.Timestamp(
                    item["effective_date"]
                ).normalize()
            except (KeyError, TypeError, ValueError):
                continue
            if (
                start != cursor
                or end > desired.end_date
                or effective != desired.effective_date
                or item.get("target_checksum")
                != desired.target_checksum
                or item.get("previous_target_checksum")
                != desired.previous_target_checksum
            ):
                continue
            candidate_segment = _ImmutableSimulationSegment(
                start_date=start,
                end_date=end,
                effective_date=effective,
                next_effective_date=desired.next_effective_date,
                kind=str(item.get("kind", desired.kind)),
                target_checksum=desired.target_checksum,
                previous_target_checksum=(
                    desired.previous_target_checksum
                ),
            )
            expected_state_digest = self._checkpoint_state_digest(
                opening_checkpoint
            )
            if item.get("opening_state_digest") != expected_state_digest:
                continue
            if cache_key != self._immutable_segment_key(
                candidate_segment,
                opening_checkpoint=opening_checkpoint,
            ):
                continue
            candidates.append((end, item))
        if not candidates:
            return None
        return max(candidates, key=lambda value: value[0])[1]


__all__ = ["ImmutableSegmentIdentityMixin"]
