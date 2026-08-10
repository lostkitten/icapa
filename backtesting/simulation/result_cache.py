"""Exact simulation cache identity, I/O, and level rebasing."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

import pandas as pd

from .cache_contracts import (
    SimulationCacheMissError as CacheMissError,
    SimulationCachePolicy as CachePolicy,
)

from .assembly import (
    checkpoint_frame as _checkpoint_frame,
    checkpoint_with_levels as _checkpoint_with_levels,
    decode_checkpoint as _decode_checkpoint,
    legacy_simulation_params_payload as _legacy_simulation_params_payload,
    simulation_params_payload as _simulation_params_payload,
)
from .enums import RebalancePhase, WeightSnapshotMode
from .models import IndexSimulationResult, SimulationCheckpoint
from .rebalance import (
    rebalance_weight_snapshot_frame as _rebalance_weight_snapshot_frame,
)


class SimulationResultCacheMixin:
    """Persist, load, and rebase exact simulation artifacts."""
    def _cache_key(self) -> str:
        encoded = json.dumps(
            self._cache_key_payload(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def _cache_key_payload(self) -> dict[str, Any]:
        result_metadata = getattr(self.backtest_result, "metadata", None)
        fingerprint = getattr(result_metadata, "fingerprint", None)
        if fingerprint:
            review_metadata = getattr(result_metadata, "reviews", {})
            weight_identity: Any = {
                "backtest_fingerprint": fingerprint,
                "review_artifacts": [
                    {
                        "effective_date": str(
                            pd.Timestamp(effective_date).date()
                        ),
                        "checksum": getattr(review, "artifact_checksum", None),
                    }
                    for effective_date, review in sorted(review_metadata.items())
                ],
            }
        else:
            weight_frame = self.backtest_result.weights.reset_index().copy()
            weight_frame["effective_date"] = pd.to_datetime(
                weight_frame["effective_date"]
            ).dt.strftime("%Y-%m-%d")
            weight_frame = weight_frame.sort_values(
                ["effective_date", "instrument_id"],
                key=lambda values: values.map(str),
            )
            weight_identity = weight_frame.to_dict(orient="records")
        legacy_identity = self._uses_legacy_cache_identity()
        parameter_identity = (
            _legacy_simulation_params_payload(self.params)
            if legacy_identity
            else _simulation_params_payload(self.params)
        )
        if not legacy_identity:
            parameter_identity.pop("base_value", None)
        payload = {
            "schema": 1 if legacy_identity else 3,
            "weights": weight_identity,
            "provider": self.market_data_provider_name,
            "provider_parameters": self.provider_parameters,
            "data_revision": self.data_revision,
            "start_date": str(self.start_date.date()),
            "end_date": str(self.end_date.date()),
            "params": parameter_identity,
        }
        if not legacy_identity:
            payload["business_day_digest"] = self._business_day_digest()
            payload["calculation_identity"] = {
                "simulator": self.identity_service.source_identity(type(self)),
                "runtime": self.identity_service.runtime_identity(),
                "index_drift": self._drift_cache_identity(
                    self.params.resolved_index_drift
                ),
                "benchmark_drift": self._drift_cache_identity(
                    self.params.resolved_benchmark_drift
                ),
            }
        return payload

    def _uses_legacy_cache_identity(self) -> bool:
        materialization = self.params.materialization
        return (
            self.params.index_drift is None
            and self.params.benchmark_drift is None
            and self.params.rebalance_phase is RebalancePhase.OPEN
            and materialization.weight_snapshots is WeightSnapshotMode.DAILY
            and materialization.include_asset_returns
        )

    def _load_cached(
        self,
        cache_key: str,
        *,
        allow_miss: bool = False,
        after_write: bool = False,
    ) -> IndexSimulationResult | None:
        if (
            self.workspace is None
            or self.data_revision is None
            or (
                self.cache_policy is CachePolicy.REFRESH
                and not after_write
            )
        ):
            return None
        load_frame = getattr(self.workspace, "load_frame", None)
        load_json = getattr(self.workspace, "load_json", None)
        if not callable(load_frame) or not callable(load_json):
            return None
        try:
            daily = load_frame("simulation", cache_key, "daily")
            holdings = load_frame("simulation", cache_key, "holdings")
            rebalances = load_frame("simulation", cache_key, "rebalances")
            asset_returns = load_frame(
                "simulation",
                cache_key,
                "asset_returns",
            )
            metadata = load_json("simulation", cache_key, "metadata")
        except CacheMissError:
            if self.cache_policy is CachePolicy.READ_ONLY and not allow_miss:
                raise
            return None
        resolved_cache_key = cache_key
        try:
            weight_snapshots = load_frame(
                "simulation",
                resolved_cache_key,
                "weight_snapshots",
            )
        except CacheMissError:
            weight_snapshots = _rebalance_weight_snapshot_frame([])
        checkpoint = self._load_checkpoint(resolved_cache_key)
        return IndexSimulationResult(
            daily=daily,
            holdings=holdings,
            rebalances=rebalances,
            asset_returns=asset_returns,
            metadata={**metadata, "cache_source": "workspace"},
            checkpoint=checkpoint,
            weight_snapshots=weight_snapshots,
        ) if self._uses_legacy_cache_identity() else self._rebase_cached_result(
            daily=daily,
            holdings=holdings,
            rebalances=rebalances,
            asset_returns=asset_returns,
            weight_snapshots=weight_snapshots,
            metadata=metadata,
            checkpoint=checkpoint,
        )

    def _save_cached(
        self,
        cache_key: str,
        result: IndexSimulationResult,
    ) -> None:
        if (
            self.workspace is None
            or self.data_revision is None
            or self.cache_policy is CachePolicy.READ_ONLY
        ):
            return
        save_frame = getattr(self.workspace, "save_frame", None)
        save_json = getattr(self.workspace, "save_json", None)
        if not callable(save_frame) or not callable(save_json):
            return
        cached_daily = result.daily
        cached_checkpoint = result.checkpoint
        cached_metadata = {**result.metadata, "cache_source": "computed"}
        if not self._uses_legacy_cache_identity():
            cached_daily, cached_checkpoint, cached_metadata = (
                self._return_factor_cache_payload(result)
            )
        save_frame("simulation", cache_key, "daily", cached_daily)
        save_frame("simulation", cache_key, "holdings", result.holdings)
        save_frame("simulation", cache_key, "rebalances", result.rebalances)
        save_frame("simulation", cache_key, "asset_returns", result.asset_returns)
        save_frame(
            "simulation",
            cache_key,
            "weight_snapshots",
            result.weight_snapshots,
        )
        if cached_checkpoint is not None:
            save_frame(
                "simulation",
                cache_key,
                "checkpoint",
                _checkpoint_frame(cached_checkpoint),
            )
            save_json(
                "simulation",
                cache_key,
                "checkpoint_metadata",
                {
                    "business_date": str(cached_checkpoint.business_date.date()),
                    "levels": dict(cached_checkpoint.levels),
                },
            )
        save_json(
            "simulation",
            cache_key,
            "metadata",
            cached_metadata,
        )

    def _return_factor_cache_payload(
        self,
        result: IndexSimulationResult,
    ) -> tuple[
        pd.DataFrame,
        SimulationCheckpoint | None,
        dict[str, Any],
    ]:
        """Remove absolute base values from reusable simulation state."""

        base_value = float(self.params.base_value)
        daily = result.daily.copy()
        level_columns = [
            column for column in daily.columns if column.endswith("_level")
        ]
        if level_columns:
            daily.loc[:, level_columns] = (
                daily.loc[:, level_columns] / base_value
            )
        checkpoint = result.checkpoint
        if checkpoint is not None:
            checkpoint = SimulationCheckpoint(
                business_date=checkpoint.business_date,
                index_weights=checkpoint.index_weights,
                benchmark_weights=checkpoint.benchmark_weights,
                levels={
                    name: value / base_value
                    for name, value in checkpoint.levels.items()
                },
                previous_observations=checkpoint.previous_observations,
            )
        simulation_params = dict(
            result.metadata.get("simulation_params", {})
        )
        simulation_params["base_value"] = 1.0
        metadata = {
            **result.metadata,
            "cache_source": "computed",
            "cached_level_representation": "return_factor",
            "simulation_params": simulation_params,
        }
        return daily, checkpoint, metadata

    def _load_checkpoint(self, cache_key: str) -> SimulationCheckpoint | None:
        load_frame = getattr(self.workspace, "load_frame", None)
        load_json = getattr(self.workspace, "load_json", None)
        if not callable(load_frame) or not callable(load_json):
            return None
        try:
            frame = load_frame("simulation", cache_key, "checkpoint")
            metadata = load_json(
                "simulation",
                cache_key,
                "checkpoint_metadata",
            )
        except CacheMissError:
            return None
        return _decode_checkpoint(frame, metadata)

    def _rebase_cached_result(
        self,
        *,
        daily: pd.DataFrame,
        holdings: pd.DataFrame,
        rebalances: pd.DataFrame,
        asset_returns: pd.DataFrame,
        weight_snapshots: pd.DataFrame,
        metadata: dict[str, Any],
        checkpoint: SimulationCheckpoint | None,
    ) -> IndexSimulationResult:
        """Apply presentation-only base value after loading reusable return factors."""

        simulation_params = metadata.get("simulation_params", {})
        stored_base = float(simulation_params.get("base_value", self.params.base_value))
        requested_base = float(self.params.base_value)
        ratio = requested_base / stored_base
        rebased_daily = daily.copy()
        level_columns = [
            column for column in rebased_daily.columns if column.endswith("_level")
        ]
        if ratio != 1.0 and level_columns:
            rebased_daily.loc[:, level_columns] = (
                rebased_daily.loc[:, level_columns] * ratio
            )
        rebased_checkpoint = checkpoint
        if checkpoint is not None and ratio != 1.0:
            rebased_checkpoint = SimulationCheckpoint(
                business_date=checkpoint.business_date,
                index_weights=checkpoint.index_weights,
                benchmark_weights=checkpoint.benchmark_weights,
                levels={
                    name: value * ratio
                    for name, value in checkpoint.levels.items()
                },
                previous_observations=checkpoint.previous_observations,
            )
        result = IndexSimulationResult(
            daily=rebased_daily,
            holdings=holdings,
            rebalances=rebalances,
            asset_returns=asset_returns,
            metadata={
                **metadata,
                "cache_source": "workspace",
                "simulation_params": {
                    **simulation_params,
                    "base_value": requested_base,
                },
            },
            checkpoint=rebased_checkpoint,
            weight_snapshots=weight_snapshots,
        )
        return self._stabilize_reusable_levels(result)

    def _stabilize_reusable_levels(
        self,
        result: IndexSimulationResult,
    ) -> IndexSimulationResult:
        """Canonicalize presentation levels after return-factor assembly."""

        if self._uses_legacy_cache_identity():
            return result
        daily = result.daily.copy()
        level_columns = [
            column for column in daily.columns if column.endswith("_level")
        ]
        if level_columns:
            daily.loc[:, level_columns] = daily.loc[
                :,
                level_columns,
            ].round(12)
        checkpoint = result.checkpoint
        if checkpoint is not None:
            checkpoint = SimulationCheckpoint(
                business_date=checkpoint.business_date,
                index_weights=checkpoint.index_weights,
                benchmark_weights=checkpoint.benchmark_weights,
                levels={
                    name: round(float(value), 12)
                    for name, value in checkpoint.levels.items()
                },
                previous_observations=checkpoint.previous_observations,
            )
        return IndexSimulationResult(
            daily=daily,
            holdings=result.holdings,
            rebalances=result.rebalances,
            asset_returns=result.asset_returns,
            metadata=result.metadata,
            checkpoint=checkpoint,
            weight_snapshots=result.weight_snapshots,
        )

    def _reconstruct_levels_from_returns(
        self,
        result: IndexSimulationResult,
    ) -> IndexSimulationResult:
        """Build presentation levels deterministically from assembled returns."""

        daily = result.daily.copy()
        levels: dict[str, float] = {}
        for level_column in (
            column for column in daily if column.endswith("_level")
        ):
            return_column = (
                level_column.removesuffix("_level") + "_return"
            )
            if return_column not in daily:
                raise ValueError(
                    f"simulation output is missing {return_column!r}"
                )
            level = float(self.params.base_value)
            values: list[float] = []
            for daily_return in daily[return_column]:
                level *= 1.0 + float(daily_return)
                values.append(level)
            daily.loc[:, level_column] = values
            levels[level_column] = level
        checkpoint = result.checkpoint
        if checkpoint is not None:
            checkpoint = _checkpoint_with_levels(
                checkpoint,
                {
                    name: levels.get(name, value)
                    for name, value in checkpoint.levels.items()
                },
            )
        return IndexSimulationResult(
            daily=daily,
            holdings=result.holdings,
            rebalances=result.rebalances,
            asset_returns=result.asset_returns,
            metadata=result.metadata,
            checkpoint=checkpoint,
            weight_snapshots=result.weight_snapshots,
        )
__all__ = ["SimulationResultCacheMixin"]
