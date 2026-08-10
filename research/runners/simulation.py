"""Effective-date segment simulation for research runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ...backtesting import Backtester, BacktestResult, IndexSimulationResult, IndexSimulator
from ...data_sources.providers.registry import registry
from ...workspace import (
    CacheMode,
    CachePolicy,
    ParquetStageStore,
    automatic_digest,
    automatic_provider_identity,
)
from ...workspace.caches.simulation import WorkspaceSimulationIdentityService
from ...workspace.caches.source import (
    BusinessDayCacheLoader,
    SourceDataCacheLoader,
    private_parameter_scope_digest,
)
from .cache import _source_content_revision
from .contracts import _CacheDecision
from ..models import ResearchSpec


class _SimulationRunner:
    """Run or reuse daily simulation segments."""

    def _simulate(
        self,
        spec: ResearchSpec,
        backtest: BacktestResult,
        backtester: Backtester,
        decision: _CacheDecision,
        source_decision: _CacheDecision,
    ) -> tuple[IndexSimulationResult, tuple[Mapping[str, Any], ...]]:
        simulation_spec = spec.simulation
        if simulation_spec is None:
            raise AssertionError("simulation spec is required")
        source_loader = SourceDataCacheLoader(
            self._workspace,
            provider_name=simulation_spec.market_data_provider_name,
            provider_parameters=simulation_spec.provider_parameters,
            mode=source_decision.actual,
        )
        business_day_loader: BusinessDayCacheLoader | None = None
        business_days: pd.DatetimeIndex | None = None
        calendar_provider_name = getattr(
            spec.calendar,
            "provider_name",
            None,
        )
        calendar_id = getattr(spec.calendar, "calendar_id", "")
        calendar_parameters = dict(
            getattr(spec.calendar, "provider_parameters", {}) or {}
        )
        if calendar_provider_name:
            calendar_provider = registry.get(calendar_provider_name)
            if callable(getattr(calendar_provider, "load_business_days", None)):
                business_day_loader = BusinessDayCacheLoader(
                    self._workspace,
                    provider_name=calendar_provider_name,
                    provider_parameters=calendar_parameters,
                    mode=source_decision.actual,
                )
                calendar_start = IndexSimulator(
                    backtest_result=backtest,
                    market_data_provider_name=(
                        simulation_spec.market_data_provider_name
                    ),
                    start_date=simulation_spec.start_date,
                    end_date=simulation_spec.end_date,
                    provider_parameters=dict(simulation_spec.provider_parameters),
                    params=simulation_spec.params,
                    streaming=simulation_spec.streaming,
                ).required_market_data_start()
                business_days = business_day_loader.load(
                    calendar_id=calendar_id,
                    start_date=calendar_start,
                    end_date=simulation_spec.end_date,
                )
        simulator_arguments = {
            "backtest_result": backtest,
            "market_data_provider_name": (simulation_spec.market_data_provider_name),
            "start_date": simulation_spec.start_date,
            "end_date": simulation_spec.end_date,
            "provider_parameters": dict(simulation_spec.provider_parameters),
            "params": simulation_spec.params,
            "streaming": simulation_spec.streaming,
            "business_days": business_days,
            "identity_service": WorkspaceSimulationIdentityService(),
        }
        market_data: pd.DataFrame | None = None
        requires_source_preflight = source_decision.actual is not CacheMode.OFF or (
            decision.actual is not CacheMode.OFF and decision.snapshot_digest is None
        )
        if requires_source_preflight:
            IndexSimulator(
                **simulator_arguments,
                market_data_loader=source_loader.load,
                cache_policy=CachePolicy.REUSE,
            ).verify_required_market_data()

        workspace: object | None = None
        data_revision: str | None = None
        cache_policy = CachePolicy.REUSE
        if decision.actual is not CacheMode.OFF:
            data_revision = decision.snapshot_digest
            if data_revision is None:
                data_revision = _source_content_revision(source_loader.records)
            cache_policy = CachePolicy(decision.actual.value)
            simulation_provider = registry.get(
                simulation_spec.market_data_provider_name
            )
            namespace_digest = automatic_digest(
                {
                    "schema_version": 2,
                    "kind": "research_simulation_artifacts",
                    "provider": automatic_provider_identity(
                        simulation_spec.market_data_provider_name,
                        simulation_provider,
                        capability="load_daily_market_data",
                        parameters=(simulation_spec.provider_parameters),
                    ),
                    "private_parameter_scope_digest": (
                        private_parameter_scope_digest(
                            simulation_spec.provider_parameters
                        )
                    ),
                }
            )
            workspace = ParquetStageStore(
                self._workspace,
                index_id=spec.definition.index_id,
                namespace_digest=namespace_digest,
            )
        result = IndexSimulator(
            **simulator_arguments,
            data_revision=data_revision,
            workspace=workspace,
            cache_policy=cache_policy,
            segmented_cache=(simulation_spec.segmented_cache and workspace is not None),
            market_data_lineage=tuple(source_loader.records),
            market_data=market_data,
            market_data_loader=(
                None if market_data is not None else source_loader.load
            ),
        ).run()
        source_records = tuple(source_loader.records)
        if not source_records:
            source_records = tuple(
                dict(record)
                for record in result.metadata.get(
                    "source_data_records",
                    (),
                )
            )
        if business_day_loader is not None:
            source_records = (
                *source_records,
                *business_day_loader.records,
            )
        return result, source_records


__all__ = ["_SimulationRunner"]
