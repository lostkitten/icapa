"""High-level orchestration for reproducible index-research workflows.

The v1 backtesting, simulation, analytics, and reporting APIs remain
available independently.  This module composes them behind a small research
API while keeping cache reuse opt-in and evidence based.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

import pandas as pd

from ..analytics import (
    AnalyticsDiagnostic,
    AnalyticsPluginResult,
    AnalyticsResult,
    AnalyticsRunResult,
    BrinsonAttribution,
)
from ..backtesting import (
    BacktestMetadata,
    BacktestResult,
    Calendar,
    IndexSimulationResult,
    RebalanceFrequency,
    ReviewResultMetadata,
)
from ..portfolio_construction.context import DataContext
from ..workspace import CacheSource, RunManifest, WorkspaceRepository
from .runners.identity import _optional_string
from .runners.contracts import _PersistedMethodology
from .runners.persistence import (
    _analytics_diagnostics_from_payload,
    _analytics_spec_from_payload,
    _decode_analytics_run_metadata,
    _decode_review_context_metadata,
)
from .models import IndexDefinition, ResearchWorkflowError


def _load_run_artifact_frames(
    workspace: WorkspaceRepository,
    manifest: RunManifest,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for reference in manifest.artifacts:
        if reference.artifact_type in frames:
            raise ResearchWorkflowError(
                "persisted run contains duplicate artifact types: "
                f"{reference.artifact_type}"
            )
        frames[reference.artifact_type] = workspace.load_frame(reference)
    if "review_weights" not in frames:
        raise ResearchWorkflowError(
            "persisted run is missing its review_weights artifact"
        )
    return frames


def _rehydrate_backtest(
    manifest: RunManifest,
    frames: Mapping[str, pd.DataFrame],
) -> BacktestResult:
    weights = _canonical_rehydrated_weights(frames["review_weights"])
    schedule = _persisted_review_schedule(manifest, weights)
    calendar = Calendar.from_dates(
        [
            {
                "reference_date": reference_date,
                "effective_date": effective_date,
            }
            for effective_date, reference_date in schedule.items()
        ]
    )
    reviews: dict[pd.Timestamp, DataContext] = {}
    review_metadata: dict[pd.Timestamp, ReviewResultMetadata] = {}
    for effective_date, reference_date in schedule.items():
        date_token = effective_date.strftime("%Y%m%d")
        constituents = frames.get(f"review_constituents.{date_token}")
        review_weights = weights.xs(
            effective_date,
            level="effective_date",
        )
        if constituents is None:
            constituents = review_weights.copy(deep=True)
        else:
            constituents = constituents.copy(deep=True)
            if constituents.index.name != "instrument_id":
                if "instrument_id" not in constituents:
                    raise ResearchWorkflowError(
                        "persisted review constituents are missing instrument_id"
                    )
                constituents = constituents.set_index(
                    "instrument_id",
                    verify_integrity=True,
                )
            if "index_weight" not in constituents:
                constituents["index_weight"] = review_weights["index_weight"].reindex(
                    constituents.index
                )
            expected_weights = review_weights["index_weight"].sort_index()
            persisted_weights = pd.to_numeric(
                constituents["index_weight"],
                errors="raise",
            ).sort_index()
            if not persisted_weights.index.equals(
                expected_weights.index
            ) or not persisted_weights.equals(expected_weights):
                raise ResearchWorkflowError(
                    "persisted review constituents do not match review weights"
                )
        context_metadata = _decode_review_context_metadata(
            frames.get(f"review_context_metadata.{date_token}")
        )
        context = DataContext(
            reference_date=reference_date,
            effective_date=effective_date,
            index_id=manifest.index_id,
            universe_id=context_metadata["universe_id"],
            calendar=calendar,
            provider_name=context_metadata["provider_name"],
            provider_parameters=context_metadata["provider_parameters"],
            diagnostics=deepcopy(context_metadata["diagnostics"]),
        )
        context.set_dataframe(constituents)
        daily = frames.get(f"review_daily.{date_token}")
        context.daily = None if daily is None else daily.copy(deep=True)
        for record in context_metadata["provenance_records"]:
            context.provenance.record_provider_call(record)
        reviews[effective_date] = context
        review_metadata[effective_date] = ReviewResultMetadata(
            reference_date=reference_date,
            effective_date=effective_date,
            cache_source=CacheSource.DISK,
        )
    return BacktestResult(
        weights=weights,
        reviews=reviews,
        metadata=BacktestMetadata(
            workspace_name=manifest.workspace_name,
            fingerprint=manifest.definition_fingerprint,
            cache_policy=None,
            workspace_path=None,
            reviews=review_metadata,
        ),
    )


def _canonical_rehydrated_weights(frame: pd.DataFrame) -> pd.DataFrame:
    weights = frame.copy(deep=True)
    if not {
        "effective_date",
        "instrument_id",
    }.issubset(weights.index.names):
        if not {
            "effective_date",
            "instrument_id",
        }.issubset(weights.columns):
            raise ResearchWorkflowError(
                "persisted review weights have an invalid schema"
            )
        weights = weights.set_index(
            ["effective_date", "instrument_id"],
            verify_integrity=True,
        )
    if "index_weight" not in weights:
        raise ResearchWorkflowError("persisted review weights are missing index_weight")
    effective_dates = pd.to_datetime(
        weights.index.get_level_values("effective_date"),
        errors="raise",
    ).normalize()
    instrument_ids = weights.index.get_level_values("instrument_id")
    weights.index = pd.MultiIndex.from_arrays(
        [effective_dates, instrument_ids],
        names=["effective_date", "instrument_id"],
    )
    return weights.sort_index(kind="mergesort")


def _persisted_review_schedule(
    manifest: RunManifest,
    weights: pd.DataFrame,
) -> dict[pd.Timestamp, pd.Timestamp]:
    request = dict(manifest.request)
    review_schedule = request.get("review_schedule", {})
    raw_dates = (
        review_schedule.get("dates", ()) if isinstance(review_schedule, Mapping) else ()
    )
    schedule: dict[pd.Timestamp, pd.Timestamp] = {}
    if isinstance(raw_dates, Sequence) and not isinstance(
        raw_dates,
        (str, bytes),
    ):
        for item in raw_dates:
            if not isinstance(item, Mapping):
                raise ResearchWorkflowError(
                    "persisted review schedule has an invalid schema"
                )
            effective = pd.Timestamp(item.get("effective_date")).normalize()
            reference = pd.Timestamp(item.get("reference_date")).normalize()
            if pd.isna(effective) or pd.isna(reference):
                raise ResearchWorkflowError(
                    "persisted review schedule contains a null date"
                )
            if reference > effective or effective in schedule:
                raise ResearchWorkflowError(
                    "persisted review schedule contains invalid dates"
                )
            schedule[effective] = reference
    weight_dates = {
        pd.Timestamp(value).normalize()
        for value in weights.index.get_level_values("effective_date")
    }
    if not schedule:
        schedule = {value: value for value in weight_dates}
    if set(schedule) != weight_dates:
        raise ResearchWorkflowError(
            "persisted review schedule and weights contain different dates"
        )
    return dict(sorted(schedule.items()))


def _rehydrate_simulation(
    manifest: RunManifest,
    frames: Mapping[str, pd.DataFrame],
) -> IndexSimulationResult | None:
    daily = frames.get("simulation_daily")
    if daily is None:
        return None
    request = dict(manifest.request)
    simulation_request = request.get("simulation", {})
    metadata = {
        "cache_source": "persisted_run",
        "rehydrated": True,
    }
    if isinstance(simulation_request, Mapping):
        for name in ("start_date", "end_date"):
            if simulation_request.get(name) is not None:
                metadata[name] = str(simulation_request[name])
    return IndexSimulationResult(
        daily=daily.copy(deep=True),
        holdings=frames.get(
            "simulation_weight_snapshots",
            pd.DataFrame(),
        ).copy(deep=True),
        rebalances=frames.get(
            "simulation_rebalances",
            pd.DataFrame(),
        ).copy(deep=True),
        asset_returns=frames.get(
            "simulation_asset_returns",
            pd.DataFrame(),
        ).copy(deep=True),
        metadata=metadata,
        weight_snapshots=frames.get(
            "simulation_rebalance_weight_snapshots",
            pd.DataFrame(),
        ).copy(deep=True),
    )


def _rehydrate_analytics(
    manifest: RunManifest,
    frames: Mapping[str, pd.DataFrame],
) -> AnalyticsRunResult | None:
    table_frames = {
        artifact_type.removeprefix("analytics."): frame.copy(deep=True)
        for artifact_type, frame in frames.items()
        if artifact_type.startswith("analytics.")
        and artifact_type != "analytics.metadata"
    }
    metadata_frame = frames.get("analytics.metadata")
    if not table_frames and metadata_frame is None:
        return None
    metadata = _decode_analytics_run_metadata(metadata_frame)
    spec = _analytics_spec_from_payload(
        (
            metadata.get("spec")
            if metadata is not None
            else dict(manifest.request).get("analytics")
        ),
        table_frames=table_frames,
    )
    plugin_tables: dict[str, dict[str, pd.DataFrame]] = {
        item.plugin_id: {} for item in spec.plugins
    }
    plugin_ids = sorted(
        plugin_tables,
        key=lambda value: (-len(value), value),
    )
    for qualified_name, frame in sorted(table_frames.items()):
        plugin_id = next(
            (name for name in plugin_ids if qualified_name.startswith(f"{name}.")),
            None,
        )
        if plugin_id is None:
            plugin_id, separator, table_name = qualified_name.partition(".")
            if not separator:
                raise ResearchWorkflowError("persisted analytics table name is invalid")
            plugin_tables.setdefault(plugin_id, {})[table_name] = frame
            continue
        table_name = qualified_name[len(plugin_id) + 1 :]
        plugin_tables[plugin_id][table_name] = frame

    plugin_metadata = (
        metadata.get("plugins", {}) if isinstance(metadata, Mapping) else {}
    )
    primary_plugin_id = next(
        (
            plugin_id
            for plugin_id in ("core_analytics", "legacy_parity")
            if plugin_id in plugin_tables
        ),
        "core_analytics",
    )
    legacy_metadata = (
        plugin_metadata.get(primary_plugin_id, {})
        if isinstance(plugin_metadata, Mapping)
        else {}
    )
    legacy_result = _rehydrate_legacy_analytics(
        plugin_tables.get(primary_plugin_id, {}),
        diagnostics=_analytics_diagnostics_from_payload(
            legacy_metadata.get("diagnostics", ())
            if isinstance(legacy_metadata, Mapping)
            else ()
        ),
    )
    plugin_results: dict[str, AnalyticsPluginResult] = {}
    for plugin_id, tables in plugin_tables.items():
        persisted = (
            plugin_metadata.get(plugin_id, {})
            if isinstance(plugin_metadata, Mapping)
            else {}
        )
        plugin_results[plugin_id] = AnalyticsPluginResult(
            metrics=(
                dict(persisted.get("metrics", {}))
                if isinstance(persisted, Mapping)
                and isinstance(persisted.get("metrics", {}), Mapping)
                else {}
            ),
            tables=tables,
            diagnostics=_analytics_diagnostics_from_payload(
                persisted.get("diagnostics", ())
                if isinstance(persisted, Mapping)
                else ()
            ),
            metadata=(
                {
                    **(
                        dict(persisted.get("metadata", {}))
                        if isinstance(persisted, Mapping)
                        and isinstance(
                            persisted.get("metadata", {}),
                            Mapping,
                        )
                        else {}
                    ),
                    **(
                        {"legacy_result": legacy_result}
                        if plugin_id == primary_plugin_id and legacy_result is not None
                        else {}
                    ),
                }
            ),
        )
    return AnalyticsRunResult(
        spec=spec,
        plugin_results=plugin_results,
        legacy_result=legacy_result,
        diagnostics=_analytics_diagnostics_from_payload(
            metadata.get("diagnostics", ()) if isinstance(metadata, Mapping) else ()
        ),
    )


def _rehydrate_legacy_analytics(
    tables: Mapping[str, pd.DataFrame],
    *,
    diagnostics: tuple[AnalyticsDiagnostic, ...] = (),
) -> AnalyticsResult | None:
    required = (
        "review_validation",
        "review_metrics",
        "country_exposures",
        "industry_exposures",
        "target_review_weight_change",
        "formal_turnover",
        "performance",
        "drawdowns",
    )
    if not all(name in tables for name in required):
        return None
    performance_frame = tables["performance"]
    if len(performance_frame.columns) != 1:
        raise ResearchWorkflowError(
            "persisted analytics performance has an invalid schema"
        )
    brinson = None
    if {
        "brinson_detail",
        "brinson_totals",
    }.issubset(tables):
        brinson = BrinsonAttribution(
            detail=tables["brinson_detail"],
            totals=tables["brinson_totals"],
        )
    return AnalyticsResult(
        review_validation=tables["review_validation"],
        review_metrics=tables["review_metrics"],
        country_exposures=tables["country_exposures"],
        industry_exposures=tables["industry_exposures"],
        target_review_weight_change=tables["target_review_weight_change"],
        formal_turnover=tables["formal_turnover"],
        performance=performance_frame.iloc[:, 0],
        drawdowns=tables["drawdowns"],
        brinson=brinson,
        diagnostics=diagnostics,
    )


def _rehydrate_definition(manifest: RunManifest) -> IndexDefinition:
    request = dict(manifest.request)
    payload = request.get("definition", {})
    if not isinstance(payload, Mapping):
        payload = {}
    methodology_name = (
        _optional_string(payload.get("methodology_name")) or "PersistedMethodology"
    )
    raw_parameters = payload.get("methodology_parameters", {})
    parameters = dict(raw_parameters) if isinstance(raw_parameters, Mapping) else {}
    raw_attributes = payload.get("attributes", {})
    attributes = dict(raw_attributes) if isinstance(raw_attributes, Mapping) else {}
    try:
        frequency = RebalanceFrequency(
            payload.get(
                "rebalance_frequency",
                RebalanceFrequency.CUSTOM.value,
            )
        )
    except ValueError as exc:
        raise ResearchWorkflowError("persisted rebalance frequency is invalid") from exc
    return IndexDefinition(
        index_id=manifest.index_id,
        methodology=_PersistedMethodology(
            report_name=methodology_name,
            parameters=parameters,
        ),
        name=_optional_string(payload.get("name"))
        or _optional_string(request.get("label")),
        base_currency=_optional_string(payload.get("base_currency")),
        attributes=attributes,
        rebalance_frequency=frequency,
    )


__all__ = [
    "_load_run_artifact_frames",
    "_rehydrate_backtest",
    "_rehydrate_simulation",
    "_rehydrate_analytics",
    "_rehydrate_definition",
]
