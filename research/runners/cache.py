"""High-level orchestration for reproducible index-research workflows.

The v1 backtesting, simulation, analytics, and reporting APIs remain
available independently.  This module composes them behind a small research
API while keeping cache reuse opt-in and evidence based.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ...backtesting import BacktestResult, IndexSimulationResult
from ...workspace import (
    CacheMode,
    CacheStage,
    automatic_digest,
    dataframe_content_digest,
)
from ...workspace.caches.analytics import AnalyticsCacheIdentity
from ...workspace.identity import canonical_json_bytes
from .contracts import _CacheDecision, _ReviewSnapshotEvidence
from ..models import ResearchSpec, UnsafeCacheReuseError


def _cache_decisions(
    spec: ResearchSpec,
    *,
    review_source_verified: bool,
    simulation_source_verified: bool,
    analytics_source_verified: bool,
    review_snapshot: _ReviewSnapshotEvidence | None,
    simulation_snapshot: str | None,
) -> tuple[_CacheDecision, ...]:
    decisions: list[_CacheDecision] = []
    for stage in CacheStage:
        requested = spec.cache.mode_for(stage)
        if requested is CacheMode.OFF:
            decisions.append(
                _CacheDecision(
                    stage,
                    requested,
                    CacheMode.OFF,
                    "Cache is disabled by the research specification.",
                )
            )
            continue
        if stage is CacheStage.ANALYTICS:
            if spec.analytics is None:
                decisions.append(
                    _CacheDecision(
                        stage,
                        requested,
                        CacheMode.OFF,
                        "No analytics calculation was requested.",
                    )
                )
                continue
            if not analytics_source_verified:
                decisions.append(
                    _CacheDecision(
                        stage,
                        requested,
                        CacheMode.OFF,
                        "Analytics caching requires a verifiable plugin-runner "
                        "implementation identity.",
                        fatal=True,
                    )
                )
                continue
            decisions.append(
                _CacheDecision(
                    stage,
                    requested,
                    requested,
                    "Analytics results use content-identified immutable "
                    "Parquet artifacts.",
                )
            )
            continue
        if stage is CacheStage.SOURCE_DATA:
            applicable = spec.simulation is not None
            if not applicable:
                decisions.append(
                    _CacheDecision(
                        stage,
                        requested,
                        CacheMode.OFF,
                        "No simulation source data was requested.",
                    )
                )
                continue
            if not simulation_source_verified:
                reason = (
                    "source_data caching was disabled because the provider "
                    "implementation identity could not be verified."
                )
                decisions.append(
                    _CacheDecision(
                        stage,
                        requested,
                        CacheMode.OFF,
                        reason,
                        fatal=True,
                    )
                )
                continue
            if requested is CacheMode.READ_ONLY and simulation_snapshot is None:
                decisions.append(
                    _CacheDecision(
                        stage,
                        requested,
                        CacheMode.OFF,
                        "READ_ONLY source_data access requires an automatic "
                        "provider snapshot identity.",
                        fatal=True,
                    )
                )
                continue
            decisions.append(
                _CacheDecision(
                    stage,
                    requested,
                    requested,
                    (
                        "Source partitions use verified provider snapshots and "
                        "scenario-independent immutable Parquet artifacts."
                        if simulation_snapshot is not None
                        else "The provider has no preflight snapshot token; source "
                        "data will be loaded and content-hashed before it is "
                        "persisted, and no unverified old artifact will be read."
                    ),
                    snapshot_digest=simulation_snapshot,
                )
            )
            continue
        snapshot = (
            (None if review_snapshot is None else review_snapshot.snapshot_digest)
            if stage is CacheStage.REVIEWS
            else simulation_snapshot
        )
        source_verified = (
            review_source_verified
            if stage is CacheStage.REVIEWS
            else simulation_source_verified
        )
        applicable = not (stage is CacheStage.SIMULATION and spec.simulation is None)
        if not applicable:
            decisions.append(
                _CacheDecision(
                    stage,
                    requested,
                    CacheMode.OFF,
                    "No simulation was requested.",
                )
            )
            continue
        if (
            stage is CacheStage.SIMULATION
            and source_verified
            and snapshot is None
            and requested in {CacheMode.REUSE, CacheMode.REFRESH}
        ):
            decisions.append(
                _CacheDecision(
                    stage,
                    requested,
                    requested,
                    "The provider has no snapshot token; canonical source data "
                    "will be loaded and content-hashed before downstream "
                    "simulation cache lookup.",
                )
            )
            continue
        if (
            stage is CacheStage.REVIEWS
            and spec.definition.recipe is not None
            and bool(spec.recipe_providers)
            and source_verified
            and snapshot is None
            and requested in {CacheMode.REUSE, CacheMode.REFRESH}
        ):
            decisions.append(
                _CacheDecision(
                    stage,
                    requested,
                    requested,
                    "Recipe providers have no preflight snapshot token; "
                    "provider stages will execute and content-identify their "
                    "outputs before downstream stage reuse.",
                )
            )
            continue
        if not source_verified:
            reason = (
                f"{stage.value} caching requires stable executable source " "identity."
            )
            decisions.append(
                _CacheDecision(
                    stage,
                    requested,
                    CacheMode.OFF,
                    reason,
                    fatal=True,
                )
            )
            continue
        if snapshot is None:
            decisions.append(
                _CacheDecision(
                    stage,
                    requested,
                    CacheMode.OFF,
                    f"{stage.value} reuse was disabled because automatic "
                    "data-snapshot identity was unavailable.",
                    fatal=requested is CacheMode.READ_ONLY,
                )
            )
            continue
        decisions.append(
            _CacheDecision(
                stage,
                requested,
                requested,
                "Automatic source and data-snapshot identities were verified.",
                snapshot_digest=snapshot,
                review_snapshot=(
                    review_snapshot if stage is CacheStage.REVIEWS else None
                ),
            )
        )
    return tuple(decisions)


def _decision_for(
    decisions: Sequence[_CacheDecision],
    stage: CacheStage,
) -> _CacheDecision:
    return next(item for item in decisions if item.stage is stage)


def _source_content_revision(
    records: Sequence[Mapping[str, Any]],
) -> str:
    """Build a downstream revision from verified canonical source content."""

    if not records:
        raise UnsafeCacheReuseError(
            "simulation caching requires verified source-data content"
        )
    partitions = [
        {
            "provider_name": record.get("provider_name"),
            "capability": record.get("capability"),
            "request_digest": record.get("request_digest"),
            "content_digest": record.get("content_digest"),
            "rows": record.get("rows"),
        }
        for record in records
    ]
    return automatic_digest(
        {
            "kind": "verified_simulation_source_content",
            "partitions": sorted(
                partitions,
                key=canonical_json_bytes,
            ),
        }
    )


def _input_digest_records(
    backtest: BacktestResult,
    decisions: Sequence[_CacheDecision],
    *,
    source_records: Sequence[Mapping[str, Any]] = (),
    analytics_records: Sequence[Mapping[str, Any]] = (),
) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = [
        dict(record) for record in (*source_records, *analytics_records)
    ]
    for decision in decisions:
        if decision.snapshot_digest is not None:
            records.append(
                {
                    "input_type": f"{decision.stage.value}_snapshot",
                    "content_digest": decision.snapshot_digest,
                }
            )
    for effective_date, context in sorted(backtest.reviews.items()):
        provenance = getattr(context, "provenance", None)
        for record in getattr(provenance, "records", ()):
            records.append(
                {
                    "input_type": record.get("provider", {}).get(
                        "capability",
                        "provider_data",
                    ),
                    "effective_date": pd.Timestamp(effective_date).normalize(),
                    "content_digest": record.get("content_digest"),
                    "request_digest": record.get("request_digest"),
                }
            )
        recipe = context.diagnostics.get("index_recipe", {})
        for stage in recipe.get("stages", ()):
            random_seed = stage.get("random_seed")
            if random_seed is not None:
                records.append(
                    {
                        "input_type": "recipe_stage_random_seed",
                        "effective_date": pd.Timestamp(effective_date).normalize(),
                        "stage": stage.get("node_id"),
                        "random_seed": int(random_seed),
                        "content_digest": automatic_digest(
                            {
                                "stage": stage.get("node_id"),
                                "random_seed": int(random_seed),
                            }
                        ),
                    }
                )
            for name, digest in sorted(dict(stage.get("output_digests", {})).items()):
                records.append(
                    {
                        "input_type": "recipe_stage_artifact",
                        "effective_date": pd.Timestamp(effective_date).normalize(),
                        "stage": stage.get("node_id"),
                        "artifact": name,
                        "content_digest": digest,
                    }
                )
    return tuple(records)


def _analytics_cache_identity(
    spec: ResearchSpec,
    backtest: BacktestResult,
    simulation: IndexSimulationResult | None,
    *,
    runner_identity: Mapping[str, Any],
) -> AnalyticsCacheIdentity:
    if spec.analytics is None:
        raise ValueError("analytics specification is required")
    reviews: list[dict[str, Any]] = []
    for effective_date, context in sorted(backtest.reviews.items()):
        reviews.append(
            {
                "effective_date": pd.Timestamp(effective_date).normalize(),
                "reference_date": pd.Timestamp(context.reference_date).normalize(),
                "universe_id": context.universe_id,
                "constituents_digest": _analytics_frame_digest(context.cons),
                "daily_digest": (
                    None
                    if context.daily is None
                    else _analytics_frame_digest(context.daily)
                ),
                "diagnostics_digest": automatic_digest(context.diagnostics),
            }
        )
    simulation_identity = (
        None
        if simulation is None
        else {
            "daily": _analytics_frame_digest(simulation.daily),
            "holdings": _analytics_frame_digest(simulation.holdings),
            "rebalances": _analytics_frame_digest(simulation.rebalances),
            "asset_returns": _analytics_frame_digest(simulation.asset_returns),
            "weight_snapshots": _analytics_frame_digest(simulation.weight_snapshots),
        }
    )
    return AnalyticsCacheIdentity.from_inputs(
        index_id=spec.definition.index_id,
        analytics_spec=spec.analytics,
        research_inputs=spec.analytics_inputs,
        analytics_runner=dict(runner_identity),
        backtest_weights=_analytics_frame_digest(backtest.weights),
        reviews=reviews,
        simulation=simulation_identity,
    )


def _analytics_frame_digest(frame: pd.DataFrame) -> str:
    """Normalize an implicit positional index before analytics identity."""

    working = frame
    if (
        frame.index.nlevels == 1
        and frame.index.name is None
        and frame.index.equals(pd.RangeIndex(len(frame)))
    ):
        working = frame.reset_index(drop=True)
    return dataframe_content_digest(working)


__all__ = [name for name in globals() if name.startswith("_")]
