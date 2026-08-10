"""High-level orchestration for reproducible index-research workflows.

The v1 backtesting, simulation, analytics, and reporting APIs remain
available independently.  This module composes them behind a small research
API while keeping cache reuse opt-in and evidence based.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import pandas as pd

from ...analytics import (
    AnalyticsDiagnostic,
    AnalyticsPluginSpec,
    AnalyticsRunResult,
    AnalyticsSpec,
    MissingInputPolicy,
    ReturnSeries,
)
from ...portfolio_construction.context import DataContext
from ...workspace import IdentityError
from ...workspace.identity import canonical_json_bytes, canonicalize
from ..models import ResearchWorkflowError


def _analytics_run_metadata_frame(
    analytics: AnalyticsRunResult,
) -> pd.DataFrame:
    payload = {
        "schema_version": 1,
        "spec": analytics.spec,
        "plugins": {
            plugin_id: {
                "metrics": {
                    str(name): _persisted_analytics_metric(value)
                    for name, value in plugin.metrics.items()
                },
                "diagnostics": [asdict(item) for item in plugin.diagnostics],
                "metadata": _persisted_plugin_metadata(plugin.metadata),
            }
            for plugin_id, plugin in analytics.plugin_results.items()
        },
        "diagnostics": [asdict(item) for item in analytics.diagnostics],
    }
    return pd.DataFrame(
        {"payload_json": [canonical_json_bytes(payload).decode("utf-8")]}
    )


def _persisted_plugin_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Persist safe plugin metadata while rebuilding v1 object references."""

    raw = {
        str(name): value
        for name, value in metadata.items()
        if str(name) != "legacy_result"
    }
    try:
        value = canonicalize(raw)
    except (IdentityError, OSError, TypeError, ValueError):
        return {
            "metadata_keys": sorted(raw),
            "values_available": False,
        }
    if not isinstance(value, Mapping):
        return {"values_available": False}
    return dict(value)


def _persisted_analytics_metric(value: Any) -> Any:
    scalar = value.item() if hasattr(value, "item") else value
    if isinstance(scalar, float) and not math.isfinite(scalar):
        if math.isnan(scalar):
            return "NaN"
        return "Infinity" if scalar > 0 else "-Infinity"
    return scalar


def _decode_analytics_run_metadata(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    if frame is None:
        return None
    if list(frame.columns) != ["payload_json"] or len(frame) != 1:
        raise ResearchWorkflowError(
            "persisted analytics metadata has an invalid schema"
        )
    value = frame.iloc[0]["payload_json"]
    if not isinstance(value, str):
        raise ResearchWorkflowError("persisted analytics metadata is not JSON text")
    decoded = json.loads(value)
    if not isinstance(decoded, Mapping) or decoded.get("schema_version") != 1:
        raise ResearchWorkflowError(
            "persisted analytics metadata version is unsupported"
        )
    return dict(decoded)


def _analytics_spec_from_payload(
    payload: Any,
    *,
    table_frames: Mapping[str, pd.DataFrame],
) -> AnalyticsSpec:
    if not isinstance(payload, Mapping):
        plugin_ids = sorted(
            {name.partition(".")[0] for name in table_frames if "." in name}
        )
        return AnalyticsSpec(
            profile="persisted_research",
            plugins=tuple(AnalyticsPluginSpec(plugin_id) for plugin_id in plugin_ids),
        )
    raw_plugins = payload.get("plugins", ())
    if not isinstance(raw_plugins, Sequence) or isinstance(
        raw_plugins,
        (str, bytes),
    ):
        raise ResearchWorkflowError("persisted analytics specification is invalid")
    plugins = []
    for item in raw_plugins:
        if not isinstance(item, Mapping):
            raise ResearchWorkflowError(
                "persisted analytics plugin specification is invalid"
            )
        parameters = item.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ResearchWorkflowError(
                "persisted analytics plugin parameters are invalid"
            )
        plugins.append(
            AnalyticsPluginSpec(
                plugin_id=str(item.get("plugin_id", "")),
                version=str(item.get("version", "1")),
                parameters=dict(parameters),
                required=bool(item.get("required", True)),
            )
        )
    return AnalyticsSpec(
        profile=str(payload.get("profile", "persisted_research")),
        plugins=tuple(plugins),
        return_series=ReturnSeries(
            payload.get("return_series", ReturnSeries.NET_TOTAL.value)
        ),
        annualization_factor=int(payload.get("annualization_factor", 252)),
        weight_tolerance=float(payload.get("weight_tolerance", 1e-8)),
        missing_optional_input=MissingInputPolicy(
            payload.get(
                "missing_optional_input",
                MissingInputPolicy.WARN_AND_SKIP.value,
            )
        ),
    )


def _analytics_diagnostics_from_payload(
    payload: Any,
) -> tuple[AnalyticsDiagnostic, ...]:
    if not isinstance(payload, Sequence) or isinstance(
        payload,
        (str, bytes),
    ):
        raise ResearchWorkflowError("persisted analytics diagnostics are invalid")
    result = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ResearchWorkflowError("persisted analytics diagnostic is invalid")
        result.append(
            AnalyticsDiagnostic(
                level=str(item.get("level", "")),
                code=str(item.get("code", "")),
                message=str(item.get("message", "")),
            )
        )
    return tuple(result)


def _result_artifact_sort_columns(
    frame: pd.DataFrame,
) -> tuple[str, ...] | None:
    available = {
        str(name) for name in (*frame.index.names, *frame.columns) if name is not None
    }
    preferred = (
        "effective_date",
        "reference_date",
        "business_date",
        "scheduled_effective_date",
        "applied_business_date",
        "period",
        "instrument_id",
        "snapshot",
        "phase",
        "constraint_name",
        "target_name",
        "exposure_type",
        "field",
        "source",
        "country",
        "industry",
        "name",
    )
    selected = tuple(name for name in preferred if name in available)
    return selected or None


def _persisted_review_constituents(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Retain the full review frame required by custom research analytics."""

    if "index_weight" not in frame.columns:
        raise ResearchWorkflowError("review constituents are missing index_weight")
    return frame.copy(deep=True)


def _review_context_metadata_frame(context: DataContext) -> pd.DataFrame:
    """Encode non-tabular review state without pickle or secret values."""

    provenance_records = tuple(context.provenance.records)
    if any(not isinstance(item, Mapping) for item in provenance_records):
        raise ResearchWorkflowError("review provenance records must be mappings")
    payload = {
        "schema_version": 1,
        "universe_id": str(context.universe_id or ""),
        "provider_name": (
            None if context.provider_name is None else str(context.provider_name)
        ),
        "provider_parameters": dict(context.provider_parameters or {}),
        "diagnostics": dict(context.diagnostics or {}),
        "provenance_records": [dict(item) for item in provenance_records],
    }
    try:
        encoded = canonical_json_bytes(payload).decode("utf-8")
    except (IdentityError, OSError, TypeError, ValueError) as error:
        raise ResearchWorkflowError(
            "review context metadata cannot be serialized safely"
        ) from error
    return pd.DataFrame({"payload_json": [encoded]})


def _decode_review_context_metadata(
    frame: pd.DataFrame | None,
) -> dict[str, Any]:
    """Restore safe review metadata, accepting pre-v2 runs without it."""

    empty = {
        "universe_id": "",
        "provider_name": None,
        "provider_parameters": {},
        "diagnostics": {},
        "provenance_records": (),
    }
    if frame is None:
        return empty
    if list(frame.columns) != ["payload_json"] or len(frame) != 1:
        raise ResearchWorkflowError(
            "persisted review context metadata has an invalid schema"
        )
    raw = frame.iloc[0]["payload_json"]
    if not isinstance(raw, str):
        raise ResearchWorkflowError(
            "persisted review context metadata is not JSON text"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ResearchWorkflowError(
            "persisted review context metadata is invalid JSON"
        ) from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ResearchWorkflowError(
            "persisted review context metadata version is unsupported"
        )
    diagnostics = payload.get("diagnostics", {})
    provider_parameters = payload.get("provider_parameters", {})
    provenance_records = payload.get("provenance_records", ())
    provider_name = payload.get("provider_name")
    universe_id = payload.get("universe_id", "")
    if not isinstance(diagnostics, Mapping):
        raise ResearchWorkflowError("persisted review diagnostics must be a mapping")
    if not isinstance(provider_parameters, Mapping):
        raise ResearchWorkflowError(
            "persisted review provider parameters must be a mapping"
        )
    if (
        not isinstance(provenance_records, Sequence)
        or isinstance(provenance_records, (str, bytes))
        or any(not isinstance(item, Mapping) for item in provenance_records)
    ):
        raise ResearchWorkflowError(
            "persisted review provenance must be an array of mappings"
        )
    if provider_name is not None and not isinstance(provider_name, str):
        raise ResearchWorkflowError(
            "persisted review provider name must be text or null"
        )
    if not isinstance(universe_id, str):
        raise ResearchWorkflowError("persisted review universe ID must be text")
    return {
        "universe_id": universe_id,
        "provider_name": provider_name,
        "provider_parameters": dict(provider_parameters),
        "diagnostics": dict(diagnostics),
        "provenance_records": tuple(dict(item) for item in provenance_records),
    }


__all__ = [name for name in globals() if name.startswith("_")]
