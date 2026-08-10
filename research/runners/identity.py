"""High-level orchestration for reproducible index-research workflows.

The v1 backtesting, simulation, analytics, and reporting APIs remain
available independently.  This module composes them behind a small research
API while keeping cache reuse opt-in and evidence based.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

import pandas as pd

from ...analytics import AnalyticsPluginRunner
from ...backtesting import Backtester, Calendar, IndexSimulator
from ...portfolio_construction import RecipeCompiler
from ...workspace import (
    IdentityError,
    automatic_component_identity,
    automatic_digest,
    automatic_runtime_identity as _collect_runtime_identity,
    dataframe_content_digest,
    safe_parameter_identity,
)
from ...workspace.identity import canonicalize
from .contracts import _PersistedMethodology
from ..models import IndexDefinition, ResearchSpec


def automatic_runtime_identity():
    """Collect runtime identity through the single research identity service."""

    return _collect_runtime_identity()


def _methodology_report_name(methodology: object) -> str:
    if isinstance(methodology, _PersistedMethodology):
        return methodology.report_name.rpartition(".")[2]
    return type(methodology).__name__


def _methodology_report_parameters(
    methodology: object,
) -> dict[str, Any]:
    if isinstance(methodology, _PersistedMethodology):
        return dict(methodology.parameters)
    return _methodology_parameters(methodology)


def _persistable_methodology_parameters(
    methodology: object,
) -> dict[str, Any]:
    raw = _methodology_parameters(methodology)
    try:
        persisted = canonicalize(raw)
    except (IdentityError, OSError, TypeError, ValueError):
        return {
            "parameter_names": sorted(map(str, raw)),
            "values_available": False,
        }
    if not isinstance(persisted, Mapping):
        return {"values_available": False}
    return dict(persisted)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _methodology_component_identity(
    definition: IndexDefinition,
) -> tuple[dict[str, Any], bool]:
    component, verified = _component_identity(definition.methodology)
    recipe = definition.recipe
    if recipe is None:
        return component, verified
    plan = RecipeCompiler().compile(
        recipe,
        allow_unfingerprintable=True,
    )
    recipe_verified = all(item.implementation_digest is not None for item in plan.nodes)
    return (
        {
            **component,
            "recipe": {
                "recipe_id": recipe.recipe_id,
                "recipe_version": recipe.recipe_version,
                "recipe_digest": plan.recipe_digest,
                "stages": [
                    {
                        "node_id": item.node.node_id,
                        "kind": item.node.stage.descriptor.kind,
                        "implementation_digest": (item.implementation_digest),
                    }
                    for item in plan.nodes
                ],
            },
            "source_verified": bool(verified and recipe_verified),
        },
        bool(verified and recipe_verified),
    )


def _component_identity(component: object) -> tuple[dict[str, Any], bool]:
    try:
        return {
            **automatic_component_identity(component),
            "source_verified": True,
        }, True
    except (IdentityError, OSError, TypeError, ValueError):
        component_type = type(component)
        fallback = {
            "type": (f"{component_type.__module__}." f"{component_type.__qualname__}"),
            "source_verified": False,
        }
        try:
            fallback["configuration_digest"] = automatic_digest(
                _methodology_parameters(component)
            )
        except (IdentityError, TypeError, ValueError):
            fallback["configuration_digest"] = automatic_digest(
                {"type": fallback["type"]}
            )
        return fallback, False


def _workflow_component_identities(
    spec: ResearchSpec,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, bool]]:
    selected: list[tuple[str, object]] = [
        ("backtester", Backtester),
        ("calendar", type(spec.calendar)),
    ]
    if spec.simulation is not None:
        selected.append(("simulator", IndexSimulator))
    if spec.analytics is not None:
        selected.append(("analytics_runner", AnalyticsPluginRunner))
    identities: dict[str, Mapping[str, Any]] = {}
    verified: dict[str, bool] = {}
    for name, component in selected:
        identity, is_verified = _component_identity(component)
        identities[name] = identity
        verified[name] = is_verified
    return identities, verified


def _definition_payload(
    definition: IndexDefinition,
    component_identity: Mapping[str, Any],
    workflow_components: Mapping[str, Mapping[str, Any]],
    construction_identity: str,
    providers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "index_id": definition.index_id,
        "name": definition.name,
        "base_currency": definition.base_currency,
        "rebalance_frequency": definition.rebalance_frequency.value,
        "attributes": dict(definition.attributes),
        "methodology": dict(component_identity),
        "construction_identity": construction_identity,
        "providers": [dict(provider) for provider in providers],
        "workflow_components": {
            name: dict(identity)
            for name, identity in sorted(workflow_components.items())
            if name in {"backtester", "calendar"}
        },
    }


def _request_payload(
    spec: ResearchSpec,
    *,
    workflow_components: Mapping[str, Mapping[str, Any]],
    providers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    schedule = spec.calendar.dates.loc[
        :,
        ["reference_date", "effective_date"],
    ].copy()
    reference_dates = pd.to_datetime(schedule["reference_date"]).dt.normalize()
    effective_dates = pd.to_datetime(schedule["effective_date"]).dt.normalize()
    simulation = spec.simulation
    return {
        "label": spec.label,
        "tags": list(spec.tags),
        "research_status": spec.status.value,
        "definition": {
            "name": spec.definition.name,
            "base_currency": spec.definition.base_currency,
            "attributes": dict(spec.definition.attributes),
            "rebalance_frequency": (spec.definition.rebalance_frequency.value),
            "methodology_name": (
                f"{type(spec.definition.methodology).__module__}."
                f"{type(spec.definition.methodology).__qualname__}"
            ),
            "methodology_parameters": _persistable_methodology_parameters(
                spec.definition.methodology
            ),
        },
        "review_schedule": {
            "content_digest": dataframe_content_digest(
                schedule,
                sort_by=["effective_date", "reference_date"],
            ),
            "review_count": int(len(schedule)),
            "first_reference_date": reference_dates.min(),
            "last_reference_date": reference_dates.max(),
            "first_effective_date": effective_dates.min(),
            "last_effective_date": effective_dates.max(),
            "dates": schedule.to_dict(orient="records"),
            "frequency_diagnostics": list(
                spec.calendar.frequency_diagnostics(spec.definition.rebalance_frequency)
            ),
            "validation_policy": {
                "allow_additional_reviews": (spec.allow_additional_reviews),
                "allow_frequency_gaps": spec.allow_frequency_gaps,
            },
        },
        "simulation": (
            {}
            if simulation is None
            else {
                "provider_name": simulation.market_data_provider_name,
                "provider_parameters": safe_parameter_identity(
                    simulation.provider_parameters
                ),
                "start_date": simulation.start_date,
                "end_date": simulation.end_date,
                "params": simulation.params,
                "segmented_cache": simulation.segmented_cache,
                "streaming": simulation.streaming,
            }
        ),
        "analytics": None if spec.analytics is None else spec.analytics,
        "analytics_inputs": (None if spec.analytics is None else spec.analytics_inputs),
        "report": (
            None
            if spec.report is None
            else {
                "name": spec.report.name,
                "contract_version": spec.report.contract_version,
                "formats": [item.value for item in spec.report.formats],
                "include_raw_tables": spec.report.include_raw_tables,
                "include_comparison": spec.report.include_comparison,
                "overwrite": spec.report.overwrite,
                "excel_large_table_policy": (
                    spec.report.excel_large_table_policy.value
                ),
            }
        ),
        "recipe_execution": {
            "random_seed": spec.random_seed,
            "allow_empty_initial_state": (spec.allow_empty_recipe_initial_state),
            "providers": {
                capability: {
                    "provider_name": binding.provider_name,
                    "parameters": safe_parameter_identity(binding.parameters),
                }
                for capability, binding in sorted(spec.recipe_providers.items())
            },
        },
        "execution_components": {
            name: dict(identity)
            for name, identity in sorted(workflow_components.items())
            if name not in {"backtester", "calendar"}
        },
        "providers": [dict(provider) for provider in providers],
    }


def _request_identity_payload(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove review-governance labels from calculation request identity."""

    return {
        str(key): value
        for key, value in request.items()
        if key not in {"label", "tags", "research_status"}
    }


def _calendar_semantics(calendar: Calendar) -> dict[str, Any]:
    return {
        "type": f"{type(calendar).__module__}.{type(calendar).__qualname__}",
        "calendar_id": calendar.calendar_id,
        "provider_name": calendar.provider_name,
        "provider_parameters": safe_parameter_identity(calendar.provider_parameters),
        "command": calendar.command,
    }


def _methodology_parameters(methodology: object) -> dict[str, Any]:
    if is_dataclass(methodology) and not isinstance(methodology, type):
        result: dict[str, Any] = {}
        for item in fields(methodology):
            if item.name.startswith("_"):
                continue
            value = getattr(methodology, item.name)
            if item.name.endswith("provider_parameters"):
                value = safe_parameter_identity(value)
            result[item.name] = value
        return result
    attributes = getattr(methodology, "__dict__", {})
    if not isinstance(attributes, Mapping):
        return {}
    return {
        str(name): (
            safe_parameter_identity(value)
            if str(name).endswith("provider_parameters") and isinstance(value, Mapping)
            else value
        )
        for name, value in attributes.items()
        if not str(name).startswith("_")
    }


def _report_data_sources(
    providers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project rich provider lineage onto the existing report contract."""

    return [
        {
            "capability": provider.get("capability", ""),
            "provider_name": provider.get("provider_name", ""),
            "data_type": "canonical_provider_contract",
            "fields": (),
        }
        for provider in providers
    ]


__all__ = [name for name in globals() if name.startswith("_")]
