"""High-level orchestration for reproducible index-research workflows.

The v1 backtesting, simulation, analytics, and reporting APIs remain
available independently.  This module composes them behind a small research
API while keeping cache reuse opt-in and evidence based.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from hashlib import sha256
import math
import re
from typing import Any

import numpy as np
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
from ...workspace.identity import secret_safe_canonicalize
from .contracts import _PersistedMethodology
from ..models import IndexDefinition, ResearchSpec


def automatic_runtime_identity():
    """Collect runtime identity through the single research identity service."""

    return _collect_runtime_identity()


def _methodology_report_name(methodology: object) -> str:
    if isinstance(methodology, _PersistedMethodology):
        return methodology.report_name.rpartition(".")[2]
    wrapped = _recipe_methodology(methodology)
    if wrapped is not None:
        return type(wrapped).__name__
    return type(methodology).__name__


def _methodology_report_parameters(
    methodology: object,
) -> dict[str, Any]:
    if isinstance(methodology, _PersistedMethodology):
        raw = dict(methodology.parameters)
    else:
        wrapped = _recipe_methodology(methodology)
        raw = _methodology_parameters(wrapped or methodology)

    parameters = _report_parameter_mapping(raw)
    recipe = getattr(methodology, "recipe", None)
    if recipe is not None:
        recipe_id = _report_parameter_value(getattr(recipe, "recipe_id", None))
        recipe_version = _report_parameter_value(
            getattr(recipe, "recipe_version", None)
        )
        if recipe_id is not _OMIT_REPORT_PARAMETER:
            parameters["recipe_id"] = recipe_id
        if recipe_version is not _OMIT_REPORT_PARAMETER:
            parameters["recipe_version"] = recipe_version
    return dict(sorted(parameters.items()))


_OMIT_REPORT_PARAMETER = object()
_REPORT_PRIVATE_NAME_PARTS = {
    "accesskey",
    "apikey",
    "cache",
    "connection",
    "credential",
    "dsn",
    "endpoint",
    "executor",
    "oauth",
    "password",
    "privatekey",
    "providername",
    "providerparameter",
    "query",
    "runtime",
    "secret",
    "solver",
    "sql",
    "token",
    "url",
}
def _recipe_methodology(methodology: object) -> object | None:
    """Return one wrapped methodology without exposing recipe internals."""

    recipe = getattr(methodology, "recipe", None)
    nodes = getattr(recipe, "nodes", ()) if recipe is not None else ()
    candidates: list[object] = []
    seen: set[int] = set()
    for node in nodes:
        stage = getattr(node, "stage", None)
        candidate = getattr(stage, "methodology", None)
        if candidate is None or not callable(getattr(candidate, "execute", None)):
            continue
        identity = id(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(candidate)
    return candidates[0] if len(candidates) == 1 else None


def _report_parameter_mapping(
    values: Mapping[str, Any],
    *,
    depth: int = 0,
    seen: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Project configuration onto the report's public scalar contract."""

    if depth > 12 or id(values) in seen:
        return {}
    lineage = seen.union((id(values),))
    result: dict[str, Any] = {}
    used: set[str] = set()
    for raw_name, value in sorted(values.items(), key=lambda pair: str(pair[0])):
        source_name = str(raw_name)
        if _private_report_parameter_name(source_name):
            continue
        candidate = _report_parameter_value(
            value,
            depth=depth + 1,
            seen=lineage,
        )
        if candidate is _OMIT_REPORT_PARAMETER:
            continue
        name = _safe_report_parameter_name(source_name, used)
        used.add(name)
        result[name] = candidate
    return result


def _report_parameter_value(
    value: Any,
    *,
    depth: int = 0,
    seen: frozenset[int] = frozenset(),
) -> Any:
    if depth > 12:
        return _OMIT_REPORT_PARAMETER
    if value is None or value is pd.NA:
        return None
    if isinstance(value, Enum):
        return _report_parameter_value(
            value.value,
            depth=depth + 1,
            seen=seen,
        )
    if isinstance(value, (pd.Timestamp, datetime, date)):
        timestamp = pd.Timestamp(value)
        return None if pd.isna(timestamp) else timestamp.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else _OMIT_REPORT_PARAMETER
    if isinstance(value, str):
        canonical = secret_safe_canonicalize(value)
        return (
            "[REDACTED]"
            if isinstance(canonical, Mapping)
            else canonical
        )
    if is_dataclass(value) and not isinstance(value, type):
        if id(value) in seen:
            return _OMIT_REPORT_PARAMETER
        return _report_parameter_mapping(
            {
                item.name: getattr(value, item.name)
                for item in fields(value)
                if not item.name.startswith("_")
            },
            depth=depth + 1,
            seen=seen.union((id(value),)),
        )
    if isinstance(value, Mapping):
        if (
            set(value) == {"redacted", "identity_digest"}
            and value.get("redacted") is True
        ):
            return "[REDACTED]"
        return _report_parameter_mapping(value, depth=depth + 1, seen=seen)
    if isinstance(value, (set, frozenset)):
        if id(value) in seen:
            return _OMIT_REPORT_PARAMETER
        lineage = seen.union((id(value),))
        normalized = [
            _report_parameter_value(item, depth=depth + 1, seen=lineage)
            for item in value
        ]
        normalized = [
            item for item in normalized if item is not _OMIT_REPORT_PARAMETER
        ]
        return tuple(sorted(normalized, key=lambda item: repr(item)))
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if id(value) in seen:
            return _OMIT_REPORT_PARAMETER
        lineage = seen.union((id(value),))
        normalized = [
            _report_parameter_value(item, depth=depth + 1, seen=lineage)
            for item in value
        ]
        normalized = [
            item for item in normalized if item is not _OMIT_REPORT_PARAMETER
        ]
        if all(not isinstance(item, Mapping) for item in normalized):
            return tuple(normalized)
        return {
            f"item_{position:03d}": item
            for position, item in enumerate(normalized, start=1)
        }
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _report_parameter_value(
                scalar(),
                depth=depth + 1,
                seen=seen,
            )
        except (TypeError, ValueError):
            pass
    return _OMIT_REPORT_PARAMETER


def _private_report_parameter_name(value: str) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", value.casefold())
    separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated)
    tokens = re.findall(r"[a-z0-9]+", separated.casefold())
    token_set = set(tokens)
    name_compact = "".join(tokens)
    return (
        any(part in compact for part in _REPORT_PRIVATE_NAME_PARTS)
        or bool(
            token_set.intersection(
                {
                    "account",
                    "auth",
                    "authorization",
                    "database",
                    "host",
                    "oauth",
                    "role",
                    "server",
                    "uri",
                    "user",
                    "warehouse",
                }
            )
        )
        or ("schema" in token_set and "version" not in token_set)
        or name_compact in {
            "accountid",
            "accountname",
            "hostname",
            "userid",
            "username",
        }
        or (
            "path" in token_set
            and not token_set.intersection({"glide", "transition"})
        )
    )


def _safe_report_parameter_name(value: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", value.strip())
    base = re.sub(r"_+", "_", base).strip("_") or "parameter"
    if not base[0].isalpha():
        base = f"item_{base}"
    base = base[:128]
    if base not in used:
        return base
    suffix = sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{base[:119]}_{suffix}"


def _persistable_methodology_parameters(
    methodology: object,
) -> dict[str, Any]:
    raw = _methodology_parameters(methodology)
    try:
        persisted = secret_safe_canonicalize(raw)
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
            "methodology_report_name": _methodology_report_name(
                spec.definition.methodology
            ),
            "methodology_report_parameters": _methodology_report_parameters(
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
