"""High-level orchestration for reproducible index-research workflows.

The v1 backtesting, simulation, analytics, and reporting APIs remain
available independently.  This module composes them behind a small research
API while keeping cache reuse opt-in and evidence based.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...data_sources.providers.registry import registry
from ...portfolio_construction import ProviderRequestSpec, StageSideEffect
from ...workspace import (
    automatic_digest,
    automatic_provider_identity,
    safe_parameter_identity,
)
from . import identity as _identity
from .identity import _calendar_semantics
from .contracts import _ProviderEvidence
from ..models import ResearchSpec, UnsafeCacheReuseError


def _provider_identities(
    spec: ResearchSpec,
) -> _ProviderEvidence:
    requests = _provider_requests(spec)
    records: dict[str, Mapping[str, Any]] = {}
    review_records: dict[str, Mapping[str, Any]] = {}
    calendar_records: dict[str, Mapping[str, Any]] = {}
    simulation_records: dict[str, Mapping[str, Any]] = {}
    review_verified = True
    calendar_verified = True
    simulation_verified = True
    for provider_name, capability, parameters, usage, _ in requests:
        try:
            provider = registry.get(provider_name)
            record: Mapping[str, Any] = {
                **automatic_provider_identity(
                    provider_name,
                    provider,
                    capability=capability,
                    parameters=parameters,
                ),
                "source_verified": True,
            }
        except Exception:
            if usage == "review":
                review_verified = False
            elif usage == "calendar":
                calendar_verified = False
            elif usage == "simulation":
                simulation_verified = False
            record = {
                "provider_name": provider_name.strip().lower(),
                "capability": capability,
                "parameters": safe_parameter_identity(parameters),
                "source_verified": False,
            }
        record_key = automatic_digest(record)
        records[record_key] = record
        if usage == "review":
            review_records[record_key] = record
        elif usage == "calendar":
            calendar_records[record_key] = record
        elif usage == "simulation":
            simulation_records[record_key] = record
    return _ProviderEvidence(
        records=tuple(records[key] for key in sorted(records)),
        review_records=tuple(review_records[key] for key in sorted(review_records)),
        calendar_records=tuple(
            calendar_records[key] for key in sorted(calendar_records)
        ),
        simulation_records=tuple(
            simulation_records[key] for key in sorted(simulation_records)
        ),
        review_verified=review_verified,
        calendar_verified=calendar_verified,
        simulation_verified=simulation_verified,
    )


def _provider_requests(
    spec: ResearchSpec,
) -> list[tuple[str, str, Mapping[str, Any], str, str]]:
    requests: list[tuple[str, str, Mapping[str, Any], str, str]] = []
    required_recipe_capabilities = (
        set()
        if spec.definition.recipe is None
        else {
            capability
            for node in spec.definition.recipe.nodes
            for capability in node.stage.requirements.provider_capabilities
        }
    )
    for capability, binding in sorted(spec.recipe_providers.items()):
        if capability not in required_recipe_capabilities:
            continue
        requests.append(
            (
                binding.provider_name,
                capability,
                dict(binding.parameters),
                "review",
                f"recipe:{capability}",
            )
        )
    methodology = spec.definition.methodology
    for name in sorted(dir(methodology)):
        if not name.endswith("_provider_name") or name.startswith("_"):
            continue
        provider_name = getattr(methodology, name, None)
        if not isinstance(provider_name, str) or not provider_name.strip():
            continue
        prefix = name[: -len("_provider_name")]
        parameters = getattr(
            methodology,
            f"{prefix}_provider_parameters",
            {},
        )
        requests.append(
            (
                provider_name,
                _capability_for_prefix(prefix),
                dict(parameters or {}),
                "review",
                prefix,
            )
        )
    if spec.calendar.provider_name:
        requests.append(
            (
                spec.calendar.provider_name,
                "load_review_schedule",
                dict(spec.calendar.provider_parameters),
                "calendar",
                "calendar",
            )
        )
        try:
            calendar_provider = registry.get(spec.calendar.provider_name)
        except Exception:
            calendar_provider = None
        if callable(getattr(calendar_provider, "load_business_days", None)):
            requests.append(
                (
                    spec.calendar.provider_name,
                    "load_business_days",
                    dict(spec.calendar.provider_parameters),
                    "simulation",
                    "business_days",
                )
            )
    if spec.simulation is not None:
        requests.append(
            (
                spec.simulation.market_data_provider_name,
                "load_daily_market_data",
                dict(spec.simulation.provider_parameters),
                "simulation",
                "market_data",
            )
        )
    return requests


def _capability_for_prefix(prefix: str) -> str:
    return {
        "universe": "load_universe",
        "returns": "load_daily_market_data",
        "calendar": "load_review_schedule",
        "factor": "load_third_party_data",
        "signal": "load_third_party_data",
        "target": "load_third_party_data",
        "liquidity": "load_third_party_data",
    }.get(prefix, f"load_{prefix}")


def _recipe_has_no_external_inputs(spec: ResearchSpec) -> bool:
    recipe = spec.definition.recipe
    if recipe is None or recipe.required_artifacts:
        return False
    return all(
        node.stage.descriptor.deterministic
        and node.stage.descriptor.side_effect is StageSideEffect.PURE
        and not node.stage.requirements.provider_capabilities
        for node in recipe.nodes
    )


def _recipe_is_stateful(spec: ResearchSpec) -> bool:
    recipe = spec.definition.recipe
    return bool(
        recipe is not None
        and any(
            node.stage.requirements.consume_all_previous
            or node.stage.requirements.prior_artifacts
            for node in recipe.nodes
        )
    )


def _construction_identity(
    spec: ResearchSpec,
    *,
    methodology_component: Mapping[str, Any],
    backtester_component: Mapping[str, Any],
    calendar_component: Mapping[str, Any],
    providers: Sequence[Mapping[str, Any]],
) -> str:
    """Identify review construction independently from requested date coverage."""

    return automatic_digest(
        {
            "index_id": spec.definition.index_id,
            "base_currency": spec.definition.base_currency,
            "rebalance_frequency": spec.definition.rebalance_frequency.value,
            "attributes": dict(spec.definition.attributes),
            "methodology": dict(methodology_component),
            "backtester": dict(backtester_component),
            "calendar": dict(calendar_component),
            "providers": [dict(provider) for provider in providers],
            "calendar_semantics": _calendar_semantics(spec.calendar),
            "runtime": _identity.automatic_runtime_identity(),
        }
    )


def _validate_recipe_provider_request_contracts(
    spec: ResearchSpec,
) -> None:
    """Require exact provider-call declarations only when cache preflight runs."""

    recipe = spec.definition.recipe
    if recipe is None:
        return
    for node in recipe.nodes:
        requirements = node.stage.requirements
        declared = {request.capability for request in requirements.provider_requests}
        missing = set(requirements.provider_capabilities) - declared
        if missing:
            raise UnsafeCacheReuseError(
                f"recipe node {node.node_id!r} cannot use non-OFF review "
                "caching because it does not declare exact provider requests "
                f"for capabilities: {sorted(missing)}"
            )


def _recipe_provider_request_specs(
    spec: ResearchSpec,
    capability: str,
) -> tuple[ProviderRequestSpec, ...]:
    recipe = spec.definition.recipe
    if recipe is None:
        return ()
    requests = tuple(
        request
        for node in recipe.nodes
        for request in node.stage.requirements.provider_requests
        if request.capability == capability
    )
    if not requests:
        raise UnsafeCacheReuseError(
            "recipe provider caching requires an exact request declaration "
            f"for capability {capability!r}"
        )
    return requests


__all__ = [name for name in globals() if name.startswith("_")]
