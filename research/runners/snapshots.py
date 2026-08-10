"""High-level orchestration for reproducible index-research workflows.

The v1 backtesting, simulation, analytics, and reporting APIs remain
available independently.  This module composes them behind a small research
API while keeping cache reuse opt-in and evidence based.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ...data_sources.providers.registry import registry
from ...portfolio_construction import ProviderRequestSpec, ReviewIdentity
from ...workspace import (
    CacheMode,
    CacheStage,
    WorkspaceRepository,
    automatic_digest,
    automatic_provider_identity,
)
from ...workspace.caches.source import (
    private_parameter_scope_digest,
    workspace_provider_snapshot_digest,
)
from ...workspace.identity import canonical_json_bytes
from .contracts import (
    _ExactReviewProviderBinding,
    _ReviewSnapshotEvidence,
    _ReviewSnapshotRecord,
)
from .providers import (
    _provider_requests,
    _recipe_has_no_external_inputs,
    _validate_recipe_provider_request_contracts,
)
from ..models import ResearchSpec, UnsafeCacheReuseError


def _review_snapshot(
    spec: ResearchSpec,
    *,
    workspace: WorkspaceRepository,
    requested: CacheMode,
    diagnostics: list[str],
    construction_identity: str,
) -> _ReviewSnapshotEvidence | None:
    _validate_recipe_provider_request_contracts(spec)
    review_requests = [item for item in _provider_requests(spec) if item[3] == "review"]
    if review_requests:
        exact_requests: list[_ExactReviewProviderBinding] = []
        for (
            provider_name,
            capability,
            parameters,
            _,
            prefix,
            declared_request,
        ) in review_requests:
            provider = registry.get(provider_name)
            if not callable(getattr(provider, "describe_snapshot", None)):
                exact_requests = []
                break
            request_specs = (
                (declared_request,)
                if prefix.startswith("recipe:")
                else (None,)
            )
            for request_spec in request_specs:
                exact_request = _exact_review_provider_request(
                    spec,
                    prefix=prefix,
                    capability=capability,
                    parameters=parameters,
                    reference_date=(spec.calendar.dates.iloc[0]["reference_date"]),
                    effective_date=(spec.calendar.dates.iloc[0]["effective_date"]),
                    request_spec=request_spec,
                )
                if exact_request is None:
                    diagnostics.append(
                        "Exact review-provider snapshot requests cannot be "
                        f"derived for capability {capability!r}; methodology "
                        "identity fallback will be attempted."
                    )
                    exact_requests = []
                    break
                exact_requests.append(
                    _ExactReviewProviderBinding(
                        provider_name=provider_name,
                        capability=capability,
                        parameters=parameters,
                        prefix=prefix,
                        provider=provider,
                        request_spec=request_spec,
                    )
                )
            if not exact_requests:
                break
        if exact_requests:
            return _exact_review_snapshots(
                spec,
                workspace=workspace,
                requested=requested,
                requests=exact_requests,
            )
    elif _recipe_has_no_external_inputs(spec):
        digest = automatic_digest(
            {
                "kind": "recipe_without_external_inputs",
                "construction_identity": construction_identity,
            }
        )
        return _global_review_snapshot_evidence(
            spec,
            snapshot_digest=digest,
            scope_digest=digest,
        )
    return _legacy_review_snapshot(
        spec,
        workspace=workspace,
        requested=requested,
        diagnostics=diagnostics,
        construction_identity=construction_identity,
    )


_LEGACY_REVIEW_SNAPSHOT_BINDING = "methodology-snapshot"
_REVIEW_SNAPSHOT_SCHEMA_VERSION = 1


def _exact_review_snapshots(
    spec: ResearchSpec,
    *,
    workspace: WorkspaceRepository,
    requested: CacheMode,
    requests: Sequence[_ExactReviewProviderBinding],
) -> _ReviewSnapshotEvidence | None:
    scope_records: list[Mapping[str, Any]] = []
    for binding in requests:
        scope_records.append(
            {
                "provider": automatic_provider_identity(
                    binding.provider_name,
                    binding.provider,
                    capability=binding.capability,
                    parameters=binding.parameters,
                ),
                "private_parameter_scope_digest": (
                    private_parameter_scope_digest(binding.parameters)
                ),
                "request_contract": (
                    None
                    if binding.request_spec is None
                    else {
                        "capability": binding.request_spec.capability,
                        "review_dimensions": sorted(
                            binding.request_spec.review_dimensions
                        ),
                        "include_provider_parameters": (
                            binding.request_spec.include_provider_parameters
                        ),
                    }
                ),
            }
        )
    records: list[_ReviewSnapshotRecord] = []
    for review in spec.calendar.dates.itertuples(index=False):
        reference_date = pd.Timestamp(review.reference_date).normalize()
        effective_date = pd.Timestamp(review.effective_date).normalize()
        provider_snapshots: list[dict[str, Any]] = []
        for binding in requests:
            exact_request = _exact_review_provider_request(
                spec,
                prefix=binding.prefix,
                capability=binding.capability,
                parameters=binding.parameters,
                reference_date=reference_date,
                effective_date=effective_date,
                request_spec=binding.request_spec,
            )
            if exact_request is None:
                raise AssertionError(
                    "review snapshot request became unavailable after preflight"
                )
            try:
                snapshot = workspace_provider_snapshot_digest(
                    workspace,
                    binding.provider,
                    provider_name=binding.provider_name,
                    capability=binding.capability,
                    parameters=binding.parameters,
                    request=exact_request,
                    mode=requested,
                )
            except Exception as error:
                raise UnsafeCacheReuseError(
                    "exact review-provider snapshot identity failed for "
                    f"provider {binding.provider_name!r} capability "
                    f"{binding.capability!r}"
                ) from error
            if snapshot is None:
                if requested is CacheMode.READ_ONLY:
                    return None
                raise UnsafeCacheReuseError(
                    "exact review-provider snapshot identity returned no "
                    f"evidence for provider {binding.provider_name!r} "
                    f"capability {binding.capability!r}"
                )
            snapshot_digest, snapshot_protocol = snapshot
            provider_snapshots.append(
                {
                    "provider_name": binding.provider_name.strip().lower(),
                    "capability": binding.capability,
                    "snapshot_digest": snapshot_digest,
                    "snapshot_protocol": snapshot_protocol,
                    "private_parameter_scope_digest": (
                        private_parameter_scope_digest(binding.parameters)
                    ),
                    "private_request_scope_digest": (
                        private_parameter_scope_digest(exact_request)
                    ),
                }
            )
        records.append(
            _ReviewSnapshotRecord(
                reference_date=reference_date,
                effective_date=effective_date,
                snapshot_digest=automatic_digest(
                    {
                        "kind": "review_provider_snapshots",
                        "snapshots": sorted(
                            provider_snapshots,
                            key=canonical_json_bytes,
                        ),
                    }
                ),
            )
        )
    return _ReviewSnapshotEvidence(
        records=tuple(records),
        scope_digest=automatic_digest(
            {
                "kind": "review_provider_snapshot_scope",
                "providers": sorted(
                    scope_records,
                    key=canonical_json_bytes,
                ),
            }
        ),
    )


def _exact_review_provider_request(
    spec: ResearchSpec,
    *,
    prefix: str,
    capability: str,
    parameters: Mapping[str, Any],
    reference_date: object,
    effective_date: object,
    request_spec: ProviderRequestSpec | None = None,
) -> dict[str, Any] | None:
    """Build the canonical request actually issued for one review capability."""

    if prefix.startswith("recipe:"):
        if request_spec is None:
            return None
        return request_spec.build_request(
            ReviewIdentity(
                index_id=spec.definition.index_id,
                reference_date=reference_date,
                effective_date=effective_date,
            ),
            parameters,
        )
    if capability != "load_universe":
        return None
    methodology = spec.definition.methodology
    universe_id = getattr(methodology, f"{prefix}_id", None)
    if universe_id is None and prefix == "universe":
        universe_id = getattr(methodology, "universe_id", None)
    if universe_id is None:
        return None
    reserved = {"universe_id", "reference_date", "effective_date"}
    overlap = reserved.intersection(parameters)
    if overlap:
        raise ValueError(
            "universe provider parameters must not override canonical request "
            f"fields: {sorted(overlap)}"
        )
    return {
        **dict(parameters),
        "universe_id": universe_id,
        "reference_date": pd.Timestamp(reference_date).normalize(),
        "effective_date": pd.Timestamp(effective_date).normalize(),
    }


def _legacy_review_snapshot(
    spec: ResearchSpec,
    *,
    workspace: WorkspaceRepository,
    requested: CacheMode,
    diagnostics: list[str],
    construction_identity: str,
) -> _ReviewSnapshotEvidence | None:
    private_provider_scope_digest = automatic_digest(
        [
            {
                "provider_name": provider_name.strip().lower(),
                "capability": capability,
                "parameter_scope": private_parameter_scope_digest(parameters),
            }
            for (
                provider_name,
                capability,
                parameters,
                usage,
                _,
                _,
            ) in _provider_requests(spec)
            if usage == "review"
        ]
    )
    cache_key = automatic_digest(
        {
            "schema_version": _REVIEW_SNAPSHOT_SCHEMA_VERSION,
            "kind": "methodology_snapshot_descriptor",
            "construction_identity": construction_identity,
            "private_provider_scope_digest": (private_provider_scope_digest),
        }
    )
    if requested is CacheMode.READ_ONLY:
        reference = workspace.resolve_artifact(
            stage=CacheStage.REVIEWS,
            cache_key=cache_key,
            name=_LEGACY_REVIEW_SNAPSHOT_BINDING,
        )
        if reference is None:
            return None
        frame = workspace.load_frame(reference)
        digest = _decode_review_snapshot_descriptor(frame)
        return _global_review_snapshot_evidence(
            spec,
            snapshot_digest=digest,
            scope_digest=cache_key,
        )

    identity_provider = getattr(
        spec.definition.methodology,
        "research_data_identity",
        None,
    )
    if not callable(identity_provider):
        return None
    try:
        # This identity describes the immutable source snapshot, not the
        # requested review range.  Date coverage remains part of the request
        # fingerprint so a longer/shorter run can reuse matching reviews.
        identity = identity_provider()
        if identity is None:
            return None
        digest = automatic_digest(
            {
                "kind": "methodology_input_snapshot",
                "identity": identity,
            }
        )
    except Exception as error:
        diagnostics.append(
            "Automatic methodology data identity failed "
            f"({type(error).__name__}); review reuse was disabled."
        )
        return None
    if requested is not CacheMode.OFF:
        descriptor = workspace.save_frame(
            "methodology_snapshot_descriptor",
            pd.DataFrame({"snapshot_digest": [digest]}),
        )
        workspace.bind_artifact(
            stage=CacheStage.REVIEWS,
            cache_key=cache_key,
            name=_LEGACY_REVIEW_SNAPSHOT_BINDING,
            artifact=descriptor,
        )
    return _global_review_snapshot_evidence(
        spec,
        snapshot_digest=digest,
        scope_digest=cache_key,
    )


def _global_review_snapshot_evidence(
    spec: ResearchSpec,
    *,
    snapshot_digest: str,
    scope_digest: str,
) -> _ReviewSnapshotEvidence:
    return _ReviewSnapshotEvidence(
        records=tuple(
            _ReviewSnapshotRecord(
                reference_date=pd.Timestamp(review.reference_date).normalize(),
                effective_date=pd.Timestamp(review.effective_date).normalize(),
                snapshot_digest=snapshot_digest,
            )
            for review in spec.calendar.dates.itertuples(index=False)
        ),
        scope_digest=scope_digest,
    )


def _decode_review_snapshot_descriptor(frame: pd.DataFrame) -> str:
    if len(frame) != 1 or "snapshot_digest" not in frame:
        raise UnsafeCacheReuseError("cached methodology snapshot descriptor is invalid")
    digest = str(frame.iloc[0]["snapshot_digest"])
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise UnsafeCacheReuseError("cached methodology snapshot descriptor is invalid")
    return digest


def _simulation_snapshot(
    spec: ResearchSpec,
    *,
    workspace: WorkspaceRepository,
    requested: CacheMode,
    diagnostics: list[str],
) -> str | None:
    simulation = spec.simulation
    if simulation is None:
        return None
    try:
        provider = registry.get(simulation.market_data_provider_name)
        request = {
            **dict(simulation.provider_parameters),
            "start_date": simulation.start_date,
            "end_date": simulation.end_date,
        }
        snapshot = workspace_provider_snapshot_digest(
            workspace,
            provider,
            provider_name=simulation.market_data_provider_name,
            capability="load_daily_market_data",
            parameters=simulation.provider_parameters,
            request=request,
            mode=requested,
        )
        if snapshot is None:
            return None
        snapshot_digest, protocol = snapshot
        return automatic_digest(
            {
                "kind": "simulation_input_snapshot",
                "provider": automatic_provider_identity(
                    simulation.market_data_provider_name,
                    provider,
                    capability="load_daily_market_data",
                    parameters=simulation.provider_parameters,
                ),
                "snapshot_digest": snapshot_digest,
                "snapshot_protocol": protocol,
                "private_parameter_scope_digest": (
                    private_parameter_scope_digest(simulation.provider_parameters)
                ),
            }
        )
    except Exception as error:
        diagnostics.append(
            "Automatic provider snapshot identity failed "
            f"({type(error).__name__}); preflight reuse was disabled and "
            "loaded source data will be identified from canonical content."
        )
        return None


__all__ = [name for name in globals() if name.startswith("_")]
