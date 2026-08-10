"""Provider snapshot protocols and persisted snapshot descriptors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ..identity import (
    automatic_digest,
    automatic_provider_identity,
    safe_parameter_identity,
)
from ..repository import WorkspaceRepository
from .models import CacheMode, CacheStage
from .source_contracts import (
    SCHEMA_VERSION as _SCHEMA_VERSION,
    SNAPSHOT_BINDING_NAME as _SNAPSHOT_BINDING_NAME,
)
from .source_identity import (
    UnsafeCacheReuseError,
    private_parameter_scope_digest,
)
from .source_partitions import decode_snapshot_frame as _decode_snapshot_frame


def provider_snapshot_digest(
    provider: object,
    *,
    capability: str,
    request: Mapping[str, Any],
) -> tuple[str, str] | None:
    """Return a secret-safe digest using the preferred snapshot protocol.

    The snapshot payload itself is never returned or persisted. The v1
    ``research_data_identity`` hook remains supported when the standard
    ``describe_snapshot`` protocol is unavailable.
    """

    candidates = (
        ("describe_snapshot", getattr(provider, "describe_snapshot", None)),
        (
            "research_data_identity",
            getattr(provider, "research_data_identity", None),
        ),
    )
    for protocol, method in candidates:
        if not callable(method):
            continue
        identity = method(capability=capability, request=dict(request))
        if identity is None:
            continue
        requested_scope = request.get("instrument_scope")
        if requested_scope == "all_instruments":
            if (
                not isinstance(identity, Mapping)
                or identity.get("instrument_scope") != requested_scope
            ):
                raise UnsafeCacheReuseError(
                    "dataset-level snapshot identity must acknowledge "
                    "instrument_scope='all_instruments'"
                )
        return (
            automatic_digest(
                {
                    "protocol": protocol,
                    "identity": identity,
                }
            ),
            protocol,
        )
    return None


def workspace_provider_snapshot_digest(
    workspace: WorkspaceRepository,
    provider: object,
    *,
    provider_name: str,
    capability: str,
    parameters: Mapping[str, Any],
    request: Mapping[str, Any],
    mode: CacheMode | str,
    provider_identity: Mapping[str, Any] | None = None,
) -> tuple[str, str] | None:
    """Resolve snapshot identity without provider calls in READ_ONLY mode."""

    selected_mode = CacheMode(mode)
    identity = (
        dict(provider_identity)
        if provider_identity is not None
        else automatic_provider_identity(
            provider_name,
            provider,
            capability=capability,
            parameters=parameters,
        )
    )
    cache_key = automatic_digest(
        {
            "schema_version": _SCHEMA_VERSION,
            "kind": "provider_snapshot_descriptor",
            "provider": identity,
            "capability": capability,
            "request": safe_parameter_identity(request),
            "private_request_scope_digest": (
                private_parameter_scope_digest(request)
            ),
        }
    )
    if selected_mode is CacheMode.READ_ONLY:
        reference = workspace.resolve_artifact(
            stage=CacheStage.SOURCE_DATA,
            cache_key=cache_key,
            name=_SNAPSHOT_BINDING_NAME,
        )
        if reference is None:
            return None
        frame = workspace.load_frame(reference)
        return _decode_snapshot_frame(frame)

    snapshot = provider_snapshot_digest(
        provider,
        capability=capability,
        request=request,
    )
    if snapshot is None or selected_mode is CacheMode.OFF:
        return snapshot
    snapshot_digest, snapshot_protocol = snapshot
    reference = workspace.save_frame(
        "provider_snapshot_descriptor",
        pd.DataFrame(
            {
                "snapshot_digest": [snapshot_digest],
                "snapshot_protocol": [snapshot_protocol],
            }
        ),
    )
    workspace.bind_artifact(
        stage=CacheStage.SOURCE_DATA,
        cache_key=cache_key,
        name=_SNAPSHOT_BINDING_NAME,
        artifact=reference,
    )
    return snapshot


__all__ = [
    "provider_snapshot_digest",
    "workspace_provider_snapshot_digest",
]
