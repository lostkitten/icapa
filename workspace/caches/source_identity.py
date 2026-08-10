"""Secret-safe private scoping for provider cache identities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...data_sources.provenance import private_parameter_digest


class UnsafeCacheReuseError(RuntimeError):
    """Raised when read-only reuse lacks verifiable source evidence."""


def private_parameter_scope_digest(
    parameters: Mapping[str, Any] | None,
) -> str:
    """Hash private provider scope for internal cache keys only.

    Unlike ``safe_parameter_identity``, this digest intentionally depends on
    credential, endpoint, and connection values. Only the digest may be
    persisted; callers must never place the source mapping in manifests,
    reports, artifact metadata, or diagnostics.
    """

    try:
        return private_parameter_digest(parameters)
    except (OSError, TypeError, ValueError) as exc:
        raise UnsafeCacheReuseError(
            "provider parameters cannot be safely scoped for cache reuse"
        ) from exc




__all__ = [
    "UnsafeCacheReuseError",
    "private_parameter_scope_digest",
]
