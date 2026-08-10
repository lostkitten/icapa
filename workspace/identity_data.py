"""Automatic identities for providers, data snapshots, and runtimes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
import platform
from typing import Any

import pandas as pd

from .identity_canonical import (
    IdentityError,
    _is_secret_key,
    automatic_digest,
    canonicalize,
    dataframe_content_digest,
    safe_parameter_identity,
)
from .identity_components import (
    _dependency_lock_identity,
    automatic_component_identity,
)


def automatic_provider_identity(
    provider_name: str,
    provider: object,
    *,
    capability: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return safe automatic provider metadata for a manifest."""

    if not isinstance(provider_name, str) or not provider_name.strip():
        raise IdentityError("provider_name must not be empty")
    if not isinstance(capability, str) or not capability.strip():
        raise IdentityError("capability must not be empty")
    return {
        "provider_name": provider_name.strip().lower(),
        "capability": capability,
        "component": automatic_component_identity(provider),
        "parameters": safe_parameter_identity(parameters),
    }


def automatic_data_identity(
    *,
    provider_name: str,
    provider: object,
    capability: str,
    request: Mapping[str, Any],
    frame: pd.DataFrame,
    sort_by: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Identify one canonical provider response by its logical DataFrame content."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("automatic data identity requires a pandas DataFrame")
    return {
        "provider": automatic_provider_identity(
            provider_name,
            provider,
            capability=capability,
            parameters=request,
        ),
        "request_digest": automatic_digest(
            {
                str(key): canonicalize(value)
                for key, value in request.items()
                if not _is_secret_key(str(key))
            }
        ),
        "content_digest": dataframe_content_digest(frame, sort_by=sort_by),
        "rows": int(len(frame)),
        "columns": list(map(str, frame.columns)),
    }


def automatic_runtime_identity() -> tuple[dict[str, Any], ...]:
    """Collect calculation-relevant runtime versions automatically."""

    records: list[dict[str, Any]] = [
        {
            "name": "python",
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        {
            "name": "platform",
            "system": platform.system(),
            "machine": platform.machine(),
        },
    ]
    for package_name in ("icapa", "numpy", "pandas", "scipy", "pyarrow", "osqp"):
        try:
            version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
        records.append({"name": package_name, "version": version})
    dependency_lock = _dependency_lock_identity(Path(__file__).resolve())
    if dependency_lock is not None:
        records.append(
            {
                "name": "dependency_lock",
                "digest": dependency_lock,
            }
        )
    return tuple(records)


__all__ = [
    "automatic_data_identity",
    "automatic_provider_identity",
    "automatic_runtime_identity",
]
