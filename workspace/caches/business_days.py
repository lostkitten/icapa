"""Business-day calendar loading with optional workspace reuse."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ...data_sources.providers.registry import registry
from ..identity import (
    IdentityError,
    automatic_digest,
    automatic_provider_identity,
    dataframe_content_digest,
    safe_parameter_identity,
)
from ..repository import ManifestIntegrityError, WorkspaceRepository
from .models import CacheMode, CacheStage
from .source_contracts import (
    BUSINESS_DAY_BINDING_NAME as _BUSINESS_DAY_BINDING_NAME,
    BUSINESS_DAY_CAPABILITY as _BUSINESS_DAY_CAPABILITY,
    SCHEMA_VERSION as _SCHEMA_VERSION,
)
from .source_identity import (
    UnsafeCacheReuseError,
    private_parameter_scope_digest,
)
from .source_partitions import (
    canonical_business_days as _canonical_business_days,
)
from .source_snapshots import workspace_provider_snapshot_digest


@dataclass
class BusinessDayCacheLoader:
    """Load and verify an optional provider business-day calendar.

    The calendar is a defensive simulation input rather than a review-schedule
    authority. It is cached independently from market data so READ_ONLY runs
    can validate date completeness without calling the provider.
    """

    workspace: WorkspaceRepository
    provider_name: str
    provider_parameters: Mapping[str, Any]
    mode: CacheMode | str = CacheMode.OFF
    _records: list[Mapping[str, Any]] = field(
        init=False,
        default_factory=list,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, WorkspaceRepository):
            raise TypeError("workspace must be a research workspace")
        if not isinstance(self.provider_name, str) or not self.provider_name.strip():
            raise ValueError("provider_name must not be empty")
        self.provider_name = self.provider_name.strip().lower()
        self.provider_parameters = dict(self.provider_parameters)
        self.mode = CacheMode(self.mode)
        self._private_scope_digest = (
            None
            if self.mode is CacheMode.OFF
            else private_parameter_scope_digest(
                self.provider_parameters
            )
        )
        self._provider = registry.resolve(
            _BUSINESS_DAY_CAPABILITY,
            self.provider_name,
        )
        try:
            self._provider_identity = automatic_provider_identity(
                self.provider_name,
                self._provider,
                capability=_BUSINESS_DAY_CAPABILITY,
                parameters=self.provider_parameters,
            )
        except (IdentityError, OSError, TypeError, ValueError):
            self._provider_identity = None
            if self.mode is not CacheMode.OFF:
                raise UnsafeCacheReuseError(
                    "business-day caching requires a verifiable provider "
                    "implementation identity"
                )

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(item) for item in self._records)

    def load(
        self,
        *,
        calendar_id: str,
        start_date: object,
        end_date: object,
    ) -> pd.DatetimeIndex:
        if not isinstance(calendar_id, str) or not calendar_id.strip():
            raise ValueError("calendar_id must not be empty")
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if start > end:
            raise ValueError(
                "business-day start_date must not be after end_date"
            )
        request = {
            **self.provider_parameters,
            "calendar_id": calendar_id.strip(),
            "start_date": start,
            "end_date": end,
        }
        request_digest = automatic_digest(
            {
                "capability": _BUSINESS_DAY_CAPABILITY,
                "calendar_id": calendar_id.strip(),
                "start_date": start,
                "end_date": end,
                "provider_parameters": safe_parameter_identity(
                    self.provider_parameters
                ),
            }
        )
        snapshot_digest: str | None = None
        snapshot_protocol: str | None = None
        if self.mode is not CacheMode.OFF:
            snapshot = workspace_provider_snapshot_digest(
                self.workspace,
                self._provider,
                provider_name=self.provider_name,
                capability=_BUSINESS_DAY_CAPABILITY,
                parameters=self.provider_parameters,
                request=request,
                mode=self.mode,
                provider_identity=self._provider_identity,
            )
            if snapshot is not None:
                snapshot_digest, snapshot_protocol = snapshot

        cache_key = (
            None
            if snapshot_digest is None or self._provider_identity is None
            else self._cache_key(
                request_digest=request_digest,
                snapshot_digest=snapshot_digest,
            )
        )
        if self.mode in {CacheMode.REUSE, CacheMode.READ_ONLY}:
            if cache_key is None:
                if self.mode is CacheMode.READ_ONLY:
                    raise UnsafeCacheReuseError(
                        "READ_ONLY business-day access requires a provider "
                        "snapshot identity"
                    )
            else:
                reference = self._resolve(cache_key)
                if reference is not None:
                    frame = self.workspace.load_frame(reference)
                    days = _canonical_business_days(
                        frame,
                        start_date=start,
                        end_date=end,
                    )
                    self._record(
                        request_digest=request_digest,
                        snapshot_digest=snapshot_digest,
                        snapshot_protocol=snapshot_protocol,
                        content_digest=dataframe_content_digest(
                            pd.DataFrame({"business_date": days})
                        ),
                        rows=len(days),
                        cache_source="workspace",
                    )
                    return days
                if self.mode is CacheMode.READ_ONLY:
                    raise UnsafeCacheReuseError(
                        "READ_ONLY source-data cache is missing the required "
                        "business-day calendar"
                    )

        raw = self._provider.load_business_days(
            calendar_id=calendar_id.strip(),
            start_date=start,
            end_date=end,
            **self.provider_parameters,
        )
        days = _canonical_business_days(
            raw,
            start_date=start,
            end_date=end,
        )
        frame = pd.DataFrame({"business_date": days})
        content_digest = dataframe_content_digest(frame)
        if self.mode is not CacheMode.OFF:
            if cache_key is None:
                cache_key = self._cache_key(
                    request_digest=request_digest,
                    content_digest=content_digest,
                )
            reference = self.workspace.save_frame(
                "source_business_days",
                frame,
            )
            self.workspace.bind_artifact(
                stage=CacheStage.SOURCE_DATA,
                cache_key=cache_key,
                name=_BUSINESS_DAY_BINDING_NAME,
                artifact=reference,
            )
        self._record(
            request_digest=request_digest,
            snapshot_digest=snapshot_digest,
            snapshot_protocol=snapshot_protocol,
            content_digest=content_digest,
            rows=len(days),
            cache_source="provider",
        )
        return days

    def _cache_key(
        self,
        *,
        request_digest: str,
        snapshot_digest: str | None = None,
        content_digest: str | None = None,
    ) -> str:
        return automatic_digest(
            {
                "schema_version": _SCHEMA_VERSION,
                "kind": "canonical_business_days",
                "provider": self._provider_identity,
                "private_parameter_scope_digest": (
                    self._private_scope_digest
                ),
                "request_digest": request_digest,
                "snapshot_digest": snapshot_digest,
                "content_digest": content_digest,
            }
        )

    def _resolve(self, cache_key: str):
        try:
            return self.workspace.resolve_artifact(
                stage=CacheStage.SOURCE_DATA,
                cache_key=cache_key,
                name=_BUSINESS_DAY_BINDING_NAME,
            )
        except ManifestIntegrityError as exc:
            if (
                self.mode is CacheMode.REUSE
                and "does not exist in its workspace" in str(exc)
            ):
                return None
            raise

    def _record(
        self,
        *,
        request_digest: str,
        snapshot_digest: str | None,
        snapshot_protocol: str | None,
        content_digest: str,
        rows: int,
        cache_source: str,
    ) -> None:
        record: dict[str, Any] = {
            "input_type": "source_business_days",
            "provider_name": self.provider_name,
            "capability": _BUSINESS_DAY_CAPABILITY,
            "request_digest": request_digest,
            "content_digest": content_digest,
            "rows": int(rows),
            "cache_source": cache_source,
        }
        if snapshot_digest is not None:
            record["snapshot_digest"] = snapshot_digest
        if snapshot_protocol is not None:
            record["snapshot_protocol"] = snapshot_protocol
        self._records.append(record)



__all__ = ["BusinessDayCacheLoader"]
