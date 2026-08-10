"""Safe workspace caching for canonical provider source-data partitions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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
from ..locking import exclusive_file_lock as _exclusive_file_lock
from ..repository import ManifestIntegrityError, WorkspaceRepository
from .business_days import BusinessDayCacheLoader
from .models import CacheMode, CacheStage
from .source_contracts import (
    BINDING_NAME as _BINDING_NAME,
    CAPABILITY as _CAPABILITY,
    MONTH_COVERAGE_BINDING_NAME as _MONTH_COVERAGE_BINDING_NAME,
    SCHEMA_VERSION as _SCHEMA_VERSION,
)
from .source_identity import (
    UnsafeCacheReuseError,
    private_parameter_scope_digest,
)
from .source_partitions import (
    canonical_partition as _canonical_partition,
    decode_month_coverage_descriptors as _decode_month_coverage_descriptors,
    select_covering_descriptors as _select_covering_descriptors,
)
from .source_snapshots import (
    provider_snapshot_digest,
    workspace_provider_snapshot_digest,
)


@dataclass
class SourceDataCacheLoader:
    """Load canonical market-data partitions through an optional v2 cache.

    Cache keys depend only on the provider implementation, safe provider
    parameters, the exact canonical data request, and the provider snapshot.
    They deliberately exclude the index, methodology, simulation parameters,
    and research scenario so identical source partitions can be shared.
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
    _verified_references: dict[tuple[str, str | None], object] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _verified_frames: dict[str, pd.DataFrame] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _verified_frame_scopes: dict[
        str,
        tuple[
            tuple[Any, ...],
            pd.Timestamp,
            pd.Timestamp,
            str | None,
        ],
    ] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _verified_frame_sources: dict[str, str] = field(
        init=False,
        default_factory=dict,
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
        self._provider = registry.resolve(_CAPABILITY, self.provider_name)
        try:
            self._provider_identity = automatic_provider_identity(
                self.provider_name,
                self._provider,
                capability=_CAPABILITY,
                parameters=self.provider_parameters,
            )
        except (IdentityError, OSError, TypeError, ValueError):
            self._provider_identity = None
            if self.mode is not CacheMode.OFF:
                raise UnsafeCacheReuseError(
                    "source-data caching requires a verifiable provider "
                    "implementation identity"
                )

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        """Return sanitized identities for data actually consumed by simulation."""

        return tuple(dict(item) for item in self._records)

    def load(
        self,
        *,
        instrument_ids: Iterable[Any],
        start_date: object,
        end_date: object,
    ) -> pd.DataFrame:
        """Load one exact daily market-data partition."""

        instruments = tuple(sorted(set(instrument_ids), key=str))
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if start > end:
            raise ValueError("source-data start_date must not be after end_date")
        request = {
            **self.provider_parameters,
            "instrument_ids": instruments,
            "start_date": start,
            "end_date": end,
        }
        request_digest = automatic_digest(
            {
                "capability": _CAPABILITY,
                "instrument_ids": instruments,
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
            try:
                snapshot = workspace_provider_snapshot_digest(
                    self.workspace,
                    self._provider,
                    provider_name=self.provider_name,
                    capability=_CAPABILITY,
                    parameters=self.provider_parameters,
                    request=request,
                    mode=self.mode,
                    provider_identity=self._provider_identity,
                )
            except Exception:
                if self.mode is CacheMode.READ_ONLY:
                    raise
                snapshot = None
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
        same_run_frame = self._same_run_frame(
            request_digest=request_digest,
            instruments=instruments,
            start_date=start,
            end_date=end,
            snapshot_digest=snapshot_digest,
        )
        if same_run_frame is not None:
            same_run_data, same_run_source = same_run_frame
            frame = _canonical_partition(
                same_run_data,
                instruments=instruments,
                start_date=start,
                end_date=end,
            )
            self._record(
                request_digest=request_digest,
                snapshot_digest=snapshot_digest,
                snapshot_protocol=snapshot_protocol,
                content_digest=dataframe_content_digest(frame),
                rows=len(frame),
                cache_source=same_run_source,
                start_date=start,
                end_date=end,
                instrument_ids=instruments,
            )
            self._remember_frame(
                request_digest=request_digest,
                instruments=instruments,
                start_date=start,
                end_date=end,
                frame=frame,
                cache_source=same_run_source,
                snapshot_digest=snapshot_digest,
            )
            return frame
        same_run_reference = self._verified_references.get(
            (request_digest, snapshot_digest)
        )
        if same_run_reference is not None:
            frame = self.workspace.load_frame(same_run_reference)
            frame = _canonical_partition(
                frame,
                instruments=instruments,
                start_date=start,
                end_date=end,
            )
            logical_content_digest = dataframe_content_digest(frame)
            self._record(
                request_digest=request_digest,
                snapshot_digest=snapshot_digest,
                snapshot_protocol=snapshot_protocol,
                content_digest=logical_content_digest,
                rows=len(frame),
                cache_source="workspace",
                start_date=start,
                end_date=end,
                instrument_ids=instruments,
            )
            self._remember_frame(
                request_digest=request_digest,
                instruments=instruments,
                start_date=start,
                end_date=end,
                frame=frame,
                cache_source="workspace",
                snapshot_digest=snapshot_digest,
            )
            return frame
        if self.mode in {CacheMode.REUSE, CacheMode.READ_ONLY}:
            if cache_key is None:
                if self.mode is CacheMode.READ_ONLY:
                    raise UnsafeCacheReuseError(
                        "READ_ONLY source-data access requires a provider "
                        "snapshot identity"
                    )
            else:
                reference = self._resolve(cache_key)
                if reference is not None:
                    frame = self.workspace.load_frame(reference)
                    frame = _canonical_partition(
                        frame,
                        instruments=instruments,
                        start_date=start,
                        end_date=end,
                    )
                    logical_content_digest = dataframe_content_digest(frame)
                    self._record(
                        request_digest=request_digest,
                        snapshot_digest=snapshot_digest,
                        snapshot_protocol=snapshot_protocol,
                        content_digest=logical_content_digest,
                        rows=len(frame),
                        cache_source="workspace",
                        start_date=start,
                        end_date=end,
                        instrument_ids=instruments,
                    )
                    self._verified_references[
                        (request_digest, snapshot_digest)
                    ] = reference
                    self._remember_frame(
                        request_digest=request_digest,
                        instruments=instruments,
                        start_date=start,
                        end_date=end,
                        frame=frame,
                        cache_source="workspace",
                        snapshot_digest=snapshot_digest,
                    )
                    return frame
                containing = self._load_containing_month_partition(
                    instruments=instruments,
                    start_date=start,
                    end_date=end,
                    snapshot_digest=snapshot_digest,
                )
                if containing is not None:
                    frame = containing
                    logical_content_digest = dataframe_content_digest(frame)
                    self._record(
                        request_digest=request_digest,
                        snapshot_digest=snapshot_digest,
                        snapshot_protocol=snapshot_protocol,
                        content_digest=logical_content_digest,
                        rows=len(frame),
                        cache_source="workspace_containing_range",
                        start_date=start,
                        end_date=end,
                        instrument_ids=instruments,
                    )
                    self._remember_frame(
                        request_digest=request_digest,
                        instruments=instruments,
                        start_date=start,
                        end_date=end,
                        frame=frame,
                        cache_source="workspace_containing_range",
                        snapshot_digest=snapshot_digest,
                    )
                    return frame
                if self.mode is CacheMode.READ_ONLY:
                    raise UnsafeCacheReuseError(
                        "READ_ONLY source-data cache is missing a required "
                        "market-data partition"
                    )

        raw = self._provider.load_daily_market_data(
            instrument_ids=instruments,
            start_date=start,
            end_date=end,
            **self.provider_parameters,
        )
        frame = _canonical_partition(
            raw,
            instruments=instruments,
            start_date=start,
            end_date=end,
        )
        content_digest = dataframe_content_digest(frame)
        if self.mode is not CacheMode.OFF:
            if cache_key is None:
                cache_key = self._cache_key(
                    request_digest=request_digest,
                    content_digest=content_digest,
                )
            reference = self.workspace.save_frame(
                "source_daily_market_data",
                frame,
            )
            self.workspace.bind_artifact(
                stage=CacheStage.SOURCE_DATA,
                cache_key=cache_key,
                name=_BINDING_NAME,
                artifact=reference,
            )
            self._verified_references[
                (request_digest, snapshot_digest)
            ] = reference
            if snapshot_digest is not None:
                self._record_month_coverage(
                    instruments=instruments,
                    start_date=start,
                    end_date=end,
                    snapshot_digest=snapshot_digest,
                    cache_key=cache_key,
                    reference=reference,
                )
        # A downstream simulation cache may require a content preflight even
        # when SOURCE_DATA is OFF. Retain verified partitions only for this
        # loader's lifetime so the immediately following calculation does not
        # call the provider a second time. This is not a reusable artifact.
        self._remember_frame(
            request_digest=request_digest,
            instruments=instruments,
            start_date=start,
            end_date=end,
            frame=frame,
            cache_source="provider",
            snapshot_digest=snapshot_digest,
        )
        self._record(
            request_digest=request_digest,
            snapshot_digest=snapshot_digest,
            snapshot_protocol=snapshot_protocol,
            content_digest=content_digest,
            rows=len(frame),
            cache_source="provider",
            start_date=start,
            end_date=end,
            instrument_ids=instruments,
        )
        return frame

    def _remember_frame(
        self,
        *,
        request_digest: str,
        instruments: tuple[Any, ...],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        frame: pd.DataFrame,
        cache_source: str,
        snapshot_digest: str | None,
    ) -> None:
        self._verified_frames[request_digest] = frame.copy(deep=True)
        self._verified_frame_scopes[request_digest] = (
            instruments,
            start_date,
            end_date,
            snapshot_digest,
        )
        self._verified_frame_sources[request_digest] = cache_source

    def _same_run_frame(
        self,
        *,
        request_digest: str,
        instruments: tuple[Any, ...],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        snapshot_digest: str | None,
    ) -> tuple[pd.DataFrame, str] | None:
        exact = self._verified_frames.get(request_digest)
        exact_scope = self._verified_frame_scopes.get(request_digest)
        if (
            exact is not None
            and exact_scope is not None
            and exact_scope[3] == snapshot_digest
        ):
            return (
                exact.copy(deep=True),
                self._verified_frame_sources.get(
                    request_digest,
                    "memory_same_run",
                ),
            )

        candidates = [
            (scope_start, scope_end, key)
            for key, (
                scope_instruments,
                scope_start,
                scope_end,
                scope_snapshot,
            )
            in self._verified_frame_scopes.items()
            if scope_instruments == instruments
            and scope_snapshot == snapshot_digest
            and scope_end >= start_date
            and scope_start <= end_date
        ]
        selected: list[str] = []
        cursor = start_date
        while cursor <= end_date:
            covering = [
                item
                for item in candidates
                if item[0] <= cursor <= item[1]
            ]
            if not covering:
                return None
            _, covered_end, key = max(
                covering,
                key=lambda item: (item[1], -item[0].value),
            )
            selected.append(key)
            cursor = covered_end + pd.Timedelta(days=1)
        combined = pd.concat(
            [self._verified_frames[key] for key in selected],
            ignore_index=True,
        )
        combined = combined.drop_duplicates(
            subset=["business_date", "instrument_id"],
            keep="last",
        )
        sources = {
            self._verified_frame_sources.get(
                key,
                "memory_same_run",
            )
            for key in selected
        }
        return (
            _canonical_partition(
                combined,
                instruments=instruments,
                start_date=start_date,
                end_date=end_date,
            ),
            next(iter(sources)) if len(sources) == 1 else "memory_same_run",
        )

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
                "kind": "canonical_source_data",
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
                name=_BINDING_NAME,
            )
        except ManifestIntegrityError as exc:
            if (
                self.mode is CacheMode.REUSE
                and "does not exist in its workspace" in str(exc)
            ):
                return None
            raise

    def _load_containing_month_partition(
        self,
        *,
        instruments: tuple[Any, ...],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        snapshot_digest: str | None,
    ):
        """Load a verified same-month artifact that contains the request.

        Immutable source tables are shared across simulation scenarios. A
        shorter simulation can therefore rebuild only its final partial
        segment from an already verified longer source partition, without
        calling the provider or persisting daily constituent weights in every
        simulation artifact.
        """

        coverage_key = self._month_coverage_key(
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            snapshot_digest=snapshot_digest,
        )
        if coverage_key is None:
            return None
        try:
            descriptor_reference = self.workspace.resolve_artifact(
                stage=CacheStage.SOURCE_DATA,
                cache_key=coverage_key,
                name=_MONTH_COVERAGE_BINDING_NAME,
            )
        except ManifestIntegrityError as exc:
            if (
                self.mode is CacheMode.REUSE
                and "does not exist in its workspace" in str(exc)
            ):
                return None
            raise
        if descriptor_reference is None:
            return None
        descriptors = _decode_month_coverage_descriptors(
            self.workspace.load_frame(descriptor_reference)
        )
        selected = _select_covering_descriptors(
            descriptors,
            start_date=start_date,
            end_date=end_date,
        )
        if not selected:
            return None
        frames: list[pd.DataFrame] = []
        for descriptor in selected:
            reference = self._resolve(descriptor["cache_key"])
            if reference is None:
                return None
            if (
                reference.content_digest != descriptor["content_digest"]
                or reference.file_checksum != descriptor["file_checksum"]
            ):
                raise UnsafeCacheReuseError(
                    "source-data month coverage points to a different artifact"
                )
            frames.append(self.workspace.load_frame(reference))
        combined = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["business_date", "instrument_id"],
            keep="last",
        )
        return _canonical_partition(
            combined,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
        )

    def _record_month_coverage(
        self,
        *,
        instruments: tuple[Any, ...],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        snapshot_digest: str,
        cache_key: str,
        reference,
    ) -> None:
        coverage_key = self._month_coverage_key(
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            snapshot_digest=snapshot_digest,
        )
        if coverage_key is None:
            return
        lock_path = self.workspace.workspace_path.joinpath(
            ".locks",
            "source-data-month-coverage",
            f"{coverage_key}.lock",
        )
        with _exclusive_file_lock(lock_path):
            existing = self.workspace.resolve_artifact(
                stage=CacheStage.SOURCE_DATA,
                cache_key=coverage_key,
                name=_MONTH_COVERAGE_BINDING_NAME,
            )
            if existing is not None:
                descriptors = _decode_month_coverage_descriptors(
                    self.workspace.load_frame(existing)
                )
                descriptors = [
                    descriptor
                    for descriptor in descriptors
                    if descriptor["cache_key"] != cache_key
                ]
                if self.mode is CacheMode.REFRESH:
                    descriptors = [
                        descriptor
                        for descriptor in descriptors
                        if descriptor["end_date"] < start_date
                        or descriptor["start_date"] > end_date
                    ]
                elif _select_covering_descriptors(
                    descriptors,
                    start_date=start_date,
                    end_date=end_date,
                ):
                    return
            else:
                descriptors = []
            descriptors = [
                descriptor
                for descriptor in descriptors
                if not (
                    start_date <= descriptor["start_date"]
                    and end_date >= descriptor["end_date"]
                )
            ]
            descriptors.append(
                {
                    "start_date": start_date,
                    "end_date": end_date,
                    "cache_key": cache_key,
                    "content_digest": reference.content_digest,
                    "file_checksum": reference.file_checksum,
                }
            )
            descriptors.sort(
                key=lambda item: (
                    item["start_date"],
                    item["end_date"],
                    item["cache_key"],
                )
            )
            descriptor_reference = self.workspace.save_frame(
                "source_daily_market_data_coverage",
                pd.DataFrame(
                    {
                        "schema_version": [
                            _SCHEMA_VERSION for _ in descriptors
                        ],
                        "start_date": [
                            item["start_date"] for item in descriptors
                        ],
                        "end_date": [
                            item["end_date"] for item in descriptors
                        ],
                        "cache_key": [
                            item["cache_key"] for item in descriptors
                        ],
                        "content_digest": [
                            item["content_digest"] for item in descriptors
                        ],
                        "file_checksum": [
                            item["file_checksum"] for item in descriptors
                        ],
                    }
                ),
            )
            self.workspace.bind_artifact(
                stage=CacheStage.SOURCE_DATA,
                cache_key=coverage_key,
                name=_MONTH_COVERAGE_BINDING_NAME,
                artifact=descriptor_reference,
            )

    def _month_coverage_key(
        self,
        *,
        instruments: tuple[Any, ...],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        snapshot_digest: str | None,
    ) -> str | None:
        if (
            snapshot_digest is None
            or self._provider_identity is None
            or self._private_scope_digest is None
            or start_date.to_period("M") != end_date.to_period("M")
        ):
            return None
        return automatic_digest(
            {
                "schema_version": _SCHEMA_VERSION,
                "kind": "canonical_source_data_month_coverage",
                "provider": self._provider_identity,
                "private_parameter_scope_digest": (
                    self._private_scope_digest
                ),
                "instrument_set_digest": automatic_digest(
                    list(instruments)
                ),
                "snapshot_digest": snapshot_digest,
                "calendar_month": start_date.strftime("%Y-%m"),
            }
        )

    def _record(
        self,
        *,
        request_digest: str,
        snapshot_digest: str | None,
        snapshot_protocol: str | None,
        content_digest: str,
        rows: int,
        cache_source: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        instrument_ids: tuple[Any, ...],
    ) -> None:
        record: dict[str, Any] = {
            "input_type": "source_daily_market_data",
            "provider_name": self.provider_name,
            "capability": _CAPABILITY,
            "request_digest": request_digest,
            "content_digest": content_digest,
            "rows": int(rows),
            "cache_source": cache_source,
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "instrument_set_digest": automatic_digest(
                list(instrument_ids)
            ),
        }
        if snapshot_digest is not None:
            record["snapshot_digest"] = snapshot_digest
        if snapshot_protocol is not None:
            record["snapshot_protocol"] = snapshot_protocol
        self._records.append(record)




__all__ = [
    "BusinessDayCacheLoader",
    "SourceDataCacheLoader",
    "UnsafeCacheReuseError",
    "private_parameter_scope_digest",
    "provider_snapshot_digest",
    "workspace_provider_snapshot_digest",
]
