"""Immutable workspace caching for analytics plugin results."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import pandas as pd

from ...analytics.plugins import AnalyticsRunResult
from ..artifacts import ArtifactError as WorkspaceArtifactError
from ..identity import automatic_digest, canonicalize
from ..manifests import ArtifactRef
from ..repository import ManifestIntegrityError, WorkspaceRepository
from .analytics_codec import (
    decode_commit as _decode_commit,
    decode_value as _decode_value,
    json_bytes as _json_bytes,
    json_digest as _json_digest,
    prepare_result as _prepare_result,
    restore_result as _restore_result,
)
from .analytics_contracts import (
    COMMIT_BINDING as _COMMIT_BINDING,
    MAX_COMMIT_BYTES as _MAX_COMMIT_BYTES,
    SCHEMA_VERSION as _SCHEMA_VERSION,
    SERIES_COLUMN as _SERIES_COLUMN,
    TABLE_BINDING_PREFIX as _TABLE_BINDING_PREFIX,
    AnalyticsWorkspaceCacheCollisionError,
    AnalyticsWorkspaceCacheError,
    AnalyticsWorkspaceCacheIntegrityError,
    AnalyticsWorkspaceCacheMissError,
    AnalyticsWorkspaceCacheSerializationError,
)
from .analytics_frames import (
    restore_pandas_schema as _restore_pandas_schema,
    stable_parquet_frame as _stable_parquet_frame,
    unique_artifacts as _unique_artifacts,
)
from .models import CacheMode, CacheStage


class AnalyticsCacheSource(str, Enum):
    """How an analytics result was obtained."""

    EXECUTED = "executed"
    WORKSPACE = "workspace"


@dataclass(frozen=True, slots=True)
class AnalyticsCacheIdentity:
    """Caller-supplied deterministic inputs for one analytics calculation.

    Callers should include the analytics specification and digests of every
    backtest, simulation, research-input, and plugin implementation that can
    affect the result. The values are canonicalized and secret-like mapping
    keys are redacted by the common identity layer before the cache key is
    calculated.
    """

    inputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, Mapping) or not self.inputs:
            raise ValueError("analytics cache identity inputs must not be empty")
        try:
            safe_inputs = canonicalize(dict(self.inputs))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "analytics cache identity inputs must be deterministic"
            ) from exc
        if not isinstance(safe_inputs, dict):
            raise ValueError("analytics cache identity inputs must be a mapping")
        object.__setattr__(
            self,
            "inputs",
            MappingProxyType(safe_inputs),
        )

    @classmethod
    def from_inputs(cls, **inputs: Any) -> "AnalyticsCacheIdentity":
        """Build an identity from named deterministic inputs."""

        return cls(inputs)

    @property
    def cache_key(self) -> str:
        """Return the deterministic analytics-stage SHA-256 key."""

        return automatic_digest(
            {
                "schema_version": _SCHEMA_VERSION,
                "stage": CacheStage.ANALYTICS.value,
                "inputs": dict(self.inputs),
            }
        )


@dataclass(frozen=True, slots=True)
class AnalyticsCacheOutcome:
    """One calculated or verified workspace analytics result."""

    result: AnalyticsRunResult
    cache_key: str
    source: AnalyticsCacheSource
    artifacts: tuple[ArtifactRef, ...] = ()

    @property
    def from_cache(self) -> bool:
        """Whether calculation was skipped because a verified result existed."""

        return self.source is AnalyticsCacheSource.WORKSPACE

class AnalyticsWorkspaceCache:
    """Persist complete :class:`AnalyticsRunResult` values in a workspace.

    ``execute`` is the primary integration API. It applies the selected cache
    mode, calls the supplied zero-argument calculation only when required, and
    returns the result plus immutable artifact references suitable for a run
    manifest.
    """

    def __init__(
        self,
        workspace: WorkspaceRepository | str,
        *,
        mode: CacheMode | str = CacheMode.REUSE,
    ) -> None:
        self.workspace = (
            WorkspaceRepository.open(workspace)
            if isinstance(workspace, str)
            else workspace
        )
        if not isinstance(self.workspace, WorkspaceRepository):
            raise TypeError(
                "workspace must be a WorkspaceRepository or workspace name"
            )
        self.mode = CacheMode(mode)

    def execute(
        self,
        identity: AnalyticsCacheIdentity | Mapping[str, Any],
        calculation: Callable[[], AnalyticsRunResult],
    ) -> AnalyticsCacheOutcome:
        """Load a verified result or execute and optionally persist analytics."""

        if not callable(calculation):
            raise TypeError("calculation must be callable")
        selected_identity = _coerce_identity(identity)
        if self.mode in {CacheMode.REUSE, CacheMode.READ_ONLY}:
            cached = self._load_committed(selected_identity.cache_key)
            if cached is not None:
                return cached
            if self.mode is CacheMode.READ_ONLY:
                raise AnalyticsWorkspaceCacheMissError(
                    "READ_ONLY analytics cache is missing a required result"
                )

        result = calculation()
        if not isinstance(result, AnalyticsRunResult):
            raise TypeError("analytics calculation must return AnalyticsRunResult")
        if self.mode is CacheMode.OFF:
            return AnalyticsCacheOutcome(
                result=result,
                cache_key=selected_identity.cache_key,
                source=AnalyticsCacheSource.EXECUTED,
            )
        return self._save(
            selected_identity.cache_key,
            result,
            allow_rebind=self.mode is CacheMode.REFRESH,
        )

    def load(
        self,
        identity: AnalyticsCacheIdentity | Mapping[str, Any],
    ) -> AnalyticsCacheOutcome | None:
        """Load a complete result according to this adapter's cache mode."""

        selected_identity = _coerce_identity(identity)
        if self.mode in {CacheMode.OFF, CacheMode.REFRESH}:
            return None
        outcome = self._load_committed(selected_identity.cache_key)
        if outcome is None and self.mode is CacheMode.READ_ONLY:
            raise AnalyticsWorkspaceCacheMissError(
                "READ_ONLY analytics cache is missing a required result"
            )
        return outcome

    def save(
        self,
        identity: AnalyticsCacheIdentity | Mapping[str, Any],
        result: AnalyticsRunResult,
    ) -> AnalyticsCacheOutcome:
        """Persist a complete result unless writes are disabled."""

        if not isinstance(result, AnalyticsRunResult):
            raise TypeError("result must be an AnalyticsRunResult")
        selected_identity = _coerce_identity(identity)
        if self.mode is CacheMode.READ_ONLY:
            raise AnalyticsWorkspaceCacheMissError(
                "READ_ONLY analytics cache cannot save a calculated result"
            )
        if self.mode is CacheMode.OFF:
            return AnalyticsCacheOutcome(
                result=result,
                cache_key=selected_identity.cache_key,
                source=AnalyticsCacheSource.EXECUTED,
            )
        return self._save(
            selected_identity.cache_key,
            result,
            allow_rebind=self.mode is CacheMode.REFRESH,
        )

    def _save(
        self,
        cache_key: str,
        result: AnalyticsRunResult,
        *,
        allow_rebind: bool,
    ) -> AnalyticsCacheOutcome:
        prepared = _prepare_result(result)
        if not allow_rebind:
            existing = self._load_committed(cache_key)
            if existing is not None:
                existing_digest = _prepare_result(existing.result).result_digest
                if existing_digest != prepared.result_digest:
                    raise AnalyticsWorkspaceCacheCollisionError(
                        "analytics cache identity is already bound to a "
                        "different result"
                    )
                return existing

        for position, table in enumerate(prepared.tables):
            table.binding = (
                f"{_TABLE_BINDING_PREFIX}-"
                f"{prepared.result_digest}-{position:04d}"
            )
            table.reference = self.workspace.save_frame(
                "analytics_table",
                _stable_parquet_frame(table.frame),
            )
            self.workspace.bind_artifact(
                stage=CacheStage.ANALYTICS,
                cache_key=cache_key,
                name=table.binding,
                artifact=table.reference,
            )

        if not allow_rebind:
            concurrent = self._load_committed(cache_key)
            if concurrent is not None:
                concurrent_digest = _prepare_result(
                    concurrent.result
                ).result_digest
                if concurrent_digest != prepared.result_digest:
                    raise AnalyticsWorkspaceCacheCollisionError(
                        "analytics cache identity was concurrently bound to a "
                        "different result"
                    )
                return concurrent

        unsigned_payload = {
            "schema_version": _SCHEMA_VERSION,
            "cache_key": cache_key,
            "result_digest": prepared.result_digest,
            "result": prepared.payload,
            "tables": [table.commit_entry() for table in prepared.tables],
        }
        payload = dict(unsigned_payload)
        payload["metadata_checksum"] = _json_digest(unsigned_payload)
        encoded = _json_bytes(payload)
        if len(encoded) > _MAX_COMMIT_BYTES:
            raise AnalyticsWorkspaceCacheSerializationError(
                "analytics cache commit metadata exceeds one megabyte"
            )
        commit_reference = self.workspace.save_frame(
            "analytics_commit",
            pd.DataFrame({"payload_json": [encoded.decode("utf-8")]}),
        )
        self.workspace.bind_artifact(
            stage=CacheStage.ANALYTICS,
            cache_key=cache_key,
            name=_COMMIT_BINDING,
            artifact=commit_reference,
        )
        return AnalyticsCacheOutcome(
            result=result,
            cache_key=cache_key,
            source=AnalyticsCacheSource.EXECUTED,
            artifacts=_unique_artifacts(
                [
                    *(
                        table.reference
                        for table in prepared.tables
                        if table.reference is not None
                    ),
                    commit_reference,
                ]
            ),
        )

    def _load_committed(
        self,
        cache_key: str,
    ) -> AnalyticsCacheOutcome | None:
        try:
            commit_reference = self.workspace.resolve_artifact(
                stage=CacheStage.ANALYTICS,
                cache_key=cache_key,
                name=_COMMIT_BINDING,
            )
        except (ManifestIntegrityError, WorkspaceArtifactError, OSError) as exc:
            raise AnalyticsWorkspaceCacheIntegrityError(
                "analytics cache commit reference is invalid"
            ) from exc
        if commit_reference is None:
            return None

        try:
            commit_frame = self.workspace.load_frame(commit_reference)
            payload = _decode_commit(commit_frame, expected_key=cache_key)
            tables, references = self._restore_tables(
                cache_key,
                payload["tables"],
            )
            result = _restore_result(payload["result"], tables)
            observed_digest = _prepare_result(result).result_digest
            if observed_digest != payload["result_digest"]:
                raise AnalyticsWorkspaceCacheIntegrityError(
                    "analytics result digest does not match its commit"
                )
            return AnalyticsCacheOutcome(
                result=result,
                cache_key=cache_key,
                source=AnalyticsCacheSource.WORKSPACE,
                artifacts=_unique_artifacts(
                    [*references, commit_reference]
                ),
            )
        except AnalyticsWorkspaceCacheIntegrityError:
            raise
        except AnalyticsWorkspaceCacheSerializationError as exc:
            raise AnalyticsWorkspaceCacheIntegrityError(
                "analytics cache commit contains unsupported metadata"
            ) from exc
        except (ManifestIntegrityError, WorkspaceArtifactError, OSError) as exc:
            raise AnalyticsWorkspaceCacheIntegrityError(
                "analytics cache artifact is missing or corrupt"
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalyticsWorkspaceCacheIntegrityError(
                "analytics cache commit has an invalid schema"
            ) from exc

    def _restore_tables(
        self,
        cache_key: str,
        entries: object,
    ) -> tuple[dict[str, pd.DataFrame | pd.Series], list[ArtifactRef]]:
        if not isinstance(entries, list):
            raise AnalyticsWorkspaceCacheIntegrityError(
                "analytics cache table entries must be a list"
            )
        values: dict[str, pd.DataFrame | pd.Series] = {}
        references: list[ArtifactRef] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise AnalyticsWorkspaceCacheIntegrityError(
                    "analytics cache table entry must be an object"
                )
            logical_id = entry["logical_id"]
            binding = entry["binding"]
            value_kind = entry["value_kind"]
            if (
                not isinstance(logical_id, str)
                or not isinstance(binding, str)
                or value_kind not in {"dataframe", "series"}
            ):
                raise AnalyticsWorkspaceCacheIntegrityError(
                    "analytics cache table entry has invalid identifiers"
                )
            if logical_id in values:
                raise AnalyticsWorkspaceCacheIntegrityError(
                    "analytics cache contains duplicate logical table IDs"
                )
            expected = ArtifactRef.from_dict(entry["artifact"])
            reference = self.workspace.resolve_artifact(
                stage=CacheStage.ANALYTICS,
                cache_key=cache_key,
                name=binding,
            )
            if reference is None:
                raise AnalyticsWorkspaceCacheIntegrityError(
                    "analytics cache references a missing table binding"
                )
            if reference != expected:
                raise AnalyticsWorkspaceCacheIntegrityError(
                    "analytics table binding does not match its commit"
                )
            frame = self.workspace.load_frame(reference)
            frame = _restore_pandas_schema(
                frame,
                entry.get("pandas_schema"),
                path=f"tables[{logical_id}].pandas_schema",
            )
            if value_kind == "series":
                if list(frame.columns) != [_SERIES_COLUMN]:
                    raise AnalyticsWorkspaceCacheIntegrityError(
                        "cached analytics Series has an invalid Parquet schema"
                    )
                value = frame[_SERIES_COLUMN].copy(deep=True)
                value.name = _decode_value(
                    entry.get("series_name"),
                    path=f"tables[{logical_id}].series_name",
                )
            else:
                value = frame
            values[logical_id] = value
            references.append(reference)
        return values, references




def _coerce_identity(
    value: AnalyticsCacheIdentity | Mapping[str, Any],
) -> AnalyticsCacheIdentity:
    if isinstance(value, AnalyticsCacheIdentity):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "identity must be an AnalyticsCacheIdentity or a mapping"
        )
    return AnalyticsCacheIdentity(value)




__all__ = [
    "AnalyticsCacheIdentity",
    "AnalyticsCacheOutcome",
    "AnalyticsCacheSource",
    "AnalyticsWorkspaceCache",
    "AnalyticsWorkspaceCacheCollisionError",
    "AnalyticsWorkspaceCacheError",
    "AnalyticsWorkspaceCacheIntegrityError",
    "AnalyticsWorkspaceCacheMissError",
    "AnalyticsWorkspaceCacheSerializationError",
]
