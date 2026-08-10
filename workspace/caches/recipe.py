"""Persistent recipe-stage caching backed by a research workspace.

The adapter deliberately implements only the small ``StageCache`` protocol.
RecipeRunner remains responsible for producing the cache identity, while the
workspace provides immutable, content-addressed Parquet objects and durable
catalog bindings.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from ..artifacts import ArtifactError as WorkspaceArtifactError
from ..repository import ManifestIntegrityError, WorkspaceRepository
from .models import CacheMode, CacheStage
from ...portfolio_construction.recipes.artifacts import Artifact, ArtifactKey
from ...portfolio_construction.recipes.contracts import (
    StageDiagnostic,
    StageResult,
)


_SCHEMA_VERSION = 1
_MANIFEST_BINDING = "stage-result"
_VALUE_BINDING_PREFIX = "stage-value"
_MAX_METADATA_BYTES = 1024 * 1024
_SERIES_COLUMN = "__icapa_series_value__"


class WorkspaceStageCacheError(RuntimeError):
    """Base error for persistent recipe-stage cache operations."""


class WorkspaceStageCacheIntegrityError(WorkspaceStageCacheError):
    """Raised when a committed cache result is missing or corrupt."""


class WorkspaceStageCacheSerializationError(WorkspaceStageCacheError):
    """Raised when a stage result cannot be persisted without coercion."""


class WorkspaceStageCacheCollisionError(WorkspaceStageCacheError):
    """Raised when one cache identity is associated with different results."""


class WorkspaceStageCacheMissError(WorkspaceStageCacheError):
    """Raised when READ_ONLY mode cannot resolve a committed stage result."""


class WorkspaceStageCache:
    """Persist ``StageResult`` objects in a named ``WorkspaceRepository``.

    DataFrames and Series are stored as immutable Parquet objects. JSON-safe
    values, artifact metadata, and diagnostics are kept in a compact manifest
    with its own checksum. The manifest binding is written last and therefore
    acts as the commit marker for the complete stage result.
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

    def load(self, key: str) -> StageResult | None:
        """Load and verify the result bound to a RecipeRunner cache key."""

        if self.mode in {CacheMode.OFF, CacheMode.REFRESH}:
            return None
        with _exclusive_cache_key_lock(self.workspace, key):
            result = self._load_committed(key)
        if result is None and self.mode is CacheMode.READ_ONLY:
            raise WorkspaceStageCacheMissError(
                "READ_ONLY recipe cache is missing a required stage result"
            )
        return result

    def _load_committed(self, key: str) -> StageResult | None:
        try:
            manifest_reference = self.workspace.resolve_artifact(
                stage=CacheStage.REVIEWS,
                cache_key=key,
                name=_MANIFEST_BINDING,
            )
        except (ManifestIntegrityError, WorkspaceArtifactError, OSError) as exc:
            raise WorkspaceStageCacheIntegrityError(
                "stage cache manifest reference is invalid"
            ) from exc
        if manifest_reference is None:
            return None

        try:
            manifest_frame = self.workspace.load_frame(manifest_reference)
            payload = _decode_manifest_frame(manifest_frame, expected_key=key)
            result = self._restore_result(key, payload)
            if _result_digest(result) != payload["result_digest"]:
                raise WorkspaceStageCacheIntegrityError(
                    "stage cache result digest does not match its manifest"
                )
            return result
        except WorkspaceStageCacheIntegrityError:
            raise
        except WorkspaceStageCacheSerializationError as exc:
            raise WorkspaceStageCacheIntegrityError(
                "stage cache manifest contains unsupported metadata"
            ) from exc
        except (ManifestIntegrityError, WorkspaceArtifactError, OSError) as exc:
            raise WorkspaceStageCacheIntegrityError(
                "stage cache artifact is missing or corrupt"
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkspaceStageCacheIntegrityError(
                "stage cache manifest has an invalid schema"
            ) from exc

    def save(self, key: str, result: StageResult) -> None:
        """Save a complete immutable result and bind its commit manifest."""

        if not isinstance(result, StageResult):
            raise TypeError("result must be a StageResult")
        if self.mode is CacheMode.READ_ONLY:
            raise WorkspaceStageCacheMissError(
                "READ_ONLY recipe cache cannot save a calculated stage result"
            )
        if self.mode is CacheMode.OFF:
            return
        result_digest = _validate_result_for_storage(result)
        existing = self._load_committed(key)
        if existing is not None:
            if _result_digest(existing) != result_digest:
                raise WorkspaceStageCacheCollisionError(
                    "cache identity is already bound to a different stage result"
                )
            return

        artifact_entries: list[dict[str, Any]] = []
        frame_values: list[tuple[dict[str, Any], str, pd.DataFrame]] = []
        for position, (artifact_key, artifact) in enumerate(
            sorted(
                result.artifacts.items(),
                key=lambda item: item[0].canonical_name,
            )
        ):
            entry, frame_value = self._prepare_artifact(
                position=position,
                artifact_key=artifact_key,
                artifact=artifact,
            )
            artifact_entries.append(entry)
            if frame_value is not None:
                frame_values.append((entry, entry["binding"], frame_value))

        unsigned_payload: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "cache_key": key,
            "artifacts": artifact_entries,
            "diagnostics": [
                {
                    "code": diagnostic.code,
                    "message": diagnostic.message,
                    "severity": diagnostic.severity,
                    "metrics": _json_value(
                        diagnostic.metrics,
                        path=f"diagnostics[{position}].metrics",
                    ),
                }
                for position, diagnostic in enumerate(result.diagnostics)
            ],
            "result_digest": result_digest,
        }
        preflight_payload = dict(unsigned_payload)
        preflight_payload["metadata_checksum"] = _json_digest(unsigned_payload)
        if len(_json_bytes(preflight_payload)) > _MAX_METADATA_BYTES:
            raise WorkspaceStageCacheSerializationError(
                "stage cache metadata exceeds the one-megabyte limit"
            )

        with _exclusive_cache_key_lock(self.workspace, key):
            # The optimistic check above avoids unnecessary preparation in
            # the common warm-cache case. This second check is authoritative:
            # another process may have committed while this result was being
            # serialized.
            concurrent = self._load_committed(key)
            if concurrent is not None:
                if _result_digest(concurrent) != result_digest:
                    raise WorkspaceStageCacheCollisionError(
                        "cache identity was concurrently bound to a different "
                        "stage result"
                    )
                return

            for entry, binding_name, frame in frame_values:
                reference = self.workspace.save_frame(
                    "recipe_stage_value",
                    frame,
                )
                self.workspace.bind_artifact(
                    stage=CacheStage.REVIEWS,
                    cache_key=key,
                    name=binding_name,
                    artifact=reference,
                )
                entry["workspace_content_digest"] = reference.content_digest
                entry["workspace_file_checksum"] = reference.file_checksum

            payload = dict(unsigned_payload)
            payload["metadata_checksum"] = _json_digest(unsigned_payload)
            encoded = _json_bytes(payload)
            if len(encoded) > _MAX_METADATA_BYTES:
                raise WorkspaceStageCacheSerializationError(
                    "stage cache metadata exceeds the one-megabyte limit"
                )

            manifest_frame = pd.DataFrame(
                {"payload_json": [encoded.decode("utf-8")]}
            )
            manifest_reference = self.workspace.save_frame(
                "recipe_stage_manifest",
                manifest_frame,
            )
            self.workspace.bind_artifact(
                stage=CacheStage.REVIEWS,
                cache_key=key,
                name=_MANIFEST_BINDING,
                artifact=manifest_reference,
            )

            committed = self._load_committed(key)
            if committed is None or _result_digest(committed) != result_digest:
                raise WorkspaceStageCacheIntegrityError(
                    "stage cache commit did not resolve to the complete result"
                )

    def _prepare_artifact(
        self,
        *,
        position: int,
        artifact_key: ArtifactKey,
        artifact: Artifact,
    ) -> tuple[dict[str, Any], pd.DataFrame | None]:
        if artifact.key != artifact_key:
            raise WorkspaceStageCacheSerializationError(
                "stage result contains an artifact key mismatch"
            )
        binding_name = f"{_VALUE_BINDING_PREFIX}-{position:04d}"
        entry: dict[str, Any] = {
            "key": {
                "namespace": artifact_key.namespace,
                "name": artifact_key.name,
                "schema_version": artifact_key.schema_version,
            },
            "artifact_digest": artifact.digest,
            "metadata": _json_value(
                artifact.metadata,
                path=f"artifacts[{artifact_key.canonical_name}].metadata",
            ),
        }
        value = artifact.value
        if isinstance(value, pd.DataFrame):
            entry.update(
                {
                    "value_kind": "dataframe",
                    "binding": binding_name,
                    "workspace_content_digest": "0" * 64,
                    "workspace_file_checksum": "0" * 64,
                }
            )
            frame = value
        elif isinstance(value, pd.Series):
            entry.update(
                {
                    "value_kind": "series",
                    "binding": binding_name,
                    "workspace_content_digest": "0" * 64,
                    "workspace_file_checksum": "0" * 64,
                    "series_name": _json_value(
                        value.name,
                        path=f"artifacts[{artifact_key.canonical_name}].series_name",
                    ),
                }
            )
            frame = value.to_frame(name=_SERIES_COLUMN)
        else:
            entry.update(
                {
                    "value_kind": "json",
                    "value": _json_value(
                        value,
                        path=f"artifacts[{artifact_key.canonical_name}].value",
                    ),
                }
            )
            return entry, None

        return entry, frame

    def _restore_result(
        self,
        cache_key: str,
        payload: Mapping[str, Any],
    ) -> StageResult:
        artifacts: dict[ArtifactKey, Artifact] = {}
        entries = payload["artifacts"]
        if not isinstance(entries, list):
            raise WorkspaceStageCacheIntegrityError(
                "stage cache artifact entries must be a list"
            )
        for entry in entries:
            artifact = self._restore_artifact(cache_key, entry)
            if artifact.key in artifacts:
                raise WorkspaceStageCacheIntegrityError(
                    "stage cache contains duplicate artifact keys"
                )
            artifacts[artifact.key] = artifact

        diagnostics_payload = payload["diagnostics"]
        if not isinstance(diagnostics_payload, list):
            raise WorkspaceStageCacheIntegrityError(
                "stage cache diagnostics must be a list"
            )
        diagnostics = tuple(
            StageDiagnostic(
                code=item["code"],
                message=item["message"],
                severity=item["severity"],
                metrics=_json_value(
                    item["metrics"],
                    path=f"diagnostics[{position}].metrics",
                ),
            )
            for position, item in enumerate(diagnostics_payload)
        )
        return StageResult(artifacts=artifacts, diagnostics=diagnostics)

    def _restore_artifact(
        self,
        cache_key: str,
        entry: Mapping[str, Any],
    ) -> Artifact:
        if not isinstance(entry, Mapping):
            raise WorkspaceStageCacheIntegrityError(
                "stage cache artifact entry must be an object"
            )
        key_payload = entry["key"]
        if not isinstance(key_payload, Mapping):
            raise WorkspaceStageCacheIntegrityError(
                "stage cache artifact key must be an object"
            )
        artifact_key = ArtifactKey(
            namespace=key_payload["namespace"],
            name=key_payload["name"],
            schema_version=key_payload["schema_version"],
        )
        value_kind = entry["value_kind"]
        if value_kind == "json":
            value = _json_value(
                entry["value"],
                path=f"artifacts[{artifact_key.canonical_name}].value",
            )
        elif value_kind in {"dataframe", "series"}:
            binding = entry["binding"]
            if not isinstance(binding, str):
                raise WorkspaceStageCacheIntegrityError(
                    "stage cache artifact binding must be a string"
                )
            reference = self.workspace.resolve_artifact(
                stage=CacheStage.REVIEWS,
                cache_key=cache_key,
                name=binding,
            )
            if reference is None:
                raise WorkspaceStageCacheIntegrityError(
                    "stage cache references a missing artifact binding"
                )
            if (
                reference.content_digest != entry["workspace_content_digest"]
                or reference.file_checksum != entry["workspace_file_checksum"]
            ):
                raise WorkspaceStageCacheIntegrityError(
                    "stage cache binding does not match its committed artifact"
                )
            frame = self.workspace.load_frame(reference)
            if value_kind == "dataframe":
                value = frame
            else:
                if list(frame.columns) != [_SERIES_COLUMN]:
                    raise WorkspaceStageCacheIntegrityError(
                        "cached Series has an invalid Parquet schema"
                    )
                value = frame[_SERIES_COLUMN].copy()
                value.name = _json_value(
                    entry["series_name"],
                    path=f"artifacts[{artifact_key.canonical_name}].series_name",
                )
        else:
            raise WorkspaceStageCacheIntegrityError(
                "stage cache artifact has an unsupported value kind"
            )

        expected_digest = entry["artifact_digest"]
        metadata = _json_value(
            entry["metadata"],
            path=f"artifacts[{artifact_key.canonical_name}].metadata",
        )
        if not isinstance(metadata, dict):
            raise WorkspaceStageCacheIntegrityError(
                "cached artifact metadata must be an object"
            )
        return Artifact(
            key=artifact_key,
            value=value,
            digest=expected_digest,
            metadata=metadata,
        )


def _decode_manifest_frame(
    frame: pd.DataFrame,
    *,
    expected_key: str,
) -> dict[str, Any]:
    if list(frame.columns) != ["payload_json"] or len(frame) != 1:
        raise WorkspaceStageCacheIntegrityError(
            "stage cache manifest has an invalid Parquet schema"
        )
    encoded = frame.iloc[0]["payload_json"]
    if not isinstance(encoded, str):
        raise WorkspaceStageCacheIntegrityError(
            "stage cache manifest payload must be text"
        )
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise WorkspaceStageCacheIntegrityError(
            "stage cache manifest payload is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise WorkspaceStageCacheIntegrityError(
            "stage cache manifest payload must be an object"
        )
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise WorkspaceStageCacheIntegrityError(
            "stage cache manifest schema version is not supported"
        )
    if payload.get("cache_key") != expected_key:
        raise WorkspaceStageCacheIntegrityError(
            "stage cache manifest is bound to a different cache identity"
        )
    checksum = payload.get("metadata_checksum")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "metadata_checksum"
    }
    if not isinstance(checksum, str) or checksum != _json_digest(unsigned):
        raise WorkspaceStageCacheIntegrityError(
            "stage cache metadata checksum does not match its payload"
        )
    required = {
        "artifacts",
        "diagnostics",
        "result_digest",
    }
    if not required.issubset(payload):
        raise WorkspaceStageCacheIntegrityError(
            "stage cache manifest is missing required fields"
        )
    return payload


def _result_digest(result: StageResult) -> str:
    identity = {
        "artifacts": [
            {
                "key": key.canonical_name,
                "digest": artifact.digest,
                "metadata": _json_value(
                    artifact.metadata,
                    path=f"artifacts[{key.canonical_name}].metadata",
                ),
            }
            for key, artifact in sorted(
                result.artifacts.items(),
                key=lambda item: item[0].canonical_name,
            )
        ],
        "diagnostics": [
            {
                "code": item.code,
                "message": item.message,
                "severity": item.severity,
                "metrics": _json_value(
                    item.metrics,
                    path=f"diagnostics[{position}].metrics",
                ),
            }
            for position, item in enumerate(result.diagnostics)
        ],
    }
    return _json_digest(identity)


def _validate_result_for_storage(result: StageResult) -> str:
    """Validate every value before creating any workspace object or binding."""

    for key, artifact in result.artifacts.items():
        _json_value(
            artifact.metadata,
            path=f"artifacts[{key.canonical_name}].metadata",
        )
        if isinstance(artifact.value, pd.DataFrame):
            continue
        if isinstance(artifact.value, pd.Series):
            _json_value(
                artifact.value.name,
                path=f"artifacts[{key.canonical_name}].series_name",
            )
            continue
        _json_value(
            artifact.value,
            path=f"artifacts[{key.canonical_name}].value",
        )
    for position, diagnostic in enumerate(result.diagnostics):
        _json_value(
            diagnostic.metrics,
            path=f"diagnostics[{position}].metrics",
        )
    return _result_digest(result)


def _json_value(value: Any, *, path: str) -> Any:
    """Copy a strict JSON value without stringifying unsupported objects."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WorkspaceStageCacheSerializationError(
                f"{path} contains a non-finite float"
            )
        return value
    if isinstance(value, list):
        return [
            _json_value(item, path=f"{path}[{position}]")
            for position, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise WorkspaceStageCacheSerializationError(
                    f"{path} contains a non-string object key"
                )
            copied[key] = _json_value(item, path=f"{path}.{key}")
        return copied
    raise WorkspaceStageCacheSerializationError(
        f"{path} contains unsupported type {type(value).__qualname__}"
    )


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkspaceStageCacheSerializationError(
            "stage cache metadata is not JSON-safe"
        ) from exc


def _json_digest(value: Any) -> str:
    return sha256(_json_bytes(value)).hexdigest()


def _cache_key_lock_path(
    workspace: WorkspaceRepository,
    key: str,
) -> Path:
    """Return a fixed, traversal-safe lock path for one stage cache key."""

    if not isinstance(key, str):
        raise TypeError("stage cache key must be a string")
    token = sha256(key.encode("utf-8")).hexdigest()
    return workspace.workspace_path.joinpath(
        ".locks",
        "recipe-stage-cache",
        f"{token}.lock",
    )


@contextmanager
def _exclusive_cache_key_lock(
    workspace: WorkspaceRepository,
    key: str,
) -> Iterator[None]:
    """Serialize a complete multi-artifact commit across processes."""

    path = _cache_key_lock_path(workspace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


__all__ = [
    "WorkspaceStageCache",
    "WorkspaceStageCacheCollisionError",
    "WorkspaceStageCacheError",
    "WorkspaceStageCacheIntegrityError",
    "WorkspaceStageCacheMissError",
    "WorkspaceStageCacheSerializationError",
]
