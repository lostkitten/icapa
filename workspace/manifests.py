"""Immutable workspace artifact and run-manifest references."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .identity import canonical_json_bytes, canonicalize


RUN_MANIFEST_SCHEMA_VERSION = 2


class WorkspaceRepositoryError(RuntimeError):
    """Base error for immutable workspace repository operations."""


class ManifestIntegrityError(WorkspaceRepositoryError):
    """Raised when stored manifest metadata is damaged or inconsistent."""


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable reference to one content-addressed workspace artifact."""

    artifact_type: str
    content_digest: str
    file_checksum: str
    relative_path: str
    format: str
    schema_version: int
    size_bytes: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            artifact_type=str(value["artifact_type"]),
            content_digest=str(value["content_digest"]),
            file_checksum=str(value["file_checksum"]),
            relative_path=str(value["relative_path"]),
            format=str(value["format"]),
            schema_version=int(value["schema_version"]),
            size_bytes=int(value["size_bytes"]),
        )


@dataclass(frozen=True)
class RunManifestRef:
    """Location and automatic identity of one execution manifest."""

    workspace_name: str
    definition_fingerprint: str
    execution_id: str
    path: str


@dataclass(frozen=True)
class RunManifest:
    """Automatically collected metadata for one research execution."""

    schema_version: int
    execution_id: str
    status: str
    definition_fingerprint: str
    request_fingerprint: str
    result_fingerprint: str | None
    workspace_name: str
    index_id: str
    created_at: str
    completed_at: str | None
    software: tuple[Mapping[str, Any], ...] = ()
    providers: tuple[Mapping[str, Any], ...] = ()
    calendar: Mapping[str, Any] = field(default_factory=dict)
    request: Mapping[str, Any] = field(default_factory=dict)
    cache_decisions: tuple[Mapping[str, Any], ...] = ()
    input_digests: tuple[Mapping[str, Any], ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    failure: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "execution_id": self.execution_id,
            "status": self.status,
            "definition_fingerprint": self.definition_fingerprint,
            "request_fingerprint": self.request_fingerprint,
            "result_fingerprint": self.result_fingerprint,
            "workspace_name": self.workspace_name,
            "index_id": self.index_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "software": [dict(item) for item in self.software],
            "providers": [dict(item) for item in self.providers],
            "calendar": dict(self.calendar),
            "request": dict(self.request),
            "cache_decisions": [dict(item) for item in self.cache_decisions],
            "input_digests": [dict(item) for item in self.input_digests],
            "artifacts": [asdict(item) for item in self.artifacts],
            "failure": None if self.failure is None else dict(self.failure),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunManifest":
        return cls(
            schema_version=int(value["schema_version"]),
            execution_id=str(value["execution_id"]),
            status=str(value["status"]),
            definition_fingerprint=str(value["definition_fingerprint"]),
            request_fingerprint=str(value["request_fingerprint"]),
            result_fingerprint=(
                None
                if value.get("result_fingerprint") is None
                else str(value["result_fingerprint"])
            ),
            workspace_name=str(value["workspace_name"]),
            index_id=str(value["index_id"]),
            created_at=str(value["created_at"]),
            completed_at=(
                None
                if value.get("completed_at") is None
                else str(value["completed_at"])
            ),
            software=tuple(dict(item) for item in value.get("software", ())),
            providers=tuple(dict(item) for item in value.get("providers", ())),
            calendar=dict(value.get("calendar", {})),
            request=dict(value.get("request", {})),
            cache_decisions=tuple(
                dict(item) for item in value.get("cache_decisions", ())
            ),
            input_digests=tuple(
                dict(item) for item in value.get("input_digests", ())
            ),
            artifacts=tuple(
                ArtifactRef.from_dict(item) for item in value.get("artifacts", ())
            ),
            failure=(
                None if value.get("failure") is None else dict(value["failure"])
            ),
        )


def merge_artifacts(
    existing: Sequence[ArtifactRef],
    additional: Sequence[ArtifactRef],
) -> tuple[ArtifactRef, ...]:
    """Merge logical artifact references in deterministic identity order."""

    merged: dict[tuple[str, str, str], ArtifactRef] = {}
    for artifact in (*existing, *additional):
        key = (
            artifact.artifact_type,
            artifact.content_digest,
            artifact.file_checksum,
        )
        merged[key] = artifact
    return tuple(merged[key] for key in sorted(merged))


def merge_input_digests(
    existing: Sequence[Mapping[str, Any]],
    additional: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Merge canonical input records without retaining duplicates."""

    merged: dict[bytes, Mapping[str, Any]] = {}
    for item in (*existing, *additional):
        if not isinstance(item, Mapping):
            raise TypeError("input digest records must be mappings")
        canonical = canonicalize(dict(item))
        if not isinstance(canonical, dict):
            raise TypeError("input digest records must canonicalize to mappings")
        key = canonical_json_bytes(canonical)
        merged[key] = canonical
    return tuple(merged[key] for key in sorted(merged))


def result_input_identities(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return cache-source-neutral inputs in deterministic content order."""

    has_verified_market_content = any(
        record.get("input_type") == "source_daily_market_data"
        and record.get("content_digest") is not None
        for record in records
    )
    normalized: dict[bytes, Mapping[str, Any]] = {}
    for record in records:
        if (
            has_verified_market_content
            and record.get("input_type")
            in {"source_data_snapshot", "simulation_snapshot"}
        ):
            continue
        identity = {
            str(key): value
            for key, value in record.items()
            if key
            not in {
                "cache_source",
                "snapshot_digest",
                "snapshot_protocol",
            }
        }
        key = canonical_json_bytes(identity)
        normalized[key] = identity
    return [normalized[key] for key in sorted(normalized)]


def validate_digest(label: str, value: str) -> None:
    """Require a lowercase SHA-256 digest."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ManifestIntegrityError(
            f"{label} is not a lowercase SHA-256 digest"
        )


def write_envelope(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist a checksummed JSON metadata envelope."""

    payload_bytes = canonical_json_bytes(dict(payload))
    atomic_write(
        path,
        canonical_json_bytes(
            {
                "checksum": sha256(payload_bytes).hexdigest(),
                "payload": dict(payload),
            }
        ),
    )


def read_envelope(path: Path) -> dict[str, Any]:
    """Read and verify a checksummed JSON metadata envelope."""

    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(envelope["payload"])
        expected = str(envelope["checksum"])
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as exc:
        raise ManifestIntegrityError(
            f"metadata envelope is invalid: {path}"
        ) from exc
    observed = sha256(canonical_json_bytes(payload)).hexdigest()
    if observed != expected:
        raise ManifestIntegrityError(
            f"metadata checksum does not match its payload: {path}"
        )
    return payload


def atomic_write(path: Path, raw: bytes) -> None:
    """Replace one metadata file atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def utc_now() -> str:
    """Return one timezone-aware UTC timestamp for persisted metadata."""

    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ArtifactRef",
    "ManifestIntegrityError",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RunManifest",
    "RunManifestRef",
    "WorkspaceRepositoryError",
]
