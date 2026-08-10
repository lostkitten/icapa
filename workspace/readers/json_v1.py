"""Content-addressed artifacts for named ICAPA workspaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any
from uuid import uuid4

import pandas as pd

from icapa.backtesting.reviews.cache_contracts import (
    ArtifactMetadata,
    CachePolicy,
    CacheSource,
    LoadedReview,
    ReviewArtifact,
    ReviewCacheMissError,
    register_review_store_factory,
)
from icapa.backtesting.reviews.identity import (
    build_run_fingerprint as build_review_run_fingerprint,
    canonical_digest as review_canonical_digest,
)
from icapa.backtesting.simulation.cache_contracts import (
    SimulationCacheMissError,
)

from ..locking import exclusive_file_lock


WORKSPACE_ROOT_ENV = "ICAPA_WORKSPACE_ROOT"
WORKSPACE_SCHEMA_VERSION = 1
_WORKSPACE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ARTIFACT_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MEMORY_ARTIFACTS: dict[str, bytes] = {}
_MEMORY_LOCK = threading.RLock()
_MANIFEST_LOCK = threading.RLock()


class WorkspaceError(RuntimeError):
    """Base error for workspace operations."""


class CacheMissError(
    WorkspaceError,
    ReviewCacheMissError,
    SimulationCacheMissError,
):
    """Raised when a read-only workspace does not contain a requested artifact."""


class CacheIntegrityError(WorkspaceError):
    """Raised when an artifact or manifest fails its checksum or identity checks."""


def get_workspace_root() -> Path:
    """Return the fixed workspace root, with only the documented environment override."""

    configured = os.environ.get(WORKSPACE_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home().joinpath(".icapa", "workspaces")


def validate_workspace_name(name: str) -> str:
    """Validate a single safe path component used as a user-facing workspace name."""

    if not isinstance(name, str) or not _WORKSPACE_NAME.fullmatch(name):
        raise ValueError(
            "workspace_name must be 1-64 characters using only letters, numbers, "
            "periods, underscores, and hyphens, and must start with a letter or number"
        )
    if name in {".", ".."}:
        raise ValueError("workspace_name must not be a relative path marker")
    return name


def canonical_digest(value: Any) -> str:
    """Hash a value after deterministic, JSON-safe normalization."""

    return review_canonical_digest(value)


def build_run_fingerprint(
    *,
    index_id: str,
    methodology: object,
    configuration: Mapping[str, Any] | None = None,
    data_revision: Any = "unversioned",
) -> str:
    """Build a range-independent fingerprint for reusable review calculations.

    ``configuration`` should contain calculation configuration such as provider
    identities and calendar semantics, but not a requested calendar date range.
    """

    return build_review_run_fingerprint(
        index_id=index_id,
        methodology=methodology,
        configuration=configuration,
        data_revision=data_revision,
    )


def clear_memory_cache() -> None:
    """Remove process-local artifact copies without changing disk workspaces."""

    with _MEMORY_LOCK:
        _MEMORY_ARTIFACTS.clear()


class WorkspaceStore:
    """Persist immutable artifacts for one named, fingerprinted calculation."""

    def __init__(
        self,
        *,
        workspace_name: str,
        fingerprint: str,
        index_id: str,
        methodology_name: str,
        configuration_digest: str,
        data_revision: Any,
    ) -> None:
        self.workspace_name = validate_workspace_name(workspace_name)
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("fingerprint must be a lowercase SHA-256 digest")
        self.fingerprint = fingerprint
        self.index_id = index_id
        self.methodology_name = methodology_name
        self.configuration_digest = configuration_digest
        self.data_revision = _canonicalize(data_revision)
        self.root = get_workspace_root()
        self.workspace_path = self.root.joinpath(
            self.workspace_name,
            "runs",
            self.fingerprint,
        )
        self.manifest_path = self.workspace_path.joinpath("manifest.json")

    @property
    def reports_path(self) -> Path:
        """Return the report output directory for this fingerprinted run."""

        return self.workspace_path.joinpath("reports")

    def report_path(self, file_name: str, *, create_parent: bool = True) -> Path:
        """Return a safe report path within the run."""

        if (
            not isinstance(file_name, str)
            or not file_name
            or Path(file_name).name != file_name
            or file_name in {".", ".."}
        ):
            raise ValueError("report file_name must be a non-empty base name")
        if create_parent:
            self.reports_path.mkdir(parents=True, exist_ok=True)
        return self.reports_path.joinpath(file_name)

    def simulation_catalog_lock(self, namespace: str):
        """Return the infrastructure lock for one simulation catalogue."""

        safe_namespace = _validate_artifact_token("namespace", namespace)
        return exclusive_file_lock(
            self.workspace_path.joinpath(
                ".locks",
                "simulation_segments",
                f"{safe_namespace}.lock",
            )
        )

    def load_review(self, reference_date, effective_date) -> LoadedReview:
        """Load and verify one cached review, preferring process memory."""

        reference = _normalize_date(reference_date)
        effective = _normalize_date(effective_date)
        review_key = self._review_key(reference, effective)
        path = self._review_path(review_key)
        memory_key = str(path)
        with _MEMORY_LOCK:
            memory_bytes = _MEMORY_ARTIFACTS.get(memory_key)
        if memory_bytes is not None:
            payload, payload_checksum = _decode_envelope(memory_bytes, path)
            return LoadedReview(
                artifact=self._decode_review(payload, reference, effective),
                metadata=ArtifactMetadata(
                    source=CacheSource.MEMORY,
                    checksum=sha256(memory_bytes).hexdigest(),
                    path=str(path),
                ),
            )

        manifest = self._load_manifest(required=True)
        entry = manifest.get("reviews", {}).get(review_key)
        if entry is None:
            raise CacheMissError(
                f"workspace {self.workspace_name!r} has no cached review for "
                f"{reference.date()} / {effective.date()}"
            )
        raw = _read_bytes(path)
        file_checksum = sha256(raw).hexdigest()
        if file_checksum != entry.get("checksum"):
            raise CacheIntegrityError(f"review artifact checksum does not match manifest: {path}")
        payload, payload_checksum = _decode_envelope(raw, path)
        if payload_checksum != entry.get("payload_checksum"):
            raise CacheIntegrityError(f"review payload checksum does not match manifest: {path}")
        with _MEMORY_LOCK:
            _MEMORY_ARTIFACTS[memory_key] = raw
        return LoadedReview(
            artifact=self._decode_review(payload, reference, effective),
            metadata=ArtifactMetadata(
                source=CacheSource.DISK,
                checksum=file_checksum,
                path=str(path),
            ),
        )

    def save_review(self, artifact: ReviewArtifact) -> ArtifactMetadata:
        """Atomically save one review and update its checksummed manifest entry."""

        reference = _normalize_date(artifact.reference_date)
        effective = _normalize_date(artifact.effective_date)
        if artifact.index_id != self.index_id:
            raise ValueError("review index_id does not match the workspace")
        diagnostics = artifact.diagnostics or {}
        if not isinstance(diagnostics, Mapping):
            raise TypeError("review diagnostics must be a mapping")
        provenance_records = tuple(artifact.provenance_records or ())
        if any(not isinstance(item, Mapping) for item in provenance_records):
            raise TypeError("review provenance records must be mappings")
        review_key = self._review_key(reference, effective)
        path = self._review_path(review_key)
        payload = {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "reference_date": reference.isoformat(),
            "effective_date": effective.isoformat(),
            "index_id": artifact.index_id,
            "universe_id": artifact.universe_id,
            "constituents": _encode_frame(artifact.constituents),
            "daily": None if artifact.daily is None else _encode_frame(artifact.daily),
            "diagnostics": _canonicalize(diagnostics),
            "provenance_records": _canonicalize(provenance_records),
        }
        raw, payload_checksum = _encode_envelope(payload)
        _atomic_write(path, raw)
        file_checksum = sha256(raw).hexdigest()
        with _MEMORY_LOCK:
            _MEMORY_ARTIFACTS[str(path)] = raw

        completed_at = _utc_now()

        def add_review(manifest: dict[str, Any]) -> None:
            manifest.setdefault("reviews", {})[review_key] = {
                "reference_date": reference.isoformat(),
                "effective_date": effective.isoformat(),
                "path": str(path.relative_to(self.workspace_path)),
                "checksum": file_checksum,
                "payload_checksum": payload_checksum,
                "size_bytes": len(raw),
                "status": "complete",
                "completed_at": completed_at,
            }

        self._update_manifest(add_review)
        return ArtifactMetadata(
            source=CacheSource.COMPUTED,
            checksum=file_checksum,
            path=str(path),
        )

    def record_request(
        self,
        reviews: Sequence[Mapping[str, Any]],
        *,
        cache_policy: CachePolicy | str,
    ) -> None:
        """Record exact requested coverage and each review's observed cache source."""

        policy = CachePolicy(cache_policy)
        request = {
            "requested_at": _utc_now(),
            "cache_policy": policy.value,
            "reviews": [
                {
                    "reference_date": _normalize_date(item["reference_date"]).isoformat(),
                    "effective_date": _normalize_date(item["effective_date"]).isoformat(),
                    "cache_source": CacheSource(item["cache_source"]).value,
                }
                for item in reviews
            ],
            "status": "complete",
        }

        def add_request(manifest: dict[str, Any]) -> None:
            manifest.setdefault("requests", []).append(request)
            manifest["last_request"] = request

        self._update_manifest(add_request)

    def save_frame(self, stage: str, key: str, name: str, frame: pd.DataFrame) -> ArtifactMetadata:
        """Save a generic tabular stage artifact for simulation or analytics."""

        path, manifest_key = self._generic_path(stage, key, name, suffix="frame.json")
        payload = {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "kind": "frame",
            "frame": _encode_frame(frame),
        }
        return self._save_generic(path, manifest_key, payload)

    def load_frame(self, stage: str, key: str, name: str) -> pd.DataFrame:
        """Load and verify a generic tabular stage artifact."""

        path, manifest_key = self._generic_path(stage, key, name, suffix="frame.json")
        payload, _ = self._load_generic(path, manifest_key)
        if payload.get("kind") != "frame":
            raise CacheIntegrityError(f"artifact is not a frame: {path}")
        return _decode_frame(payload["frame"])

    def save_json(self, stage: str, key: str, name: str, value: Any) -> ArtifactMetadata:
        """Save deterministic JSON metadata for a calculation stage."""

        path, manifest_key = self._generic_path(stage, key, name, suffix="json")
        payload = {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "kind": "json",
            "value": _canonicalize(value),
        }
        return self._save_generic(path, manifest_key, payload)

    def load_json(self, stage: str, key: str, name: str) -> Any:
        """Load and verify deterministic JSON metadata for a calculation stage."""

        path, manifest_key = self._generic_path(stage, key, name, suffix="json")
        payload, _ = self._load_generic(path, manifest_key)
        if payload.get("kind") != "json":
            raise CacheIntegrityError(f"artifact is not JSON metadata: {path}")
        return payload["value"]

    def manifest(self) -> dict[str, Any]:
        """Return a defensive JSON round-trip of the current manifest."""

        manifest = self._load_manifest(required=True)
        return json.loads(json.dumps(manifest))

    def _save_generic(
        self,
        path: Path,
        manifest_key: str,
        payload: Mapping[str, Any],
    ) -> ArtifactMetadata:
        raw, payload_checksum = _encode_envelope(dict(payload))
        _atomic_write(path, raw)
        file_checksum = sha256(raw).hexdigest()
        with _MEMORY_LOCK:
            _MEMORY_ARTIFACTS[str(path)] = raw

        def add_artifact(manifest: dict[str, Any]) -> None:
            manifest.setdefault("artifacts", {})[manifest_key] = {
                "path": str(path.relative_to(self.workspace_path)),
                "checksum": file_checksum,
                "payload_checksum": payload_checksum,
                "size_bytes": len(raw),
                "status": "complete",
                "completed_at": _utc_now(),
            }

        self._update_manifest(add_artifact)
        return ArtifactMetadata(CacheSource.COMPUTED, file_checksum, str(path))

    def _load_generic(self, path: Path, manifest_key: str) -> tuple[dict[str, Any], ArtifactMetadata]:
        with _MEMORY_LOCK:
            raw = _MEMORY_ARTIFACTS.get(str(path))
        source = CacheSource.MEMORY
        if raw is None:
            manifest = self._load_manifest(required=True)
            entry = manifest.get("artifacts", {}).get(manifest_key)
            if entry is None:
                raise CacheMissError(f"workspace artifact is not cached: {manifest_key}")
            raw = _read_bytes(path)
            checksum = sha256(raw).hexdigest()
            if checksum != entry.get("checksum"):
                raise CacheIntegrityError(f"artifact checksum does not match manifest: {path}")
            source = CacheSource.DISK
            with _MEMORY_LOCK:
                _MEMORY_ARTIFACTS[str(path)] = raw
        payload, _ = _decode_envelope(raw, path)
        metadata = ArtifactMetadata(source, sha256(raw).hexdigest(), str(path))
        return payload, metadata

    def _decode_review(
        self,
        payload: Mapping[str, Any],
        reference_date: pd.Timestamp,
        effective_date: pd.Timestamp,
    ) -> ReviewArtifact:
        if payload.get("schema_version") != WORKSPACE_SCHEMA_VERSION:
            raise CacheIntegrityError("review artifact schema version is not supported")
        if payload.get("index_id") != self.index_id:
            raise CacheIntegrityError("review artifact index_id does not match the workspace")
        stored_reference = _normalize_date(payload.get("reference_date"))
        stored_effective = _normalize_date(payload.get("effective_date"))
        if stored_reference != reference_date or stored_effective != effective_date:
            raise CacheIntegrityError("review artifact dates do not match the cache key")
        diagnostics = payload.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            raise CacheIntegrityError("review artifact diagnostics must be an object")
        provenance_records = payload.get("provenance_records", [])
        if not isinstance(provenance_records, list) or any(
            not isinstance(item, Mapping) for item in provenance_records
        ):
            raise CacheIntegrityError(
                "review artifact provenance_records must be an array of objects"
            )
        return ReviewArtifact(
            reference_date=stored_reference,
            effective_date=stored_effective,
            index_id=self.index_id,
            universe_id=str(payload.get("universe_id", "")),
            constituents=_decode_frame(payload["constituents"]),
            daily=None if payload.get("daily") is None else _decode_frame(payload["daily"]),
            diagnostics=dict(diagnostics),
            provenance_records=tuple(
                dict(item) for item in provenance_records
            ),
        )

    def _load_manifest(self, *, required: bool) -> dict[str, Any]:
        if not self.manifest_path.exists():
            if required:
                raise CacheMissError(
                    f"workspace manifest does not exist for {self.workspace_name!r} "
                    f"and fingerprint {self.fingerprint}"
                )
            return self._new_manifest()
        raw = _read_bytes(self.manifest_path)
        manifest, _ = _decode_envelope(raw, self.manifest_path)
        if (
            manifest.get("schema_version") != WORKSPACE_SCHEMA_VERSION
            or manifest.get("workspace_name") != self.workspace_name
            or manifest.get("fingerprint") != self.fingerprint
            or manifest.get("index_id") != self.index_id
        ):
            raise CacheIntegrityError("workspace manifest identity does not match its path")
        return manifest

    def _update_manifest(self, update) -> None:
        with _MANIFEST_LOCK:
            manifest = self._load_manifest(required=False)
            update(manifest)
            manifest["updated_at"] = _utc_now()
            raw, _ = _encode_envelope(manifest)
            _atomic_write(self.manifest_path, raw)

    def _new_manifest(self) -> dict[str, Any]:
        now = _utc_now()
        return {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "workspace_name": self.workspace_name,
            "fingerprint": self.fingerprint,
            "index_id": self.index_id,
            "methodology": self.methodology_name,
            "configuration_digest": self.configuration_digest,
            "data_revision": self.data_revision,
            "created_at": now,
            "updated_at": now,
            "reviews": {},
            "artifacts": {},
            "requests": [],
        }

    def _generic_path(
        self,
        stage: str,
        key: str,
        name: str,
        *,
        suffix: str,
    ) -> tuple[Path, str]:
        safe_stage = _validate_artifact_token("stage", stage)
        safe_key = _validate_artifact_token("key", key)
        safe_name = _validate_artifact_token("name", name)
        path = self.workspace_path.joinpath(
            "artifacts",
            safe_stage,
            safe_key,
            f"{safe_name}.{suffix}",
        )
        return path, f"{safe_stage}/{safe_key}/{safe_name}.{suffix}"

    def _review_key(self, reference_date: pd.Timestamp, effective_date: pd.Timestamp) -> str:
        return f"{reference_date.date().isoformat()}__{effective_date.date().isoformat()}"

    def _review_path(self, review_key: str) -> Path:
        return self.workspace_path.joinpath("reviews", f"{review_key}.json")


def _encode_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame artifacts must be pandas DataFrames")
    working = frame.copy()
    original_names = list(working.index.names)
    safe_names = [
        name if name is not None else f"__index_level_{position}__"
        for position, name in enumerate(original_names)
    ]
    working.index = working.index.set_names(safe_names)
    table = working.reset_index().to_json(
        orient="table",
        date_format="iso",
        date_unit="ns",
        index=False,
        default_handler=str,
    )
    return {
        "index_columns": safe_names,
        "index_names": original_names,
        "table": table,
    }


def _decode_frame(encoded: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_json(StringIO(encoded["table"]), orient="table")
    index_columns = list(encoded.get("index_columns", []))
    if index_columns:
        frame = frame.set_index(index_columns, verify_integrity=True)
        frame.index = frame.index.set_names(list(encoded.get("index_names", index_columns)))
    return frame


def _encode_envelope(payload: Mapping[str, Any]) -> tuple[bytes, str]:
    canonical_payload = _canonical_json(payload)
    payload_checksum = sha256(canonical_payload).hexdigest()
    envelope = {
        "checksum": payload_checksum,
        "payload": payload,
    }
    return _canonical_json(envelope), payload_checksum


def _decode_envelope(raw: bytes, path: Path) -> tuple[dict[str, Any], str]:
    try:
        envelope = json.loads(raw.decode("utf-8"))
        payload = envelope["payload"]
        expected = envelope["checksum"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CacheIntegrityError(f"artifact envelope is invalid: {path}") from exc
    observed = sha256(_canonical_json(payload)).hexdigest()
    if observed != expected:
        raise CacheIntegrityError(f"artifact payload checksum is invalid: {path}")
    if not isinstance(payload, dict):
        raise CacheIntegrityError(f"artifact payload must be an object: {path}")
    return payload, observed


def _atomic_write(path: Path, raw: bytes) -> None:
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


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise CacheMissError(f"workspace artifact does not exist: {path}") from exc


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cache configuration must not contain non-finite numbers")
        return value
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonicalize(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=lambda item: _canonical_json(item))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _canonicalize(scalar())
        except (TypeError, ValueError):
            pass
    if callable(value):
        return {"callable": _qualified_type_name(value)}
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            "type": _qualified_type_name(value),
            "configuration": {
                key: _canonicalize(item)
                for key, item in sorted(attributes.items())
                if not key.startswith("_")
            },
        }
    return {"type": _qualified_type_name(value)}


# Preserve the British-English private name for compatibility.
_canonicalise = _canonicalize


def _qualified_type_name(value: Any) -> str:
    target = value if isinstance(value, type) else type(value)
    if callable(value) and not isinstance(value, type) and not hasattr(value, "__dict__"):
        target = value
    module = getattr(target, "__module__", "")
    name = getattr(target, "__qualname__", getattr(target, "__name__", target.__class__.__name__))
    return f"{module}.{name}" if module else str(name)


def _normalize_date(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value).normalize()
    if pd.isna(timestamp):
        raise ValueError("review dates must not be null")
    return timestamp


def _validate_artifact_token(label: str, value: str) -> str:
    if not isinstance(value, str) or not _ARTIFACT_TOKEN.fullmatch(value):
        raise ValueError(
            f"{label} must be 1-128 characters using only letters, numbers, "
            "periods, underscores, and hyphens"
        )
    if value in {".", ".."}:
        raise ValueError(f"{label} must not be a relative path marker")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


register_review_store_factory(WorkspaceStore)


__all__ = [
    "ArtifactMetadata",
    "CacheIntegrityError",
    "CacheMissError",
    "CachePolicy",
    "CacheSource",
    "LoadedReview",
    "ReviewArtifact",
    "WORKSPACE_ROOT_ENV",
    "WorkspaceError",
    "WorkspaceStore",
    "build_run_fingerprint",
    "canonical_digest",
    "clear_memory_cache",
    "get_workspace_root",
    "validate_workspace_name",
]
