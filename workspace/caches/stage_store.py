"""Parquet-backed compatibility store for research calculation stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..artifacts import ArtifactIntegrityError
from .models import CacheStage
from .diagnostic_codec import (
    DiagnosticDecodingMixin,
    _prepare_diagnostics,
)
from .diagnostic_tables import _diagnostic_table_frame
from .diagnostic_types import (
    register_review_diagnostic_enum,
)
from ..identity import automatic_digest, canonical_json_bytes, canonicalize
from ..locking import exclusive_file_lock
from ..manifests import ArtifactRef
from ..readers import (
    ArtifactMetadata,
    CacheMissError,
    CacheSource,
    LoadedReview,
    ReviewArtifact,
)
from ..repository import ManifestIntegrityError, WorkspaceRepository


_SCHEMA_VERSION = 2
_LEGACY_SCHEMA_VERSION = 1
_REVIEW_COMMIT = "review-commit"
_JSON_PREFIX = "json-"
_DIAGNOSTIC_BINDING_PREFIX = "review-diagnostic"


class ParquetStageStore(DiagnosticDecodingMixin):
    """Adapt v2 workspace artifacts to v1 stage-store call signatures.

    A checksummed Parquet commit object is bound last for multi-object review
    results. Readers therefore never accept an incomplete collection, and the
    commit records the exact immutable artifact references it expects.
    """

    def __init__(
        self,
        workspace: WorkspaceRepository | str,
        *,
        index_id: str,
        namespace_digest: str,
    ) -> None:
        self.workspace = (
            WorkspaceRepository.open(workspace)
            if isinstance(workspace, str)
            else workspace
        )
        if not isinstance(self.workspace, WorkspaceRepository):
            raise TypeError("workspace must be a WorkspaceRepository or workspace name")
        if not isinstance(index_id, str) or not index_id.strip():
            raise ValueError("index_id must not be empty")
        if (
            not isinstance(namespace_digest, str)
            or len(namespace_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in namespace_digest
            )
        ):
            raise ValueError("namespace_digest must be a lowercase SHA-256 digest")
        self.index_id = index_id
        self.namespace_digest = namespace_digest

    @property
    def workspace_name(self) -> str:
        return self.workspace.workspace_name

    @property
    def workspace_path(self) -> Path:
        return self.workspace.workspace_path

    @property
    def reports_path(self) -> Path:
        return self.workspace_path.joinpath("reports")

    def report_path(
        self,
        file_name: str,
        *,
        create_parent: bool = True,
    ) -> Path:
        """Return a safe report path below the named workspace."""

        if (
            not isinstance(file_name, str)
            or not file_name
            or Path(file_name).name != file_name
            or file_name in {".", ".."}
        ):
            raise ValueError("file_name must be a non-empty base name")
        if create_parent:
            self.reports_path.mkdir(parents=True, exist_ok=True)
        return self.reports_path.joinpath(file_name)

    def simulation_catalog_lock(self, namespace: str):
        """Return the infrastructure lock for one simulation catalogue."""

        if (
            not isinstance(namespace, str)
            or len(namespace) != 64
            or any(character not in "0123456789abcdef" for character in namespace)
        ):
            raise ValueError("simulation namespace must be a lowercase SHA-256 digest")
        return exclusive_file_lock(
            self.workspace_path.joinpath(
                ".locks",
                "simulation_segments",
                f"{namespace}.lock",
            )
        )

    def load_review(
        self,
        reference_date: object,
        effective_date: object,
    ) -> LoadedReview:
        """Load one complete, verified review result."""

        reference = _normalize_date(reference_date)
        effective = _normalize_date(effective_date)
        cache_key, commit_reference = self._resolve_review_commit(
            reference,
            effective,
        )
        if commit_reference is None:
            raise CacheMissError(
                f"workspace {self.workspace_name!r} has no cached review for "
                f"{reference.date()} / {effective.date()}"
            )
        commit = _decode_json_frame(self._load_reference(commit_reference))
        _validate_review_commit(
            commit,
            index_id=self.index_id,
            reference_date=reference,
            effective_date=effective,
        )
        constituent_reference = ArtifactRef.from_dict(commit["constituents"])
        constituents = self._load_reference(constituent_reference)
        daily_payload = commit.get("daily")
        daily = (
            None
            if daily_payload is None
            else self._load_reference(ArtifactRef.from_dict(daily_payload))
        )
        diagnostics = commit.get("diagnostics", {})
        provenance_records = commit.get("provenance_records", [])
        if not isinstance(provenance_records, list) or any(
            not isinstance(item, Mapping) for item in provenance_records
        ):
            raise ValueError(
                "review cache provenance_records must be an array of mappings"
            )
        if commit["schema_version"] == _LEGACY_SCHEMA_VERSION:
            if not isinstance(diagnostics, Mapping):
                raise ValueError("review cache diagnostics must be a mapping")
            restored_diagnostics = dict(diagnostics)
        else:
            restored_diagnostics = self._decode_diagnostic_value(
                diagnostics,
                cache_key=cache_key,
                path="diagnostics",
            )
            if not isinstance(restored_diagnostics, Mapping):
                raise ValueError("review cache diagnostics must be a mapping")
        return LoadedReview(
            artifact=ReviewArtifact(
                reference_date=reference,
                effective_date=effective,
                index_id=self.index_id,
                universe_id=str(commit.get("universe_id", "")),
                constituents=constituents,
                daily=daily,
                diagnostics=dict(restored_diagnostics),
                provenance_records=tuple(dict(item) for item in provenance_records),
            ),
            metadata=ArtifactMetadata(
                source=CacheSource.DISK,
                checksum=commit_reference.file_checksum,
                path=str(self.workspace_path.joinpath(commit_reference.relative_path)),
            ),
        )

    def save_review(self, artifact: ReviewArtifact) -> ArtifactMetadata:
        """Persist a review and atomically publish its commit binding."""

        if not isinstance(artifact, ReviewArtifact):
            raise TypeError("artifact must be a ReviewArtifact")
        reference = _normalize_date(artifact.reference_date)
        effective = _normalize_date(artifact.effective_date)
        if artifact.index_id != self.index_id:
            raise ValueError("review index_id does not match the store")
        diagnostic_payload, diagnostic_tables = _prepare_diagnostics(
            artifact.diagnostics or {}
        )
        provenance_records = tuple(artifact.provenance_records or ())
        if any(not isinstance(item, Mapping) for item in provenance_records):
            raise TypeError("review provenance records must be mappings")
        constituent_reference = self.workspace.save_frame(
            "review_constituents",
            artifact.constituents,
        )
        daily_reference = (
            None
            if artifact.daily is None
            else self.workspace.save_frame("review_daily", artifact.daily)
        )
        cache_key = self._review_key(reference, effective)
        for position, table in enumerate(diagnostic_tables):
            storage_frame, pandas_schema = _diagnostic_table_frame(
                table.value,
                path=table.path,
            )
            table_reference = self.workspace.save_frame(
                "review_diagnostic_table",
                storage_frame,
            )
            binding = (
                f"{_DIAGNOSTIC_BINDING_PREFIX}-"
                f"{table_reference.content_digest}-{position:04d}"
            )
            self.workspace.bind_artifact(
                stage=CacheStage.REVIEWS,
                cache_key=cache_key,
                name=binding,
                artifact=table_reference,
            )
            table.node.update(
                {
                    "binding": binding,
                    "artifact": asdict(table_reference),
                    "pandas_schema": pandas_schema,
                }
            )
            table.node.pop("position", None)
        commit = {
            "schema_version": _SCHEMA_VERSION,
            "kind": "review",
            "index_id": self.index_id,
            "reference_date": reference,
            "effective_date": effective,
            "universe_id": artifact.universe_id,
            "constituents": asdict(constituent_reference),
            "daily": (None if daily_reference is None else asdict(daily_reference)),
            "diagnostics": diagnostic_payload,
            "provenance_records": canonicalize(provenance_records),
        }
        commit_reference = self.workspace.save_frame(
            "review_commit",
            _json_frame(commit),
        )
        self.workspace.bind_artifact(
            stage=CacheStage.REVIEWS,
            cache_key=cache_key,
            name=_REVIEW_COMMIT,
            artifact=commit_reference,
        )
        return ArtifactMetadata(
            source=CacheSource.COMPUTED,
            checksum=commit_reference.file_checksum,
            path=str(self.workspace_path.joinpath(commit_reference.relative_path)),
        )

    def record_request(
        self,
        reviews: Sequence[Mapping[str, Any]],
        *,
        cache_policy: object,
    ) -> None:
        """Keep compatibility; v2 execution manifests record this request."""

        del reviews, cache_policy

    def save_frame(
        self,
        stage: str,
        key: str,
        name: str,
        frame: pd.DataFrame,
    ) -> ArtifactMetadata:
        """Save and bind one immutable stage DataFrame."""

        selected_stage = _cache_stage(stage)
        reference = self.workspace.save_frame(
            f"{selected_stage.value}.{name}",
            frame,
        )
        self.workspace.bind_artifact(
            stage=selected_stage,
            cache_key=key,
            name=name,
            artifact=reference,
        )
        return ArtifactMetadata(
            source=CacheSource.COMPUTED,
            checksum=reference.file_checksum,
            path=str(self.workspace_path.joinpath(reference.relative_path)),
        )

    def load_frame(
        self,
        stage: str,
        key: str,
        name: str,
    ) -> pd.DataFrame:
        """Resolve and verify one stage DataFrame."""

        selected_stage = _cache_stage(stage)
        try:
            reference = self.workspace.resolve_artifact(
                stage=selected_stage,
                cache_key=key,
                name=name,
            )
        except ManifestIntegrityError as exc:
            if "does not exist in its workspace" in str(exc):
                raise CacheMissError(
                    f"workspace artifact file is missing: "
                    f"{selected_stage.value}/{key}/{name}"
                ) from exc
            raise
        if reference is None:
            raise CacheMissError(
                f"workspace artifact is not cached: "
                f"{selected_stage.value}/{key}/{name}"
            )
        return self._load_reference(reference)

    def save_json(
        self,
        stage: str,
        key: str,
        name: str,
        value: Any,
    ) -> ArtifactMetadata:
        """Store compact metadata as a checksummed one-row Parquet object."""

        return self.save_frame(
            stage,
            key,
            f"{_JSON_PREFIX}{name}",
            _json_frame(value),
        )

    def load_json(self, stage: str, key: str, name: str) -> Any:
        """Load compact metadata written by :meth:`save_json`."""

        return _decode_json_frame(self.load_frame(stage, key, f"{_JSON_PREFIX}{name}"))

    def _review_key(
        self,
        reference_date: pd.Timestamp,
        effective_date: pd.Timestamp,
        *,
        schema_version: int = _SCHEMA_VERSION,
    ) -> str:
        return automatic_digest(
            {
                "schema_version": schema_version,
                "kind": "review",
                "namespace_digest": self.namespace_digest,
                "index_id": self.index_id,
                "reference_date": reference_date,
                "effective_date": effective_date,
            }
        )

    def _resolve_review_commit(
        self,
        reference_date: pd.Timestamp,
        effective_date: pd.Timestamp,
    ) -> tuple[str, ArtifactRef | None]:
        for schema_version in (_SCHEMA_VERSION, _LEGACY_SCHEMA_VERSION):
            cache_key = self._review_key(
                reference_date,
                effective_date,
                schema_version=schema_version,
            )
            try:
                reference = self.workspace.resolve_artifact(
                    stage=CacheStage.REVIEWS,
                    cache_key=cache_key,
                    name=_REVIEW_COMMIT,
                )
            except ManifestIntegrityError as exc:
                if "does not exist in its workspace" in str(exc):
                    raise CacheMissError(
                        "cached review commit file is missing"
                    ) from exc
                raise
            if reference is not None:
                return cache_key, reference
        return (
            self._review_key(
                reference_date,
                effective_date,
            ),
            None,
        )

    def _load_reference(self, reference: ArtifactRef) -> pd.DataFrame:
        path = self.workspace_path.joinpath(reference.relative_path)
        if not path.is_file():
            raise CacheMissError(f"workspace artifact file is missing: {path}")
        try:
            return self.workspace.load_frame(reference)
        except ArtifactIntegrityError:
            raise


def _cache_stage(stage: str) -> CacheStage:
    aliases = {
        "simulation_segments": CacheStage.SIMULATION,
    }
    if stage in aliases:
        return aliases[stage]
    try:
        return CacheStage(stage)
    except ValueError as exc:
        raise ValueError(f"unsupported calculation cache stage: {stage}") from exc


def _json_frame(value: Any) -> pd.DataFrame:
    encoded = canonical_json_bytes(value).decode("utf-8")
    return pd.DataFrame({"payload_json": [encoded]})


def _decode_json_frame(frame: pd.DataFrame) -> Any:
    if (
        not isinstance(frame, pd.DataFrame)
        or list(frame.columns) != ["payload_json"]
        or len(frame) != 1
    ):
        raise ValueError("metadata artifact has an invalid schema")
    raw = frame.iloc[0]["payload_json"]
    if not isinstance(raw, str):
        raise ValueError("metadata artifact payload must be JSON text")
    return json.loads(raw)


def _normalize_date(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value).normalize()
    if pd.isna(result):
        raise ValueError("review dates must not be null")
    return result


def _validate_review_commit(
    value: Any,
    *,
    index_id: str,
    reference_date: pd.Timestamp,
    effective_date: pd.Timestamp,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("review cache commit must be a mapping")
    if (
        value.get("schema_version") not in {_LEGACY_SCHEMA_VERSION, _SCHEMA_VERSION}
        or value.get("kind") != "review"
        or value.get("index_id") != index_id
        or _normalize_date(value.get("reference_date")) != reference_date
        or _normalize_date(value.get("effective_date")) != effective_date
    ):
        raise ValueError("review cache commit identity does not match its key")


__all__ = [
    "ParquetStageStore",
    "register_review_diagnostic_enum",
]
