"""Artifact bindings and metadata for named research workspaces."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .artifacts import physical_artifact_identity
from .caches import CacheStage
from .catalog import validate_catalog_token
from .locking import exclusive_file_lock
from .manifests import (
    ArtifactRef,
    ManifestIntegrityError,
    read_envelope,
    utc_now,
    validate_digest,
    write_envelope,
)


class ArtifactOperations:
    """Internal artifact methods composed into :class:`WorkspaceRepository`."""

    def save_frame(
        self,
        artifact_type: str,
        frame: pd.DataFrame,
        *,
        sort_by: Sequence[str] | None = None,
    ) -> ArtifactRef:
        """Save a Parquet artifact and register it independently of a run."""

        reference = self.artifact_store.save_frame(
            artifact_type,
            frame,
            sort_by=sort_by,
        )
        self._write_artifact_metadata(reference)
        with self._connect() as connection:
            self._register_artifact(connection, reference)
        return reference

    def load_frame(self, reference: ArtifactRef) -> pd.DataFrame:
        """Load a verified Parquet or v1 JSON frame."""

        return self.artifact_store.load_frame(reference)

    def bind_artifact(
        self,
        *,
        stage: CacheStage | str,
        cache_key: str,
        name: str,
        artifact: ArtifactRef,
    ) -> None:
        """Bind an automatic calculation key to an immutable artifact.

        Rebinding updates only the catalog pointer. Previously referenced
        content-addressed files remain untouched for reproducibility.
        """

        selected_stage = CacheStage(stage)
        validate_digest("cache_key", cache_key)
        safe_name = validate_catalog_token("name", name)
        self._validate_artifact_reference(artifact)
        lock_path = self._binding_lock_path(
            selected_stage,
            cache_key,
            safe_name,
        )
        with exclusive_file_lock(lock_path):
            self._write_binding_metadata(
                stage=selected_stage,
                cache_key=cache_key,
                name=safe_name,
                artifact=artifact,
            )
            with self._connect() as connection:
                self._register_artifact(connection, artifact)
                connection.execute(
                    """
                    INSERT INTO artifact_bindings (
                        stage,
                        cache_key,
                        name,
                        content_digest,
                        file_checksum,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(stage, cache_key, name) DO UPDATE SET
                        content_digest=excluded.content_digest,
                        file_checksum=excluded.file_checksum,
                        updated_at=excluded.updated_at
                    """,
                    (
                        selected_stage.value,
                        cache_key,
                        safe_name,
                        artifact.content_digest,
                        artifact.file_checksum,
                        utc_now(),
                    ),
                )

    def resolve_artifact(
        self,
        *,
        stage: CacheStage | str,
        cache_key: str,
        name: str,
    ) -> ArtifactRef | None:
        """Resolve an internal cache key without requiring a user artifact ID."""

        selected_stage = CacheStage(stage)
        validate_digest("cache_key", cache_key)
        safe_name = validate_catalog_token("name", name)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    a.artifact_type,
                    a.content_digest,
                    a.file_checksum,
                    a.relative_path,
                    a.format,
                    a.schema_version,
                    a.size_bytes
                FROM artifact_bindings AS b
                JOIN artifacts AS a
                  ON a.content_digest = b.content_digest
                 AND a.file_checksum = b.file_checksum
                WHERE b.stage = ? AND b.cache_key = ? AND b.name = ?
                """,
                (selected_stage.value, cache_key, safe_name),
            ).fetchone()
        if row is None:
            return None
        reference = ArtifactRef(
            artifact_type=row[0],
            content_digest=row[1],
            file_checksum=row[2],
            relative_path=row[3],
            format=row[4],
            schema_version=int(row[5]),
            size_bytes=int(row[6]),
        )
        binding_path = self._binding_metadata_path(
            selected_stage,
            cache_key,
            safe_name,
        )
        if binding_path.is_file():
            (
                metadata_stage,
                metadata_key,
                metadata_name,
                bound_reference,
            ) = self._read_binding_metadata(binding_path)
            if (
                metadata_stage is not selected_stage
                or metadata_key != cache_key
                or metadata_name != safe_name
                or physical_artifact_identity(bound_reference)
                != physical_artifact_identity(reference)
            ):
                raise ManifestIntegrityError(
                    "binding sidecar does not match its catalog row"
                )
            self._validate_artifact_reference(bound_reference)
            return bound_reference
        # Catalogs created before binding sidecars remain readable. A subsequent
        # bind or catalog rebuild writes the durable sidecar representation.
        self._validate_artifact_reference(reference)
        return reference


    def _write_artifact_metadata(self, artifact: ArtifactRef) -> None:
        path = self._artifact_metadata_path(
            artifact.content_digest,
            artifact.file_checksum,
        )
        payload = {
            "artifact": {
                "artifact_type": artifact.artifact_type,
                "content_digest": artifact.content_digest,
                "file_checksum": artifact.file_checksum,
                "relative_path": artifact.relative_path,
                "format": artifact.format,
                "schema_version": artifact.schema_version,
                "size_bytes": artifact.size_bytes,
            }
        }
        write_envelope(path, payload)

    def _read_artifact_metadata(self, path: Path) -> ArtifactRef:
        payload = read_envelope(path)
        try:
            artifact = ArtifactRef.from_dict(payload["artifact"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestIntegrityError(
                f"artifact metadata is invalid: {path}"
            ) from exc
        expected = self._artifact_metadata_path(
            artifact.content_digest,
            artifact.file_checksum,
        )
        if path.resolve() != expected:
            raise ManifestIntegrityError(
                "artifact metadata identity does not match its fixed path"
            )
        return artifact

    def _write_binding_metadata(
        self,
        *,
        stage: CacheStage,
        cache_key: str,
        name: str,
        artifact: ArtifactRef,
    ) -> None:
        path = self._binding_metadata_path(stage, cache_key, name)
        payload = {
            "stage": stage.value,
            "cache_key": cache_key,
            "name": name,
            "artifact": {
                "artifact_type": artifact.artifact_type,
                "content_digest": artifact.content_digest,
                "file_checksum": artifact.file_checksum,
                "relative_path": artifact.relative_path,
                "format": artifact.format,
                "schema_version": artifact.schema_version,
                "size_bytes": artifact.size_bytes,
            },
        }
        write_envelope(path, payload)

    def _read_binding_metadata(
        self,
        path: Path,
    ) -> tuple[CacheStage, str, str, ArtifactRef]:
        payload = read_envelope(path)
        try:
            stage = CacheStage(payload["stage"])
            cache_key = str(payload["cache_key"])
            name = validate_catalog_token("name", str(payload["name"]))
            artifact = ArtifactRef.from_dict(payload["artifact"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestIntegrityError(
                f"binding metadata is invalid: {path}"
            ) from exc
        validate_digest("cache_key", cache_key)
        expected = self._binding_metadata_path(stage, cache_key, name)
        if path.resolve() != expected:
            raise ManifestIntegrityError(
                "binding metadata identity does not match its fixed path"
            )
        return stage, cache_key, name, artifact

    def _artifact_metadata_path(
        self,
        content_digest: str,
        file_checksum: str,
    ) -> Path:
        validate_digest("content_digest", content_digest)
        validate_digest("file_checksum", file_checksum)
        return self.workspace_path.joinpath(
            "objects",
            "metadata",
            content_digest[:2],
            content_digest,
            f"{file_checksum}.json",
        ).resolve()

    def _binding_metadata_path(
        self,
        stage: CacheStage,
        cache_key: str,
        name: str,
    ) -> Path:
        validate_digest("cache_key", cache_key)
        safe_name = validate_catalog_token("name", name)
        return self.workspace_path.joinpath(
            "bindings",
            stage.value,
            cache_key,
            f"{safe_name}.json",
        ).resolve()

    def _binding_lock_path(
        self,
        stage: CacheStage,
        cache_key: str,
        name: str,
    ) -> Path:
        validate_digest("cache_key", cache_key)
        safe_name = validate_catalog_token("name", name)
        return self.workspace_path.joinpath(
            ".locks",
            "bindings",
            stage.value,
            cache_key,
            f"{safe_name}.lock",
        ).resolve()


    def _validate_artifact_reference(self, artifact: ArtifactRef) -> None:
        validate_digest("artifact content_digest", artifact.content_digest)
        validate_digest("artifact file_checksum", artifact.file_checksum)
        candidate = self.workspace_path.joinpath(artifact.relative_path).resolve()
        if not candidate.is_relative_to(self.workspace_path):
            raise ManifestIntegrityError("artifact reference escapes its workspace")
        if not candidate.is_file():
            raise ManifestIntegrityError(
                f"artifact reference does not exist in its workspace: {candidate}"
            )



__all__ = ["ArtifactOperations"]
