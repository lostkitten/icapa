"""Verification, invalidation, pruning, and catalog rebuild operations."""

from __future__ import annotations

from typing import Any

from .caches import CacheStage
from .catalog import validate_catalog_token
from .manifests import (
    ArtifactRef,
    ManifestIntegrityError,
    utc_now,
    validate_digest,
)


class MaintenanceOperations:
    """Internal maintenance methods composed into :class:`WorkspaceRepository`."""

    def verify(self) -> dict[str, Any]:
        """Verify catalog rows, immutable objects, and rebuild sidecars."""

        manifest_failures: list[dict[str, str]] = []
        artifact_failures: list[dict[str, str]] = []
        sidecar_failures: list[dict[str, str]] = []
        catalog_failures: list[dict[str, str]] = []
        manifests = self.list_manifests()
        for reference in manifests:
            try:
                self.open_manifest(reference)
            except (ManifestIntegrityError, OSError) as exc:
                manifest_failures.append(
                    {
                        "execution_id": reference.execution_id,
                        "error_type": type(exc).__qualname__,
                    }
                )

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    artifact_type,
                    content_digest,
                    file_checksum,
                    relative_path,
                    format,
                    schema_version,
                    size_bytes
                FROM artifacts
                ORDER BY content_digest, file_checksum
                """
            ).fetchall()
            binding_rows = connection.execute(
                """
                SELECT
                    stage,
                    cache_key,
                    name,
                    content_digest,
                    file_checksum
                FROM artifact_bindings
                ORDER BY stage, cache_key, name
                """
            ).fetchall()
        catalog_artifacts = {
            (str(row[1]), str(row[2])) for row in rows
        }
        catalog_bindings = {
            (
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[4]),
            )
            for row in binding_rows
        }
        for row in rows:
            reference = ArtifactRef(
                artifact_type=row[0],
                content_digest=row[1],
                file_checksum=row[2],
                relative_path=row[3],
                format=row[4],
                schema_version=int(row[5]),
                size_bytes=int(row[6]),
            )
            status = "verified"
            error_type: str | None = None
            try:
                self.load_frame(reference)
            except Exception as exc:
                status = "corrupt"
                error_type = type(exc).__qualname__
                artifact_failures.append(
                    {
                        "content_digest": reference.content_digest,
                        "file_checksum": reference.file_checksum,
                        "error_type": error_type,
                    }
                )
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO artifact_health (
                        content_digest,
                        file_checksum,
                        status,
                        checked_at,
                        error_type
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(content_digest, file_checksum) DO UPDATE SET
                        status=excluded.status,
                        checked_at=excluded.checked_at,
                        error_type=excluded.error_type
                    """,
                    (
                        reference.content_digest,
                        reference.file_checksum,
                        status,
                        utc_now(),
                        error_type,
                    ),
                )
        metadata_seen: set[tuple[str, str]] = set()
        for path in sorted(
            self.workspace_path.glob("objects/metadata/*/*/*.json")
        ):
            try:
                artifact = self._read_artifact_metadata(path.resolve())
                self._validate_artifact_reference(artifact)
                self.load_frame(artifact)
                key = (artifact.content_digest, artifact.file_checksum)
                metadata_seen.add(key)
                if key not in catalog_artifacts:
                    catalog_failures.append(
                        {
                            "record_type": "artifact_metadata",
                            "error_type": "MissingCatalogRow",
                        }
                    )
            except Exception as exc:
                sidecar_failures.append(
                    {
                        "record_type": "artifact_metadata",
                        "error_type": type(exc).__qualname__,
                    }
                )
        for content_digest, file_checksum in catalog_artifacts.difference(
            metadata_seen
        ):
            catalog_failures.append(
                {
                    "record_type": "artifact_metadata",
                    "content_digest": content_digest,
                    "file_checksum": file_checksum,
                    "error_type": "MissingArtifactMetadataSidecar",
                }
            )

        bindings_seen: set[tuple[str, str, str, str, str]] = set()
        for path in sorted(self.workspace_path.glob("bindings/*/*/*.json")):
            try:
                stage, cache_key, name, artifact = (
                    self._read_binding_metadata(path.resolve())
                )
                self._validate_artifact_reference(artifact)
                self.load_frame(artifact)
                key = (
                    stage.value,
                    cache_key,
                    name,
                    artifact.content_digest,
                    artifact.file_checksum,
                )
                bindings_seen.add(key)
                if key not in catalog_bindings:
                    catalog_failures.append(
                        {
                            "record_type": "artifact_binding",
                            "error_type": "MissingOrDifferentCatalogRow",
                        }
                    )
            except Exception as exc:
                sidecar_failures.append(
                    {
                        "record_type": "artifact_binding",
                        "error_type": type(exc).__qualname__,
                    }
                )
        for _ in catalog_bindings.difference(bindings_seen):
            catalog_failures.append(
                {
                    "record_type": "artifact_binding",
                    "error_type": "MissingBindingSidecar",
                }
            )
        return {
            "workspace_name": self.workspace_name,
            "status": (
                "ok"
                if not manifest_failures
                and not artifact_failures
                and not sidecar_failures
                and not catalog_failures
                else "corrupt"
            ),
            "manifest_count": len(manifests),
            "artifact_count": len(rows),
            "artifact_metadata_count": len(metadata_seen),
            "binding_count": len(binding_rows),
            "manifest_failures": manifest_failures,
            "artifact_failures": artifact_failures,
            "sidecar_failures": sidecar_failures,
            "catalog_failures": catalog_failures,
        }

    def invalidate(
        self,
        *,
        stage: CacheStage | str,
        cache_key: str | None = None,
        name: str | None = None,
    ) -> int:
        """Remove matching reusable bindings while preserving immutable objects."""

        selected_stage = CacheStage(stage)
        conditions = ["stage = ?"]
        parameters: list[str] = [selected_stage.value]
        if cache_key is not None:
            validate_digest("cache_key", cache_key)
            conditions.append("cache_key = ?")
            parameters.append(cache_key)
        if name is not None:
            safe_name = validate_catalog_token("name", name)
            conditions.append("name = ?")
            parameters.append(safe_name)
        where = " AND ".join(conditions)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT stage, cache_key, name FROM artifact_bindings WHERE {where}",
                tuple(parameters),
            ).fetchall()
            connection.execute(
                f"DELETE FROM artifact_bindings WHERE {where}",
                tuple(parameters),
            )
        for row_stage, row_key, row_name in rows:
            path = self._binding_metadata_path(
                CacheStage(row_stage),
                row_key,
                row_name,
            )
            if path.is_file():
                path.unlink()
        return len(rows)

    def prune(self, *, dry_run: bool = True) -> dict[str, Any]:
        """Find or remove immutable objects no longer referenced by any run or binding."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    a.content_digest,
                    a.file_checksum,
                    a.relative_path,
                    a.size_bytes
                FROM artifacts AS a
                LEFT JOIN manifest_artifacts AS m
                  ON m.content_digest = a.content_digest
                 AND m.file_checksum = a.file_checksum
                LEFT JOIN artifact_bindings AS b
                  ON b.content_digest = a.content_digest
                 AND b.file_checksum = a.file_checksum
                WHERE m.execution_id IS NULL AND b.cache_key IS NULL
                ORDER BY a.content_digest, a.file_checksum
                """
            ).fetchall()
        candidates = [
            {
                "content_digest": row[0],
                "file_checksum": row[1],
                "relative_path": row[2],
                "size_bytes": int(row[3]),
            }
            for row in rows
        ]
        if not dry_run and candidates:
            for candidate in candidates:
                path = self._catalog_path(candidate["relative_path"])
                if path.is_file():
                    path.unlink()
                metadata_path = self._artifact_metadata_path(
                    candidate["content_digest"],
                    candidate["file_checksum"],
                )
                if metadata_path.is_file():
                    metadata_path.unlink()
            with self._connect() as connection:
                connection.executemany(
                    """
                    DELETE FROM artifact_health
                    WHERE content_digest = ? AND file_checksum = ?
                    """,
                    [
                        (item["content_digest"], item["file_checksum"])
                        for item in candidates
                    ],
                )
                connection.executemany(
                    """
                    DELETE FROM artifacts
                    WHERE content_digest = ? AND file_checksum = ?
                    """,
                    [
                        (item["content_digest"], item["file_checksum"])
                        for item in candidates
                    ],
                )
        return {
            "workspace_name": self.workspace_name,
            "dry_run": bool(dry_run),
            "candidate_count": len(candidates),
            "size_bytes": sum(item["size_bytes"] for item in candidates),
            "candidates": candidates,
        }

    def rebuild_catalog(self) -> dict[str, int]:
        """Rebuild catalog rows from checksummed manifests and metadata sidecars."""

        manifest_paths = tuple(
            sorted(self.workspace_path.glob("runs/*/executions/*/run_manifest.json"))
        )
        artifact_metadata_paths = tuple(
            sorted(self.workspace_path.glob("objects/metadata/*/*/*.json"))
        )
        binding_paths = tuple(
            sorted(self.workspace_path.glob("bindings/*/*/*.json"))
        )
        manifests = [self._read_manifest(path.resolve()) for path in manifest_paths]
        standalone_artifacts = [
            self._read_artifact_metadata(path) for path in artifact_metadata_paths
        ]
        bindings = [
            self._read_binding_metadata(path) for path in binding_paths
        ]

        with self._connect() as connection:
            connection.execute("DELETE FROM manifest_artifacts")
            connection.execute("DELETE FROM artifact_bindings")
            connection.execute("DELETE FROM artifact_health")
            connection.execute("DELETE FROM manifests")
            connection.execute("DELETE FROM artifacts")
            for artifact in standalone_artifacts:
                self._validate_artifact_reference(artifact)
                self._register_artifact(connection, artifact)
            for manifest in manifests:
                relative_path = str(
                    self._manifest_path(
                        manifest.definition_fingerprint,
                        manifest.execution_id,
                    ).relative_to(self.workspace_path)
                )
                connection.execute(
                    """
                    INSERT INTO manifests (
                        execution_id,
                        definition_fingerprint,
                        request_fingerprint,
                        result_fingerprint,
                        status,
                        relative_path,
                        created_at,
                        completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        manifest.execution_id,
                        manifest.definition_fingerprint,
                        manifest.request_fingerprint,
                        manifest.result_fingerprint,
                        manifest.status,
                        relative_path,
                        manifest.created_at,
                        manifest.completed_at,
                    ),
                )
                for artifact in manifest.artifacts:
                    self._validate_artifact_reference(artifact)
                    self._register_artifact(connection, artifact)
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO manifest_artifacts (
                            execution_id,
                            content_digest,
                            file_checksum,
                            artifact_type
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            manifest.execution_id,
                            artifact.content_digest,
                            artifact.file_checksum,
                            artifact.artifact_type,
                        ),
                    )
            for stage, cache_key, name, artifact in bindings:
                self._validate_artifact_reference(artifact)
                self._register_artifact(connection, artifact)
                connection.execute(
                    """
                    INSERT OR REPLACE INTO artifact_bindings (
                        stage,
                        cache_key,
                        name,
                        content_digest,
                        file_checksum,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stage.value,
                        cache_key,
                        name,
                        artifact.content_digest,
                        artifact.file_checksum,
                        utc_now(),
                    ),
                )
        return {
            "manifests": len(manifests),
            "artifacts": len(
                {
                    (artifact.content_digest, artifact.file_checksum)
                    for artifact in (
                        *standalone_artifacts,
                        *(item for manifest in manifests for item in manifest.artifacts),
                        *(binding[3] for binding in bindings),
                    )
                }
            ),
            "bindings": len(bindings),
        }



__all__ = ["MaintenanceOperations"]
