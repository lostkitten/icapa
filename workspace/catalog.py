"""SQLite catalog ownership for named research workspaces."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3
from typing import Iterator

from .manifests import (
    ArtifactRef,
    ManifestIntegrityError,
    utc_now,
)
from .locking import exclusive_file_lock


_CATALOG_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_catalog_token(label: str, value: str) -> str:
    """Validate a safe catalog key or binding name."""

    if not isinstance(value, str) or not _CATALOG_TOKEN.fullmatch(value):
        raise ValueError(
            f"{label} must use only letters, numbers, periods, underscores, "
            "and hyphens"
        )
    return value


class CatalogOperations:
    """Internal catalog methods composed into :class:`WorkspaceRepository`."""

    def _initialize_catalog(self) -> None:
        lock_path = self.workspace_path.joinpath(".catalog-initialization.lock")
        with exclusive_file_lock(lock_path):
            with self._connect(ensure_wal=True) as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS manifests (
                        execution_id TEXT PRIMARY KEY,
                        definition_fingerprint TEXT NOT NULL,
                        request_fingerprint TEXT NOT NULL,
                        result_fingerprint TEXT,
                        status TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        completed_at TEXT
                    );

                    CREATE INDEX IF NOT EXISTS manifests_definition_created
                    ON manifests(definition_fingerprint, created_at);

                    CREATE TABLE IF NOT EXISTS artifacts (
                        content_digest TEXT NOT NULL,
                        file_checksum TEXT NOT NULL,
                        artifact_type TEXT NOT NULL,
                        format TEXT NOT NULL,
                        schema_version INTEGER NOT NULL,
                        relative_path TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(content_digest, file_checksum)
                    );

                    CREATE TABLE IF NOT EXISTS manifest_artifacts (
                        execution_id TEXT NOT NULL,
                        content_digest TEXT NOT NULL,
                        file_checksum TEXT NOT NULL,
                        artifact_type TEXT NOT NULL,
                        PRIMARY KEY(
                            execution_id,
                            content_digest,
                            file_checksum,
                            artifact_type
                        ),
                        FOREIGN KEY(execution_id) REFERENCES manifests(execution_id)
                    );

                    CREATE TABLE IF NOT EXISTS artifact_bindings (
                        stage TEXT NOT NULL,
                        cache_key TEXT NOT NULL,
                        name TEXT NOT NULL,
                        content_digest TEXT NOT NULL,
                        file_checksum TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(stage, cache_key, name),
                        FOREIGN KEY(content_digest, file_checksum)
                            REFERENCES artifacts(content_digest, file_checksum)
                    );

                    CREATE TABLE IF NOT EXISTS artifact_health (
                        content_digest TEXT NOT NULL,
                        file_checksum TEXT NOT NULL,
                        status TEXT NOT NULL,
                        checked_at TEXT NOT NULL,
                        error_type TEXT,
                        PRIMARY KEY(content_digest, file_checksum),
                        FOREIGN KEY(content_digest, file_checksum)
                            REFERENCES artifacts(content_digest, file_checksum)
                    );
                    """
                )

    @contextmanager
    def _connect(
        self,
        *,
        ensure_wal: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.catalog_path, timeout=30.0)
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            if ensure_wal:
                current_mode = connection.execute(
                    "PRAGMA journal_mode"
                ).fetchone()
                if current_mode is None or str(current_mode[0]).lower() != "wal":
                    connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _register_artifact(
        self,
        connection: sqlite3.Connection,
        artifact: ArtifactRef,
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO artifacts (
                content_digest,
                file_checksum,
                artifact_type,
                format,
                schema_version,
                relative_path,
                size_bytes,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.content_digest,
                artifact.file_checksum,
                artifact.artifact_type,
                artifact.format,
                artifact.schema_version,
                artifact.relative_path,
                artifact.size_bytes,
                utc_now(),
            ),
        )


    def _catalog_path(self, relative_path: str) -> Path:
        candidate = self.workspace_path.joinpath(relative_path).resolve()
        if not candidate.is_relative_to(self.workspace_path):
            raise ManifestIntegrityError("catalog path escapes its workspace")
        return candidate




__all__ = ["CatalogOperations"]
