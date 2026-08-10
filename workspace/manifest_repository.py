"""Run-manifest persistence for named research workspaces."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .caches import CacheOptions, CacheStage
from .identity import (
    automatic_digest,
    automatic_runtime_identity,
    canonical_json_bytes,
    canonicalize,
)
from .manifests import (
    ArtifactRef,
    ManifestIntegrityError,
    RUN_MANIFEST_SCHEMA_VERSION,
    RunManifest,
    RunManifestRef,
    atomic_write,
    merge_artifacts,
    merge_input_digests,
    result_input_identities,
    utc_now,
    validate_digest,
)


class ManifestOperations:
    """Internal manifest methods composed into :class:`WorkspaceRepository`."""

    def start_run(
        self,
        *,
        index_id: str,
        definition: Mapping[str, Any],
        request: Mapping[str, Any],
        request_identity: Mapping[str, Any] | None = None,
        providers: Sequence[Mapping[str, Any]] = (),
        calendar: Mapping[str, Any] | None = None,
        cache: CacheOptions | None = None,
    ) -> RunManifest:
        """Create and persist an automatically identified running manifest."""

        if not isinstance(index_id, str) or not index_id.strip():
            raise ValueError("index_id must not be empty")
        selected_cache = cache or CacheOptions.off()
        software = automatic_runtime_identity()
        safe_calendar = canonicalize(dict(calendar or {}))
        safe_providers = tuple(
            canonicalize(dict(provider)) for provider in providers
        )
        safe_request = canonicalize(dict(request))
        safe_request_identity = canonicalize(
            dict(request if request_identity is None else request_identity)
        )
        definition_fingerprint = automatic_digest(
            {
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                "index_id": index_id,
                "definition": canonicalize(dict(definition)),
                "calendar_semantics": safe_calendar,
                "software": software,
            }
        )
        request_fingerprint = automatic_digest(
            {
                "definition_fingerprint": definition_fingerprint,
                "request": safe_request_identity,
            }
        )
        execution_id = uuid4().hex
        manifest = RunManifest(
            schema_version=RUN_MANIFEST_SCHEMA_VERSION,
            execution_id=execution_id,
            status="running",
            definition_fingerprint=definition_fingerprint,
            request_fingerprint=request_fingerprint,
            result_fingerprint=None,
            workspace_name=self.workspace_name,
            index_id=index_id,
            created_at=utc_now(),
            completed_at=None,
            software=software,
            providers=safe_providers,
            calendar=safe_calendar,
            request=safe_request,
            cache_decisions=tuple(
                {
                    "stage": stage.value,
                    "mode": selected_cache.mode_for(stage).value,
                }
                for stage in CacheStage
            ),
        )
        self.write_manifest(manifest)
        return manifest

    def complete_run(
        self,
        manifest: RunManifest,
        *,
        artifacts: Sequence[ArtifactRef] = (),
        input_digests: Sequence[Mapping[str, Any]] = (),
    ) -> RunManifest:
        """Persist a complete manifest and its automatic Merkle-style result ID."""

        self._validate_manifest_workspace(manifest)
        combined = merge_artifacts(manifest.artifacts, artifacts)
        combined_inputs = merge_input_digests(
            manifest.input_digests,
            input_digests,
        )
        result_fingerprint = automatic_digest(
            {
                "request_fingerprint": manifest.request_fingerprint,
                "inputs": result_input_identities(combined_inputs),
                "artifacts": [
                    {
                        "artifact_type": artifact.artifact_type,
                        "content_digest": artifact.content_digest,
                        "schema_version": artifact.schema_version,
                    }
                    for artifact in sorted(
                        combined,
                        key=lambda item: (
                            item.artifact_type,
                            item.content_digest,
                            item.file_checksum,
                        ),
                    )
                ],
            }
        )
        completed = replace(
            manifest,
            status="complete",
            result_fingerprint=result_fingerprint,
            completed_at=utc_now(),
            input_digests=combined_inputs,
            artifacts=combined,
            failure=None,
        )
        self.write_manifest(completed)
        return completed

    def fail_run(self, manifest: RunManifest, error: BaseException) -> RunManifest:
        """Record a sanitized failure without serializing provider exception data."""

        self._validate_manifest_workspace(manifest)
        failed = replace(
            manifest,
            status="failed",
            completed_at=utc_now(),
            failure={
                "error_type": type(error).__qualname__,
                "message": (
                    "Execution failed. The original exception is intentionally "
                    "not persisted because it may contain sensitive provider data."
                ),
            },
        )
        self.write_manifest(failed)
        return failed

    def write_manifest(self, manifest: RunManifest) -> RunManifestRef:
        """Atomically write one manifest and update the local SQLite catalog."""

        self._validate_manifest_workspace(manifest)
        self._ensure_definition_layout(
            manifest.definition_fingerprint
        )
        path = self._manifest_path(
            manifest.definition_fingerprint,
            manifest.execution_id,
        )
        payload = manifest.as_dict()
        payload_bytes = canonical_json_bytes(payload)
        envelope = {
            "checksum": sha256(payload_bytes).hexdigest(),
            "payload": payload,
        }
        atomic_write(path, canonical_json_bytes(envelope))
        relative_path = str(path.relative_to(self.workspace_path))
        with self._connect() as connection:
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
                ON CONFLICT(execution_id) DO UPDATE SET
                    result_fingerprint=excluded.result_fingerprint,
                    status=excluded.status,
                    relative_path=excluded.relative_path,
                    completed_at=excluded.completed_at
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
        return self.manifest_ref(manifest)

    def manifest_ref(self, manifest: RunManifest) -> RunManifestRef:
        self._validate_manifest_workspace(manifest)
        return RunManifestRef(
            workspace_name=self.workspace_name,
            definition_fingerprint=manifest.definition_fingerprint,
            execution_id=manifest.execution_id,
            path=str(
                self._manifest_path(
                    manifest.definition_fingerprint,
                    manifest.execution_id,
                )
            ),
        )

    def open_manifest(self, reference: RunManifestRef) -> RunManifest:
        """Load and verify a manifest without trusting its caller-supplied path."""

        if reference.workspace_name != self.workspace_name:
            raise ManifestIntegrityError(
                "manifest reference belongs to a different workspace"
            )
        path = self._manifest_path(
            reference.definition_fingerprint,
            reference.execution_id,
        )
        return self._read_manifest(path)

    def latest_manifest(
        self,
        *,
        definition_fingerprint: str | None = None,
    ) -> RunManifest | None:
        """Return the most recently created manifest, when one exists."""

        query = "SELECT relative_path FROM manifests"
        parameters: tuple[str, ...] = ()
        if definition_fingerprint is not None:
            validate_digest("definition_fingerprint", definition_fingerprint)
            query += " WHERE definition_fingerprint = ?"
            parameters = (definition_fingerprint,)
        query += " ORDER BY created_at DESC, execution_id DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        if row is None:
            return None
        return self._read_manifest(self._catalog_path(row[0]))

    def list_manifests(self) -> tuple[RunManifestRef, ...]:
        """List automatically generated execution references, newest first."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT definition_fingerprint, execution_id, relative_path
                FROM manifests
                ORDER BY created_at DESC, execution_id DESC
                """
            ).fetchall()
        return tuple(
            RunManifestRef(
                workspace_name=self.workspace_name,
                definition_fingerprint=row[0],
                execution_id=row[1],
                path=str(self._catalog_path(row[2])),
            )
            for row in rows
        )


    def _read_manifest(self, path: Path) -> RunManifest:
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            expected = envelope["checksum"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
            raise ManifestIntegrityError(f"manifest envelope is invalid: {path}") from exc
        observed = sha256(canonical_json_bytes(payload)).hexdigest()
        if observed != expected:
            raise ManifestIntegrityError(
                f"manifest checksum does not match its payload: {path}"
            )
        manifest = RunManifest.from_dict(payload)
        self._validate_manifest_workspace(manifest)
        expected_path = self._manifest_path(
            manifest.definition_fingerprint,
            manifest.execution_id,
        )
        if expected_path != path.resolve():
            raise ManifestIntegrityError(
                "manifest identity does not match its fixed workspace path"
            )
        return manifest


    def _validate_manifest_workspace(self, manifest: RunManifest) -> None:
        if manifest.workspace_name != self.workspace_name:
            raise ManifestIntegrityError(
                "manifest belongs to a different research workspace"
            )
        if manifest.schema_version != RUN_MANIFEST_SCHEMA_VERSION:
            raise ManifestIntegrityError("manifest schema version is not supported")
        validate_digest(
            "definition_fingerprint",
            manifest.definition_fingerprint,
        )
        validate_digest("request_fingerprint", manifest.request_fingerprint)
        if manifest.result_fingerprint is not None:
            validate_digest("result_fingerprint", manifest.result_fingerprint)
        if (
            not isinstance(manifest.execution_id, str)
            or len(manifest.execution_id) != 32
            or any(character not in "0123456789abcdef" for character in manifest.execution_id)
        ):
            raise ManifestIntegrityError("execution_id is not a generated UUID hex value")


    def _manifest_path(
        self,
        definition_fingerprint: str,
        execution_id: str,
    ) -> Path:
        validate_digest("definition_fingerprint", definition_fingerprint)
        if (
            len(execution_id) != 32
            or any(character not in "0123456789abcdef" for character in execution_id)
        ):
            raise ManifestIntegrityError("execution_id is invalid")
        return self.workspace_path.joinpath(
            "runs",
            definition_fingerprint,
            "executions",
            execution_id,
            "run_manifest.json",
        ).resolve()

    def _ensure_definition_layout(
        self,
        definition_fingerprint: str,
    ) -> None:
        """Create the stable per-definition research-stage namespaces."""

        validate_digest(
            "definition_fingerprint",
            definition_fingerprint,
        )
        definition_path = self.workspace_path.joinpath(
            "runs",
            definition_fingerprint,
        ).resolve()
        if not definition_path.is_relative_to(self.workspace_path):
            raise ManifestIntegrityError(
                "definition path escapes its workspace"
            )
        for name in (
            "executions",
            "reviews",
            "simulations",
            "analytics",
            "reports",
        ):
            definition_path.joinpath(name).mkdir(
                parents=True,
                exist_ok=True,
            )



__all__ = ["ManifestOperations"]
