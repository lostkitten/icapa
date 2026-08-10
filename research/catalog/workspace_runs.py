"""High-level orchestration for reproducible index-research workflows.

The v1 backtesting, simulation, analytics, and reporting APIs remain
available independently.  This module composes them behind a small research
API while keeping cache reuse opt-in and evidence based.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ...workspace import RunManifest, RunManifestRef
from ..runners.identity import _optional_string
from ..runners.io import _atomic_json_write
from ..models import (
    CatalogRebuildResult,
    PruneResult,
    ResearchRun,
    ResearchStatus,
    ResearchWorkflowError,
    WorkspaceVerification,
)
from ..results import (
    _load_run_artifact_frames,
    _rehydrate_analytics,
    _rehydrate_backtest,
    _rehydrate_definition,
    _rehydrate_simulation,
)


class _ResearchRunCatalog:
    def list(
        self,
        *,
        include_invalidated: bool = False,
    ) -> tuple[RunManifestRef, ...]:
        """List run references, newest first."""

        references = self._workspace.list_manifests()
        if include_invalidated:
            return references
        return tuple(
            item for item in references if not self._is_invalidated(item.execution_id)
        )

    def open_run(
        self,
        reference: RunManifestRef | ResearchRun | str,
    ) -> RunManifest:
        """Open one verified manifest by reference, run, or execution ID."""

        if isinstance(reference, ResearchRun):
            reference = reference.manifest_ref
        if isinstance(reference, RunManifestRef):
            return self._workspace.open_manifest(reference)
        if not isinstance(reference, str) or not reference:
            raise TypeError("reference must be a manifest reference or execution ID")
        matches = [
            item
            for item in self.list(include_invalidated=True)
            if item.execution_id == reference
        ]
        if not matches:
            raise KeyError(f"research execution is not catalogued: {reference}")
        return self._workspace.open_manifest(matches[0])

    def load_run(
        self,
        reference: RunManifestRef | ResearchRun | str,
    ) -> ResearchRun:
        """Rehydrate persisted outputs without rerunning any calculation."""

        manifest = self.open_run(reference)
        if manifest.status != "complete":
            raise ResearchWorkflowError(
                "only complete research executions can be rehydrated"
            )
        frames = _load_run_artifact_frames(self._workspace, manifest)
        backtest = _rehydrate_backtest(manifest, frames)
        simulation = _rehydrate_simulation(manifest, frames)
        analytics = _rehydrate_analytics(manifest, frames)
        definition = _rehydrate_definition(manifest)
        request = dict(manifest.request)
        return ResearchRun(
            definition=definition,
            backtest=backtest,
            simulation=simulation,
            analytics=analytics,
            manifest=manifest,
            manifest_ref=self._workspace.manifest_ref(manifest),
            label=_optional_string(request.get("label")),
            tags=tuple(str(item) for item in request.get("tags", ())),
        )

    def rehydrate(
        self,
        reference: RunManifestRef | ResearchRun | str,
    ) -> ResearchRun:
        """Alias for :meth:`load_run` for explicit persisted-run workflows."""

        return self.load_run(reference)

    def latest(
        self,
        *,
        definition_fingerprint: str | None = None,
        include_invalidated: bool = False,
    ) -> RunManifest | None:
        """Return the newest matching manifest."""

        for reference in self.list(include_invalidated=include_invalidated):
            manifest = self._workspace.open_manifest(reference)
            if (
                definition_fingerprint is None
                or manifest.definition_fingerprint == definition_fingerprint
            ):
                return manifest
        return None

    def coverage(self, *, include_invalidated: bool = False) -> pd.DataFrame:
        """Summarize persisted run, date, status, tag, and artifact coverage."""

        rows: list[dict[str, Any]] = []
        for reference in self.list(include_invalidated=include_invalidated):
            manifest = self._workspace.open_manifest(reference)
            request = dict(manifest.request)
            schedule = request.get("review_schedule", {})
            rows.append(
                {
                    "execution_id": manifest.execution_id,
                    "definition_fingerprint": manifest.definition_fingerprint,
                    "index_id": manifest.index_id,
                    "label": request.get("label"),
                    "tags": tuple(request.get("tags", ())),
                    "status": manifest.status,
                    "research_status": self.get_status(manifest).value,
                    "invalidated": self._is_invalidated(manifest.execution_id),
                    "created_at": manifest.created_at,
                    "completed_at": manifest.completed_at,
                    "first_reference_date": schedule.get("first_reference_date"),
                    "last_reference_date": schedule.get("last_reference_date"),
                    "first_effective_date": schedule.get("first_effective_date"),
                    "last_effective_date": schedule.get("last_effective_date"),
                    "simulation_start_date": request.get("simulation", {}).get(
                        "start_date"
                    ),
                    "simulation_end_date": request.get("simulation", {}).get(
                        "end_date"
                    ),
                    "artifact_count": len(manifest.artifacts),
                    "artifact_bytes": sum(
                        artifact.size_bytes for artifact in manifest.artifacts
                    ),
                }
            )
        return pd.DataFrame.from_records(rows)

    def verify(self) -> WorkspaceVerification:
        """Verify manifests, objects, sidecars, bindings, and catalog parity."""

        errors: list[str] = []
        warnings: list[str] = []
        try:
            low_level = self._workspace.verify()
        except Exception as error:
            low_level = {}
            errors.append(f"workspace verification failed: {type(error).__name__}")
        manifests_checked = int(low_level.get("manifest_count", 0))
        artifacts_checked = int(low_level.get("artifact_count", 0))
        for category in (
            "manifest_failures",
            "artifact_failures",
            "sidecar_failures",
            "catalog_failures",
        ):
            records = low_level.get(category, ())
            if not isinstance(records, Sequence):
                errors.append(f"{category}: InvalidVerificationResult")
                continue
            for record in records:
                error_type = (
                    record.get("error_type", "IntegrityError")
                    if isinstance(record, Mapping)
                    else "InvalidVerificationResult"
                )
                errors.append(f"{category}: {error_type}")

        catalogued: set[str] = set()
        try:
            references = self._workspace.list_manifests()
        except Exception as error:
            errors.append(f"catalog cannot be read: {type(error).__name__}")
            references = ()
        for reference in references:
            catalogued.add(reference.execution_id)
        discovered, discovery_errors = self._discover_manifests()
        errors.extend(discovery_errors)
        missing = sorted(set(discovered).difference(catalogued))
        if missing:
            warnings.append(
                f"{len(missing)} verified manifest files are absent from the catalog"
            )
        return WorkspaceVerification(
            ok=not errors,
            manifests_checked=manifests_checked,
            artifacts_checked=artifacts_checked,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def rebuild_catalog(self) -> CatalogRebuildResult:
        """Rebuild manifests, objects, and reusable bindings from sidecars."""

        discovered, errors = self._discover_manifests()
        mutable_errors = list(errors)
        counts: Mapping[str, int] = {}
        try:
            counts = self._workspace.rebuild_catalog()
        except Exception as error:
            mutable_errors.append(f"catalog rebuild failed: {type(error).__name__}")
        return CatalogRebuildResult(
            discovered=len(discovered),
            registered=int(counts.get("manifests", 0)),
            errors=tuple(mutable_errors),
            artifacts=int(counts.get("artifacts", 0)),
            bindings=int(counts.get("bindings", 0)),
        )

    def invalidate(
        self,
        reference: RunManifestRef | ResearchRun | str,
    ) -> RunManifest:
        """Mark a run inactive without deleting its manifest or artifacts."""

        manifest = self.open_run(reference)
        path = self._invalidation_path(manifest.execution_id)
        _atomic_json_write(
            path,
            {
                "execution_id": manifest.execution_id,
                "invalidated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return manifest

    def get_status(
        self,
        reference: RunManifestRef | ResearchRun | RunManifest | str,
    ) -> ResearchStatus:
        """Return the current research-governance state for one execution."""

        manifest = (
            reference
            if isinstance(reference, RunManifest)
            else self.open_run(reference)
        )
        path = self._research_status_path(manifest.execution_id)
        if not path.is_file():
            return ResearchStatus.DRAFT
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return ResearchStatus(payload["status"])
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ) as exc:
            raise ResearchWorkflowError("research status metadata is invalid") from exc

    def set_status(
        self,
        reference: RunManifestRef | ResearchRun | RunManifest | str,
        status: ResearchStatus | str,
        *,
        supersedes: RunManifestRef | ResearchRun | RunManifest | str | None = None,
    ) -> ResearchStatus:
        """Update review state without mutating calculation artifacts."""

        manifest = (
            reference
            if isinstance(reference, RunManifest)
            else self.open_run(reference)
        )
        selected = ResearchStatus(status)
        superseded_execution_id: str | None = None
        if supersedes is not None:
            superseded_manifest = (
                supersedes
                if isinstance(supersedes, RunManifest)
                else self.open_run(supersedes)
            )
            if superseded_manifest.execution_id == manifest.execution_id:
                raise ValueError("a research run cannot supersede itself")
            superseded_execution_id = superseded_manifest.execution_id
            self._write_research_status(
                superseded_execution_id,
                ResearchStatus.SUPERSEDED,
                replacement_execution_id=manifest.execution_id,
            )
        self._write_research_status(
            manifest.execution_id,
            selected,
            supersedes_execution_id=superseded_execution_id,
        )
        return selected

    def prune(
        self,
        *,
        dry_run: bool = True,
        include_invalidated_artifacts: bool = False,
    ) -> PruneResult:
        """Plan or explicitly remove unreferenced content-addressed artifacts.

        The default is a non-mutating dry run.  Manifests, reports, review
        caches, and catalog files are never deleted by this method.
        """

        if include_invalidated_artifacts:
            raise ResearchWorkflowError(
                "artifacts referenced by immutable run manifests cannot be "
                "pruned; invalidate reusable bindings or retain the run"
            )
        verification = self._workspace.verify()
        if verification.get("status") != "ok":
            raise ResearchWorkflowError(
                "workspace verification must pass before pruning"
            )
        result = self._workspace.prune(dry_run=dry_run)
        candidates = tuple(
            self.workspace_path.joinpath(item["relative_path"]).resolve()
            for item in result["candidates"]
        )
        removed = (
            tuple(path for path in candidates if not path.exists())
            if not dry_run
            else ()
        )
        return PruneResult(
            dry_run=dry_run,
            candidates=candidates,
            removed=removed,
            candidate_bytes=int(result["size_bytes"]),
        )

    def _discover_manifests(
        self,
    ) -> tuple[dict[str, RunManifest], tuple[str, ...]]:
        manifests: dict[str, RunManifest] = {}
        errors: list[str] = []
        for path in self.workspace_path.glob("runs/*/executions/*/run_manifest.json"):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                payload = envelope["payload"]
                reference = RunManifestRef(
                    workspace_name=self.workspace_name,
                    definition_fingerprint=str(payload["definition_fingerprint"]),
                    execution_id=str(payload["execution_id"]),
                    path=str(path),
                )
                manifest = self._workspace.open_manifest(reference)
                manifests[manifest.execution_id] = manifest
            except Exception as error:
                relative = path.relative_to(self.workspace_path)
                errors.append(f"{relative}: {type(error).__name__}")
        return manifests, tuple(errors)

    def _invalidation_path(self, execution_id: str) -> Path:
        if len(execution_id) != 32 or any(
            character not in "0123456789abcdef" for character in execution_id
        ):
            raise ValueError("execution_id is invalid")
        return self.workspace_path.joinpath(
            "state",
            "invalidations",
            f"{execution_id}.json",
        )

    def _is_invalidated(self, execution_id: str) -> bool:
        return self._invalidation_path(execution_id).is_file()

    def _research_status_path(self, execution_id: str) -> Path:
        if len(execution_id) != 32 or any(
            character not in "0123456789abcdef" for character in execution_id
        ):
            raise ValueError("execution_id is invalid")
        return self.workspace_path.joinpath(
            "state",
            "research_status",
            f"{execution_id}.json",
        )

    def _write_research_status(
        self,
        execution_id: str,
        status: ResearchStatus,
        *,
        supersedes_execution_id: str | None = None,
        replacement_execution_id: str | None = None,
    ) -> None:
        path = self._research_status_path(execution_id)
        history: list[Mapping[str, Any]] = []
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                history = list(existing.get("history", ()))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
                history = []
        changed_at = datetime.now(timezone.utc).isoformat()
        history.append(
            {
                "status": status.value,
                "changed_at": changed_at,
            }
        )
        _atomic_json_write(
            path,
            {
                "execution_id": execution_id,
                "status": status.value,
                "changed_at": changed_at,
                "supersedes_execution_id": supersedes_execution_id,
                "replacement_execution_id": replacement_execution_id,
                "history": history,
            },
        )

    def _bound_artifact_paths(self) -> set[Path]:
        """Return artifacts protected by active v2 cache bindings."""

        if not self._workspace.catalog_path.is_file():
            return set()
        try:
            with sqlite3.connect(self._workspace.catalog_path) as connection:
                rows = connection.execute(
                    """
                    SELECT DISTINCT a.relative_path
                    FROM artifact_bindings AS b
                    JOIN artifacts AS a
                      ON a.content_digest = b.content_digest
                     AND a.file_checksum = b.file_checksum
                    """
                ).fetchall()
        except sqlite3.DatabaseError as error:
            raise ResearchWorkflowError(
                "workspace catalog cannot be inspected safely for pruning"
            ) from error
        return {self.workspace_path.joinpath(str(row[0])).resolve() for row in rows}


__all__ = ["_ResearchRunCatalog"]
