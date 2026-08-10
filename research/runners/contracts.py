"""High-level orchestration for reproducible index-research workflows.

The v1 backtesting, simulation, analytics, and reporting APIs remain
available independently.  This module composes them behind a small research
API while keeping cache reuse opt-in and evidence based.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ...portfolio_construction import ProviderRequestSpec
from ...workspace import (
    CacheMode,
    CacheStage,
    ParquetStageStore,
    WorkspaceRepository,
    automatic_digest,
)
from ..models import ResearchWorkflowError, UnsafeCacheReuseError


@dataclass(frozen=True, slots=True)
class _ReviewSnapshotRecord:
    reference_date: pd.Timestamp
    effective_date: pd.Timestamp
    snapshot_digest: str


@dataclass(frozen=True, slots=True)
class _ReviewSnapshotEvidence:
    records: tuple[_ReviewSnapshotRecord, ...]
    scope_digest: str

    @property
    def snapshot_digest(self) -> str:
        unique = {record.snapshot_digest for record in self.records}
        if len(unique) == 1:
            return next(iter(unique))
        return automatic_digest(
            {
                "kind": "review_snapshots_by_review",
                "reviews": [
                    {
                        "reference_date": record.reference_date,
                        "effective_date": record.effective_date,
                        "snapshot_digest": record.snapshot_digest,
                    }
                    for record in self.records
                ],
            }
        )

    def digest_for(
        self,
        reference_date: object,
        effective_date: object,
    ) -> str:
        reference = pd.Timestamp(reference_date).normalize()
        effective = pd.Timestamp(effective_date).normalize()
        for record in self.records:
            if (
                record.reference_date == reference
                and record.effective_date == effective
            ):
                return record.snapshot_digest
        raise UnsafeCacheReuseError(
            "review snapshot evidence is missing for "
            f"{reference.date()} / {effective.date()}"
        )


@dataclass(frozen=True, slots=True)
class _ExactReviewProviderBinding:
    provider_name: str
    capability: str
    parameters: Mapping[str, Any]
    prefix: str
    provider: object
    request_spec: ProviderRequestSpec | None = None


class _PerReviewParquetStageStore:
    """Route review artifacts by their exact point-in-time snapshot identity."""

    def __init__(
        self,
        workspace: WorkspaceRepository,
        *,
        index_id: str,
        construction_identity: str,
        evidence: _ReviewSnapshotEvidence,
    ) -> None:
        self.workspace = workspace
        self.index_id = index_id
        self.construction_identity = construction_identity
        self.evidence = evidence
        self._stores: dict[str, ParquetStageStore] = {}
        self._base_store = ParquetStageStore(
            workspace,
            index_id=index_id,
            namespace_digest=automatic_digest(
                {
                    "kind": "review_cache_scope",
                    "construction_identity": construction_identity,
                    "snapshot_scope": evidence.scope_digest,
                }
            ),
        )

    @property
    def workspace_name(self) -> str:
        return self.workspace.workspace_name

    @property
    def workspace_path(self) -> Path:
        return self.workspace.workspace_path

    @property
    def reports_path(self) -> Path:
        return self._base_store.reports_path

    def report_path(self, *args, **kwargs):
        return self._base_store.report_path(*args, **kwargs)

    def load_review(self, reference_date: object, effective_date: object):
        return self._store(reference_date, effective_date).load_review(
            reference_date,
            effective_date,
        )

    def save_review(self, artifact):
        return self._store(
            artifact.reference_date,
            artifact.effective_date,
        ).save_review(artifact)

    def record_request(self, reviews, *, cache_policy: object) -> None:
        self._base_store.record_request(reviews, cache_policy=cache_policy)

    def save_frame(self, *args, **kwargs):
        return self._base_store.save_frame(*args, **kwargs)

    def load_frame(self, *args, **kwargs):
        return self._base_store.load_frame(*args, **kwargs)

    def save_json(self, *args, **kwargs):
        return self._base_store.save_json(*args, **kwargs)

    def load_json(self, *args, **kwargs):
        return self._base_store.load_json(*args, **kwargs)

    def _store(
        self,
        reference_date: object,
        effective_date: object,
    ) -> ParquetStageStore:
        snapshot_digest = self.evidence.digest_for(
            reference_date,
            effective_date,
        )
        store = self._stores.get(snapshot_digest)
        if store is None:
            store = ParquetStageStore(
                self.workspace,
                index_id=self.index_id,
                namespace_digest=automatic_digest(
                    {
                        "kind": "review_cache",
                        "construction_identity": self.construction_identity,
                        "snapshot_digest": snapshot_digest,
                        "snapshot_scope": self.evidence.scope_digest,
                    }
                ),
            )
            self._stores[snapshot_digest] = store
        return store


@dataclass(frozen=True, slots=True)
class _CacheDecision:
    stage: CacheStage
    requested: CacheMode
    actual: CacheMode
    reason: str
    snapshot_digest: str | None = None
    review_snapshot: _ReviewSnapshotEvidence | None = None
    fatal: bool = False

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "stage": self.stage.value,
            "requested_mode": self.requested.value,
            "mode": self.actual.value,
            "reason": self.reason,
        }
        if self.snapshot_digest is not None:
            value["snapshot_digest"] = self.snapshot_digest
        return value


@dataclass(frozen=True, slots=True)
class _ProviderEvidence:
    records: tuple[Mapping[str, Any], ...]
    review_records: tuple[Mapping[str, Any], ...]
    calendar_records: tuple[Mapping[str, Any], ...]
    simulation_records: tuple[Mapping[str, Any], ...]
    review_verified: bool
    calendar_verified: bool
    simulation_verified: bool


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    component: Mapping[str, Any]
    workflow_components: Mapping[str, Mapping[str, Any]]
    provider_evidence: _ProviderEvidence
    construction_identity: str
    decisions: tuple[_CacheDecision, ...]
    request_payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _PersistedMethodology:
    """Report-facing identity for a calculation that is not re-executed."""

    report_name: str
    parameters: Mapping[str, Any]

    def execute(self, context):
        del context
        raise ResearchWorkflowError(
            "a rehydrated run cannot execute its original methodology"
        )


@dataclass(frozen=True, slots=True)
class _RunReportWorkspace:
    """Constrain one high-level bundle to its definition-scoped run tree."""

    workspace_name: str
    reports_path: Path


__all__ = [name for name in globals() if name.startswith("_")]
