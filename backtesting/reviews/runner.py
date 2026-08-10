"""Run methodologies across effective-date review schedules."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from icapa.backtesting.calendar import Calendar
from icapa.portfolio_construction.context import DataContext
from .cache_contracts import (
    CachePolicy,
    CacheSource,
    ReviewArtifact,
    ReviewCacheMissError,
    ReviewCacheStore,
    create_review_store,
)
from .identity import build_run_fingerprint, canonical_digest
from .models import BacktestMetadata, BacktestResult, ReviewResultMetadata


@dataclass
class Backtester:
    """Run one methodology over an explicit review calendar.

    Without ``workspace_name`` this preserves the original in-memory,
    side-effect-free behavior. A named workspace adds deterministic review
    reuse without changing methodology execution.
    """

    index_id: str
    calendar: Calendar
    methodology: object
    provider_name: str | None = None
    provider_parameters: dict = field(default_factory=dict)
    workspace_name: str | None = None
    cache_policy: CachePolicy | str = CachePolicy.REUSE
    data_revision: Any = "unversioned"
    cache_configuration: dict = field(default_factory=dict)
    cache_store: ReviewCacheStore | None = field(default=None, repr=False)
    _workspace_store: ReviewCacheStore | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _fingerprint: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.index_id:
            raise ValueError("index_id must not be empty")
        if not callable(getattr(self.methodology, "execute", None)):
            raise TypeError("methodology must implement execute(data_context)")
        self.cache_policy = CachePolicy(self.cache_policy)
        if self.cache_store is not None:
            required = ("load_review", "save_review", "record_request")
            missing = [
                name
                for name in required
                if not callable(getattr(self.cache_store, name, None))
            ]
            if missing:
                raise TypeError(
                    "cache_store is missing required methods: "
                    + ", ".join(missing)
                )
            if self.workspace_name is None:
                self.workspace_name = getattr(
                    self.cache_store,
                    "workspace_name",
                    None,
                )
            configuration = self._fingerprint_configuration()
            self._fingerprint = build_run_fingerprint(
                index_id=self.index_id,
                methodology=self.methodology,
                configuration=configuration,
                data_revision=self.data_revision,
            )
            self._workspace_store = self.cache_store
        elif self.workspace_name is not None:
            configuration = self._fingerprint_configuration()
            self._fingerprint = build_run_fingerprint(
                index_id=self.index_id,
                methodology=self.methodology,
                configuration=configuration,
                data_revision=self.data_revision,
            )
            methodology_type = type(self.methodology)
            methodology_name = (
                f"{methodology_type.__module__}.{methodology_type.__qualname__}"
            )
            self._workspace_store = create_review_store(
                workspace_name=self.workspace_name,
                fingerprint=self._fingerprint,
                index_id=self.index_id,
                methodology_name=methodology_name,
                configuration_digest=canonical_digest(configuration),
                data_revision=self.data_revision,
            )
        elif self.cache_policy is CachePolicy.READ_ONLY:
            raise ValueError(
                "READ_ONLY cache policy requires workspace_name or cache_store"
            )

    @property
    def workspace_store(self) -> ReviewCacheStore | None:
        """Return the optional artifact store used by this run."""

        return self._workspace_store

    @property
    def workspace_path(self) -> Path | None:
        """Return the fingerprinted workspace path, when caching is enabled."""

        if self._workspace_store is None:
            return None
        return self._workspace_store.workspace_path

    @property
    def reports_path(self) -> Path | None:
        """Return the report directory for a named workspace."""

        if self._workspace_store is None:
            return None
        return self._workspace_store.reports_path

    def report_path(self, file_name: str, *, create_parent: bool = True) -> Path:
        """Return a safe report path for a named workspace."""

        if self._workspace_store is None:
            raise ValueError("report paths require workspace_name")
        return self._workspace_store.report_path(
            file_name,
            create_parent=create_parent,
        )

    def run_review(self, reference_date, effective_date) -> DataContext:
        """Load or execute a single review and return its populated context."""

        context, _ = self._run_review_with_metadata(reference_date, effective_date)
        return context

    def _compute_review(self, reference_date, effective_date) -> DataContext:
        """Execute one methodology review without consulting a workspace."""

        context = DataContext(
            reference_date=reference_date,
            effective_date=effective_date,
            index_id=self.index_id,
            provider_name=self.provider_name,
            provider_parameters=dict(self.provider_parameters),
            calendar=self.calendar,
        )
        result = self.methodology.execute(context)
        if result is not None:
            context = result
        self._validate_review(context)
        return context

    @staticmethod
    def _validate_review(context: DataContext) -> None:
        """Validate the final public weight contract for computed and cached reviews."""

        if "index_weight" not in context.cons.columns:
            raise ValueError("methodology did not produce index_weight")
        weights = context.cons["index_weight"]
        if weights.isna().any() or (weights < 0).any():
            raise ValueError("methodology produced invalid index weights")
        if abs(float(weights.sum()) - 1.0) > 1e-8:
            raise ValueError("methodology index weights must sum to one")

    def _run_review_with_metadata(
        self,
        reference_date,
        effective_date,
    ) -> tuple[DataContext, ReviewResultMetadata]:
        reference = pd.Timestamp(reference_date).normalize()
        effective = pd.Timestamp(effective_date).normalize()
        store = self._workspace_store

        if store is not None and self.cache_policy is not CachePolicy.REFRESH:
            try:
                loaded = store.load_review(reference, effective)
            except ReviewCacheMissError:
                if self.cache_policy is CachePolicy.READ_ONLY:
                    raise
            else:
                context = self._context_from_artifact(loaded.artifact)
                self._validate_review(context)
                return context, ReviewResultMetadata(
                    reference_date=reference,
                    effective_date=effective,
                    cache_source=loaded.metadata.source,
                    artifact_checksum=loaded.metadata.checksum,
                    artifact_path=loaded.metadata.path,
                )

        context = self._compute_review(reference, effective)
        if store is None:
            return context, ReviewResultMetadata(
                reference_date=reference,
                effective_date=effective,
                cache_source=CacheSource.COMPUTED,
            )

        saved = store.save_review(
            ReviewArtifact(
                reference_date=reference,
                effective_date=effective,
                index_id=self.index_id,
                universe_id=context.universe_id,
                constituents=context.cons,
                daily=context.daily,
                diagnostics=context.diagnostics,
                provenance_records=tuple(context.provenance.records),
            )
        )
        return context, ReviewResultMetadata(
            reference_date=reference,
            effective_date=effective,
            cache_source=CacheSource.COMPUTED,
            artifact_checksum=saved.checksum,
            artifact_path=saved.path,
        )

    def _context_from_artifact(self, artifact: ReviewArtifact) -> DataContext:
        context = DataContext(
            reference_date=artifact.reference_date,
            effective_date=artifact.effective_date,
            index_id=self.index_id,
            universe_id=artifact.universe_id,
            provider_name=self.provider_name,
            provider_parameters=dict(self.provider_parameters),
            calendar=self.calendar,
            diagnostics=deepcopy(dict(artifact.diagnostics or {})),
        )
        context.set_dataframe(artifact.constituents)
        context.daily = None if artifact.daily is None else artifact.daily.copy(deep=True)
        for record in artifact.provenance_records:
            context.provenance.record_provider_call(record)
        return context

    def run(self) -> BacktestResult:
        """Execute every review in effective-date order."""

        reviews: dict[pd.Timestamp, DataContext] = {}
        review_metadata: dict[pd.Timestamp, ReviewResultMetadata] = {}
        rows: list[pd.DataFrame] = []
        for review in self.calendar.dates.itertuples(index=False):
            effective_date = pd.Timestamp(review.effective_date).normalize()
            context, metadata = self._run_review_with_metadata(
                review.reference_date,
                effective_date,
            )
            reviews[effective_date] = context
            review_metadata[effective_date] = metadata
            review_weights = context.cons[["index_weight"]].copy()
            review_weights.insert(0, "effective_date", effective_date)
            rows.append(review_weights.reset_index())

        if not rows:
            weights = pd.DataFrame(
                columns=["effective_date", "instrument_id", "index_weight"]
            )
        else:
            weights = pd.concat(rows, ignore_index=True)
        weights = weights.set_index(["effective_date", "instrument_id"])
        if (
            self._workspace_store is not None
            and self.cache_policy is not CachePolicy.READ_ONLY
        ):
            self._workspace_store.record_request(
                [
                    {
                        "reference_date": metadata.reference_date,
                        "effective_date": metadata.effective_date,
                        "cache_source": metadata.cache_source,
                    }
                    for metadata in review_metadata.values()
                ],
                cache_policy=self.cache_policy,
            )
        result_metadata = BacktestMetadata(
            workspace_name=self.workspace_name,
            fingerprint=self._fingerprint,
            cache_policy=self.cache_policy if self.workspace_name is not None else None,
            workspace_path=(
                None
                if self._workspace_store is None
                else str(self._workspace_store.workspace_path)
            ),
            reviews=review_metadata,
        )
        return BacktestResult(
            weights=weights,
            reviews=reviews,
            metadata=result_metadata,
        )

    def _fingerprint_configuration(self) -> dict[str, Any]:
        """Return calculation configuration while intentionally excluding date range."""

        calendar_configuration = {
            "type": (
                f"{type(self.calendar).__module__}."
                f"{type(self.calendar).__qualname__}"
            ),
            "calendar_id": getattr(self.calendar, "calendar_id", ""),
            "provider_name": getattr(self.calendar, "provider_name", None),
            "provider_parameters": dict(
                getattr(self.calendar, "provider_parameters", {}) or {}
            ),
            "command": getattr(self.calendar, "command", None),
        }
        return {
            "provider_name": self.provider_name,
            "provider_parameters": dict(self.provider_parameters),
            "calendar": calendar_configuration,
            "run": dict(self.cache_configuration),
        }


__all__ = [
    "BacktestMetadata",
    "BacktestResult",
    "Backtester",
    "ReviewResultMetadata",
]
