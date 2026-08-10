"""Domain contracts for optional review-result persistence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class CachePolicy(StrEnum):
    """Control whether a review store may read or replace cached results."""

    REUSE = "reuse"
    REFRESH = "refresh"
    READ_ONLY = "read_only"


class CacheSource(StrEnum):
    """Describe how one review result was obtained."""

    COMPUTED = "computed"
    MEMORY = "memory"
    DISK = "disk"


class ReviewCacheMissError(RuntimeError):
    """Raised when a required review artifact is unavailable."""


@dataclass(frozen=True)
class ArtifactMetadata:
    """Storage-neutral provenance for one immutable artifact."""

    source: CacheSource
    checksum: str
    path: str


@dataclass(frozen=True)
class ReviewArtifact:
    """Serializable result of one portfolio-construction review."""

    reference_date: pd.Timestamp
    effective_date: pd.Timestamp
    index_id: str
    universe_id: str
    constituents: pd.DataFrame
    daily: pd.DataFrame | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    provenance_records: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class LoadedReview:
    """A review artifact together with cache provenance."""

    artifact: ReviewArtifact
    metadata: ArtifactMetadata


class ReviewCacheStore(Protocol):
    """Persistence operations required by the review runner."""

    workspace_name: str
    workspace_path: Path
    reports_path: Path

    def report_path(
        self,
        file_name: str,
        *,
        create_parent: bool = True,
    ) -> Path: ...

    def load_review(
        self,
        reference_date,
        effective_date,
    ) -> LoadedReview: ...

    def save_review(self, artifact: ReviewArtifact) -> ArtifactMetadata: ...

    def record_request(
        self,
        reviews,
        *,
        cache_policy: CachePolicy | str,
    ) -> None: ...


ReviewStoreFactory = Callable[..., ReviewCacheStore]
_review_store_factory: ReviewStoreFactory | None = None


def register_review_store_factory(factory: ReviewStoreFactory) -> None:
    """Register the infrastructure adapter used by named legacy workspaces."""

    if not callable(factory):
        raise TypeError("review store factory must be callable")
    global _review_store_factory
    _review_store_factory = factory


def create_review_store(**parameters) -> ReviewCacheStore:
    """Create the registered review store without importing infrastructure."""

    if _review_store_factory is None:
        raise RuntimeError(
            "named review caching requires an injected cache_store or an "
            "installed workspace review-store adapter"
        )
    return _review_store_factory(**parameters)


__all__ = [
    "ArtifactMetadata",
    "CachePolicy",
    "CacheSource",
    "LoadedReview",
    "ReviewArtifact",
    "ReviewCacheMissError",
    "ReviewCacheStore",
    "create_review_store",
    "register_review_store_factory",
]
