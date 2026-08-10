"""Stable result contracts for effective-date methodology reviews."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from icapa.portfolio_construction.context import DataContext

from .cache_contracts import CachePolicy, CacheSource


@dataclass(frozen=True)
class ReviewResultMetadata:
    """Cache provenance for one effective-date result."""

    reference_date: pd.Timestamp
    effective_date: pd.Timestamp
    cache_source: CacheSource
    artifact_checksum: str | None = None
    artifact_path: str | None = None


@dataclass(frozen=True)
class BacktestMetadata:
    """Run identity, workspace location, and per-review cache provenance."""

    workspace_name: str | None
    fingerprint: str | None
    cache_policy: CachePolicy | None
    workspace_path: str | None
    reviews: dict[pd.Timestamp, ReviewResultMetadata]


@dataclass(frozen=True)
class BacktestResult:
    """Collected target weights, review contexts, and cache metadata."""

    weights: pd.DataFrame
    reviews: dict[pd.Timestamp, DataContext]
    metadata: BacktestMetadata | None = None


__all__ = [
    "BacktestMetadata",
    "BacktestResult",
    "ReviewResultMetadata",
]
