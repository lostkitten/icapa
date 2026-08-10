"""Business-level provider interfaces used by data-loading rules.

Connection technology and physical schemas deliberately do not appear here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class UniverseProvider(Protocol):
    def load_universe(
        self,
        universe_id: str,
        reference_date,
        effective_date,
        **kwargs: Any,
    ) -> pd.DataFrame: ...


@runtime_checkable
class CalendarProvider(Protocol):
    def load_business_days(self, calendar_id: str, start_date, end_date) -> pd.DataFrame: ...

    def load_review_schedule(
        self,
        calendar_id: str,
        start_date,
        end_date,
        **kwargs: Any,
    ) -> pd.DataFrame: ...


@runtime_checkable
class MarketDataProvider(Protocol):
    def load_daily_market_data(
        self,
        instrument_ids: Iterable,
        start_date,
        end_date,
        **kwargs: Any,
    ) -> pd.DataFrame: ...


@runtime_checkable
class MembershipProvider(Protocol):
    def load_membership(
        self,
        index_id: str,
        start_date,
        end_date,
        **kwargs: Any,
    ) -> pd.DataFrame: ...


@runtime_checkable
class ReferenceDataProvider(Protocol):
    def load_reference_data(
        self,
        instrument_ids: Iterable,
        reference_date,
        fields: Iterable[str],
        **kwargs: Any,
    ) -> pd.DataFrame: ...


@runtime_checkable
class ThirdPartyDataProvider(Protocol):
    """Provider for explicitly classified, non-canonical external data."""

    def load_third_party_data(
        self,
        data_type: str,
        instrument_ids: Iterable,
        fields: Iterable[str],
        reference_date,
        parameters: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame: ...


@runtime_checkable
class SnapshotAwareProvider(Protocol):
    """Optional lightweight identity probe for provider-backed data.

    The returned mapping must contain only non-sensitive metadata. Providers
    that cannot expose a stable snapshot may omit this protocol; callers then
    hash the canonical response after loading it.
    """

    def describe_snapshot(
        self,
        capability: str,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


__all__ = [
    "CalendarProvider",
    "MarketDataProvider",
    "MembershipProvider",
    "ReferenceDataProvider",
    "SnapshotAwareProvider",
    "ThirdPartyDataProvider",
    "UniverseProvider",
]
