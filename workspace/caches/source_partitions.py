"""Canonical partition and descriptor codecs for source-data caches."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from ...data_sources.contracts import validate_daily_market_data
from .source_contracts import SCHEMA_VERSION
from .source_identity import UnsafeCacheReuseError


def _decode_snapshot_frame(frame: pd.DataFrame) -> tuple[str, str]:
    required = {"snapshot_digest", "snapshot_protocol"}
    if len(frame) != 1 or not required.issubset(frame.columns):
        raise UnsafeCacheReuseError(
            "cached provider snapshot descriptor is invalid"
        )
    digest = str(frame.iloc[0]["snapshot_digest"])
    protocol = str(frame.iloc[0]["snapshot_protocol"])
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not protocol
    ):
        raise UnsafeCacheReuseError(
            "cached provider snapshot descriptor is invalid"
        )
    return digest, protocol


def _decode_month_coverage_descriptors(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    required = {
        "schema_version",
        "start_date",
        "end_date",
        "cache_key",
        "content_digest",
        "file_checksum",
    }
    if frame.empty or not required.issubset(frame.columns):
        raise UnsafeCacheReuseError(
            "cached source-data month coverage descriptor is invalid"
        )
    descriptors: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        try:
            schema_version = int(row.schema_version)
            start_date = pd.Timestamp(row.start_date).normalize()
            end_date = pd.Timestamp(row.end_date).normalize()
            digests = {
                name: str(getattr(row, name))
                for name in (
                    "cache_key",
                    "content_digest",
                    "file_checksum",
                )
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise UnsafeCacheReuseError(
                "cached source-data month coverage descriptor is invalid"
            ) from exc
        if (
            schema_version != SCHEMA_VERSION
            or start_date > end_date
            or start_date.to_period("M") != end_date.to_period("M")
            or any(
                len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
                for value in digests.values()
            )
        ):
            raise UnsafeCacheReuseError(
                "cached source-data month coverage descriptor is invalid"
            )
        descriptors.append(
            {
                "start_date": start_date,
                "end_date": end_date,
                **digests,
            }
        )
    return descriptors


def _select_covering_descriptors(
    descriptors: Iterable[Mapping[str, Any]],
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Select a deterministic minimal chain that covers a calendar range."""

    candidates = [
        dict(item)
        for item in descriptors
        if item["end_date"] >= start_date
        and item["start_date"] <= end_date
    ]
    selected: list[dict[str, Any]] = []
    cursor = start_date
    while cursor <= end_date:
        covering = [
            item
            for item in candidates
            if item["start_date"] <= cursor <= item["end_date"]
        ]
        if not covering:
            return []
        best = max(
            covering,
            key=lambda item: (
                item["end_date"],
                -item["start_date"].value,
                item["cache_key"],
            ),
        )
        selected.append(best)
        cursor = best["end_date"] + pd.Timedelta(days=1)
    return selected

def _canonical_partition(
    raw: pd.DataFrame,
    *,
    instruments: tuple[Any, ...],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("market-data provider must return a pandas DataFrame")
    frame = validate_daily_market_data(raw)
    frame = frame.loc[
        frame["instrument_id"].isin(instruments)
        & frame["business_date"].between(start_date, end_date)
    ].copy()
    frame["__instrument_order__"] = frame["instrument_id"].map(str)
    frame = (
        frame.sort_values(
            ["business_date", "__instrument_order__"],
            kind="mergesort",
        )
        .drop(columns="__instrument_order__")
        .reset_index(drop=True)
    )
    return frame


def _canonical_business_days(
    raw: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DatetimeIndex:
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(
            "business-day provider must return a pandas DataFrame"
        )
    if "business_date" not in raw:
        raise ValueError(
            "business-day provider response is missing business_date"
        )
    converted = pd.to_datetime(raw["business_date"], errors="raise")
    if converted.isna().any():
        raise ValueError(
            "business-day provider response contains null dates"
        )
    days = pd.DatetimeIndex(converted).normalize()
    if days.has_duplicates:
        raise ValueError(
            "business-day provider response contains duplicate dates"
        )
    days = days[(days >= start_date) & (days <= end_date)].sort_values()
    return days


__all__ = [
    "canonical_business_days",
    "canonical_partition",
    "decode_month_coverage_descriptors",
    "decode_snapshot_frame",
    "select_covering_descriptors",
]


canonical_business_days = _canonical_business_days
canonical_partition = _canonical_partition
decode_month_coverage_descriptors = _decode_month_coverage_descriptors
decode_snapshot_frame = _decode_snapshot_frame
select_covering_descriptors = _select_covering_descriptors
