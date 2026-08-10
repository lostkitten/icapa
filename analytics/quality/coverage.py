"""Provider-neutral data-quality calculations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def calculate_data_coverage(
    frame: pd.DataFrame,
    required_fields: Sequence[str],
    *,
    weight_column: str | None = None,
) -> pd.DataFrame:
    """Calculate row and optional weight coverage for each required field."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a non-empty pandas DataFrame")
    fields = tuple(str(field) for field in required_fields)
    if not fields:
        raise ValueError("required_fields must not be empty")
    missing = sorted(set(fields).difference(frame.columns))
    if missing:
        raise ValueError(f"coverage input is missing columns: {missing}")
    weights: pd.Series | None = None
    if weight_column is not None:
        if weight_column not in frame:
            raise ValueError(f"coverage input is missing {weight_column!r}")
        weights = pd.to_numeric(frame[weight_column], errors="coerce")
        if weights.isna().any() or not np.isfinite(
            weights.to_numpy(dtype=float)
        ).all():
            raise ValueError("coverage weights must be finite numeric values")
        if (weights < 0).any():
            raise ValueError("coverage weights must be non-negative")

    rows = []
    for field in fields:
        available = frame[field].notna()
        if frame[field].dtype == "object":
            available &= frame[field].astype(str).str.strip().ne("")
        row: dict[str, object] = {
            "field": field,
            "row_count": len(frame),
            "available_count": int(available.sum()),
            "missing_count": int((~available).sum()),
            "row_coverage": float(available.mean()),
        }
        if weights is not None:
            total = float(weights.sum())
            row["weight_coverage"] = (
                float(weights[available].sum()) / total if total > 0 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_data_freshness(
    observation_dates: pd.Series,
    reference_date: object,
) -> pd.Series:
    """Summarize age and future-dated observations at one review cutoff."""

    if not isinstance(observation_dates, pd.Series) or observation_dates.empty:
        raise ValueError("observation_dates must be a non-empty pandas Series")
    observations = pd.to_datetime(observation_dates, errors="coerce")
    if observations.isna().any():
        raise ValueError("observation_dates contain invalid or missing values")
    cutoff = pd.Timestamp(reference_date).normalize()
    observations = observations.dt.normalize()
    ages = (cutoff - observations).dt.days
    return pd.Series(
        {
            "reference_date": cutoff,
            "observation_count": len(observations),
            "future_observation_count": int((ages < 0).sum()),
            "minimum_age_days": int(ages.min()),
            "median_age_days": float(ages.median()),
            "maximum_age_days": int(ages.max()),
        },
        name="value",
        dtype=object,
    )
