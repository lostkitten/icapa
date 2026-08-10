"""Weight concentration statistics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def calculate_concentration(
    weights: pd.Series,
    *,
    top_counts: Iterable[int] = (1, 5, 10),
    weight_tolerance: float = 1e-8,
) -> pd.Series:
    """Calculate HHI, effective count, and top-weight concentration."""

    if weight_tolerance <= 0:
        raise ValueError("weight_tolerance must be positive")
    values = _weights(weights, weight_tolerance)
    hhi = float(np.square(values).sum())
    result: dict[str, float] = {
        "constituent_count": float((values > weight_tolerance).sum()),
        "maximum_weight": float(values.max()),
        "hhi": hhi,
        "effective_constituent_count": 1.0 / hhi if hhi > 0 else np.nan,
    }
    selected = tuple(sorted({int(count) for count in top_counts}))
    if not selected or any(count <= 0 for count in selected):
        raise ValueError("top_counts must contain positive integers")
    ordered = values.sort_values(ascending=False)
    for count in selected:
        result[f"top_{count}_weight"] = float(ordered.head(count).sum())
    return pd.Series(result, dtype=float, name="value")


def _weights(value: pd.Series, tolerance: float) -> pd.Series:
    if not isinstance(value, pd.Series) or value.empty:
        raise ValueError("weights must be a non-empty pandas Series")
    result = pd.to_numeric(value.copy(deep=True), errors="coerce")
    if result.isna().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("weights must contain finite numeric values")
    if (result < -tolerance).any():
        raise ValueError("weights must be non-negative")
    result = result.clip(lower=0.0)
    total = float(result.sum())
    if not np.isclose(total, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError(f"weights must sum to one; observed {total:.12g}")
    return result.astype(float)
