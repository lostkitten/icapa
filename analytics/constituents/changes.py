"""Constituent membership and weight-change explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def explain_weight_change(
    previous_weights: pd.Series,
    current_weights: pd.Series,
    *,
    weight_tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Explain entrants, exits, and continuing constituent weight changes."""

    if weight_tolerance <= 0:
        raise ValueError("weight_tolerance must be positive")
    previous = _weights(previous_weights, "previous_weights", weight_tolerance)
    current = _weights(current_weights, "current_weights", weight_tolerance)
    previous, current = previous.align(current, join="outer", fill_value=0.0)
    change = current - previous
    status = np.select(
        [
            (previous <= weight_tolerance) & (current > weight_tolerance),
            (previous > weight_tolerance) & (current <= weight_tolerance),
            change > weight_tolerance,
            change < -weight_tolerance,
        ],
        ["entrant", "exit", "weight_increase", "weight_decrease"],
        default="unchanged",
    )
    result = pd.DataFrame(
        {
            "previous_weight": previous,
            "current_weight": current,
            "weight_change": change,
            "absolute_weight_change": change.abs(),
            "one_way_turnover_contribution": 0.5 * change.abs(),
            "status": status,
        }
    )
    result.index.name = previous.index.name or "instrument_id"
    return result.sort_values(
        ["absolute_weight_change"],
        ascending=False,
        kind="stable",
    )


def _weights(value: pd.Series, label: str, tolerance: float) -> pd.Series:
    if not isinstance(value, pd.Series) or value.empty:
        raise ValueError(f"{label} must be a non-empty pandas Series")
    if value.index.has_duplicates:
        raise ValueError(f"{label} index must be unique")
    result = pd.to_numeric(value.copy(deep=True), errors="coerce")
    if result.isna().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must contain finite numeric values")
    if (result < -tolerance).any():
        raise ValueError(f"{label} must be non-negative")
    total = float(result.sum())
    if not np.isclose(total, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError(f"{label} must sum to one; observed {total:.12g}")
    return result.clip(lower=0.0).astype(float)
