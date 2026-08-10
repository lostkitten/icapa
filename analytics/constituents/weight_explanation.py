"""Stepwise explanation of multiplicative methodology tilts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class WeightExplanationResult:
    """Instrument-level step changes and step-level diagnostics."""

    detail: pd.DataFrame
    summary: pd.DataFrame
    final_weights: pd.Series
    reconciled: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", self.detail.copy(deep=True))
        object.__setattr__(self, "summary", self.summary.copy(deep=True))
        object.__setattr__(
            self,
            "final_weights",
            self.final_weights.copy(deep=True),
        )


def explain_weight_construction(
    frame: pd.DataFrame,
    *,
    tilt_columns: Sequence[str],
    base_weight_column: str = "benchmark_weight",
    target_weight_column: str = "index_weight",
    weight_tolerance: float = 1e-8,
    require_reconciliation: bool = True,
) -> WeightExplanationResult:
    """Apply and normalize tilts in order, recording each weight contribution."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a non-empty pandas DataFrame")
    tilts = tuple(str(column) for column in tilt_columns)
    if not tilts:
        raise ValueError("tilt_columns must not be empty")
    required = {base_weight_column, target_weight_column, *tilts}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"weight-explanation input is missing columns: {missing}")
    if frame.index.has_duplicates:
        raise ValueError("weight-explanation instrument index must be unique")
    working = frame.loc[:, [base_weight_column, target_weight_column, *tilts]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if working.isna().any().any() or not np.isfinite(
        working.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "weight-explanation inputs must contain finite numeric values"
        )
    base = _normalized_weights(
        working[base_weight_column],
        base_weight_column,
        weight_tolerance,
    )
    target = _normalized_weights(
        working[target_weight_column],
        target_weight_column,
        weight_tolerance,
    )
    if (working.loc[:, tilts] < 0).any().any():
        raise ValueError("tilt multipliers must be non-negative")

    current = base.copy(deep=True)
    detail_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for sequence, tilt_name in enumerate(tilts, start=1):
        previous = current.copy(deep=True)
        unnormalized = previous * working[tilt_name]
        total = float(unnormalized.sum())
        if total <= 0:
            raise ValueError(f"tilt {tilt_name!r} produces zero total weight")
        current = unnormalized / total
        change = current - previous
        detail_rows.append(
            pd.DataFrame(
                {
                    "instrument_id": frame.index,
                    "sequence": sequence,
                    "step": tilt_name,
                    "tilt_multiplier": working[tilt_name].to_numpy(dtype=float),
                    "previous_weight": previous.to_numpy(dtype=float),
                    "unnormalized_weight": unnormalized.to_numpy(dtype=float),
                    "resulting_weight": current.to_numpy(dtype=float),
                    "weight_change": change.to_numpy(dtype=float),
                    "one_way_contribution": (
                        0.5 * change.abs().to_numpy(dtype=float)
                    ),
                }
            )
        )
        summary_rows.append(
            {
                "sequence": sequence,
                "step": tilt_name,
                "one_way_weight_change": 0.5 * float(change.abs().sum()),
                "maximum_absolute_weight_change": float(change.abs().max()),
                "resulting_weight_sum": float(current.sum()),
            }
        )
    reconciled = bool(
        np.allclose(
            current.to_numpy(dtype=float),
            target.to_numpy(dtype=float),
            atol=weight_tolerance,
            rtol=0.0,
        )
    )
    if require_reconciliation and not reconciled:
        maximum = float((current - target).abs().max())
        raise ValueError(
            "stepwise tilt result does not reconcile to target weights; "
            f"maximum absolute difference is {maximum:.12g}"
        )
    current.name = "explained_index_weight"
    current.index.name = frame.index.name or "instrument_id"
    return WeightExplanationResult(
        detail=pd.concat(detail_rows, ignore_index=True),
        summary=pd.DataFrame(summary_rows),
        final_weights=current,
        reconciled=reconciled,
    )


def _normalized_weights(
    value: pd.Series,
    label: str,
    tolerance: float,
) -> pd.Series:
    if (value < -tolerance).any():
        raise ValueError(f"{label} must be non-negative")
    result = value.clip(lower=0.0).astype(float)
    total = float(result.sum())
    if not np.isclose(total, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError(f"{label} must sum to one; observed {total:.12g}")
    return result
