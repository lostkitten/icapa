"""Generic Brinson-Fachler performance attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import AnalyticsValidationError, BrinsonAttribution, BrinsonInput


_UNCLASSIFIED = "Unclassified"


def calculate_brinson_attribution(
    inputs: BrinsonInput,
    *,
    weight_tolerance: float = 1e-8,
) -> BrinsonAttribution:
    """Calculate single-period Brinson-Fachler attribution by classification.

    The function is deliberately data-source agnostic. Returns and
    classifications must already be point-in-time aligned by the caller.
    """

    if weight_tolerance <= 0:
        raise AnalyticsValidationError("weight_tolerance must be positive")

    frame = inputs.data.copy(deep=True)
    required = {
        inputs.period_column,
        inputs.instrument_column,
        inputs.classification_column,
        inputs.portfolio_weight_column,
        inputs.benchmark_weight_column,
        inputs.return_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AnalyticsValidationError(f"Brinson input is missing columns: {missing}")
    if frame.empty:
        raise AnalyticsValidationError("Brinson input must contain at least one row")
    if frame.duplicated([inputs.period_column, inputs.instrument_column]).any():
        raise AnalyticsValidationError(
            "Brinson input contains duplicate period/instrument rows"
        )

    numeric_columns = (
        inputs.portfolio_weight_column,
        inputs.benchmark_weight_column,
        inputs.return_column,
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise AnalyticsValidationError(
                f"Brinson input contains non-finite values in {column!r}"
            )
    if (
        frame[
            [inputs.portfolio_weight_column, inputs.benchmark_weight_column]
        ]
        < 0
    ).any().any():
        raise AnalyticsValidationError("Brinson weights must be non-negative")

    weight_sums = frame.groupby(inputs.period_column)[
        [inputs.portfolio_weight_column, inputs.benchmark_weight_column]
    ].sum()
    if not np.allclose(
        weight_sums.to_numpy(dtype=float),
        1.0,
        atol=weight_tolerance,
        rtol=0.0,
    ):
        raise AnalyticsValidationError(
            "Brinson portfolio and benchmark weights must sum to one in every period"
        )

    frame[inputs.classification_column] = frame[
        inputs.classification_column
    ].where(frame[inputs.classification_column].notna(), _UNCLASSIFIED)
    keys = [inputs.period_column, inputs.classification_column]

    def aggregate(group: pd.DataFrame) -> pd.Series:
        portfolio_weight = float(group[inputs.portfolio_weight_column].sum())
        benchmark_weight = float(group[inputs.benchmark_weight_column].sum())
        portfolio_contribution = float(
            (
                group[inputs.portfolio_weight_column]
                * group[inputs.return_column]
            ).sum()
        )
        benchmark_contribution = float(
            (
                group[inputs.benchmark_weight_column]
                * group[inputs.return_column]
            ).sum()
        )
        return pd.Series(
            {
                "portfolio_weight": portfolio_weight,
                "benchmark_weight": benchmark_weight,
                "portfolio_return": (
                    portfolio_contribution / portfolio_weight
                    if portfolio_weight > 0
                    else 0.0
                ),
                "benchmark_return": (
                    benchmark_contribution / benchmark_weight
                    if benchmark_weight > 0
                    else 0.0
                ),
                "portfolio_contribution": portfolio_contribution,
                "benchmark_contribution": benchmark_contribution,
            }
        )

    detail = frame.groupby(keys, dropna=False, sort=True).apply(
        aggregate, include_groups=False
    )
    period_benchmark_return = detail["benchmark_contribution"].groupby(
        level=inputs.period_column
    ).sum()
    benchmark_total = detail.index.get_level_values(inputs.period_column).map(
        period_benchmark_return
    )

    active_group_weight = detail["portfolio_weight"] - detail["benchmark_weight"]
    group_return_difference = (
        detail["portfolio_return"] - detail["benchmark_return"]
    )
    detail["allocation"] = active_group_weight * (
        detail["benchmark_return"] - benchmark_total.to_numpy(dtype=float)
    )
    detail["selection"] = detail["benchmark_weight"] * group_return_difference
    detail["interaction"] = active_group_weight * group_return_difference
    detail["total_attribution"] = detail[
        ["allocation", "selection", "interaction"]
    ].sum(axis=1)
    detail["active_contribution"] = (
        detail["portfolio_contribution"] - detail["benchmark_contribution"]
    )

    totals = detail.groupby(level=inputs.period_column)[
        [
            "portfolio_contribution",
            "benchmark_contribution",
            "allocation",
            "selection",
            "interaction",
            "total_attribution",
            "active_contribution",
        ]
    ].sum()
    totals = totals.rename(
        columns={
            "portfolio_contribution": "portfolio_return",
            "benchmark_contribution": "benchmark_return",
        }
    )
    totals["active_return"] = (
        totals["portfolio_return"] - totals["benchmark_return"]
    )
    if not np.allclose(
        totals["total_attribution"],
        totals["active_return"],
        atol=max(weight_tolerance, 1e-12),
        rtol=0.0,
    ):
        raise AnalyticsValidationError(
            "Brinson attribution does not reconcile to active return"
        )

    return BrinsonAttribution(detail=detail, totals=totals)


__all__ = ["calculate_brinson_attribution"]
