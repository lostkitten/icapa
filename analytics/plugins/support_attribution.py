"""Multi-period attribution linking helpers for analytics plugins."""
from __future__ import annotations
from typing import Any
import math
import pandas as pd
from ..contracts import AnalyticsValidationError

def carino_link_attribution(
    totals: pd.DataFrame,
    *,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "portfolio_return",
        "benchmark_return",
        "allocation",
        "selection",
        "interaction",
    }
    missing = sorted(required.difference(totals.columns))
    if missing:
        raise AnalyticsValidationError(
            f"attribution totals are missing columns: {missing}"
        )
    portfolio = pd.to_numeric(
        totals["portfolio_return"],
        errors="raise",
    ).astype(float)
    benchmark = pd.to_numeric(
        totals["benchmark_return"],
        errors="raise",
    ).astype(float)
    if (portfolio <= -1.0).any() or (benchmark <= -1.0).any():
        raise AnalyticsValidationError(
            "multi-period attribution returns must be greater than negative one"
        )
    portfolio_total = float((1.0 + portfolio).prod() - 1.0)
    benchmark_total = float((1.0 + benchmark).prod() - 1.0)
    active_total = portfolio_total - benchmark_total

    def coefficient(portfolio_return: float, benchmark_return: float) -> float:
        difference = portfolio_return - benchmark_return
        if abs(difference) <= tolerance:
            return 1.0 / (1.0 + portfolio_return)
        return (
            math.log1p(portfolio_return)
            - math.log1p(benchmark_return)
        ) / difference

    overall = coefficient(portfolio_total, benchmark_total)
    period_coefficients = pd.Series(
        [
            coefficient(float(left), float(right))
            for left, right in zip(portfolio, benchmark)
        ],
        index=totals.index,
        dtype=float,
    )
    components = ("allocation", "selection", "interaction")
    detail_rows: list[dict[str, Any]] = []
    linked_sums: dict[str, float] = {}
    period_name = totals.index.name or "period"
    for component in components:
        values = pd.to_numeric(totals[component], errors="raise").astype(
            float
        )
        linked = values * period_coefficients / overall
        linked_sums[component] = float(linked.sum())
        for period, value, period_coefficient, linked_value in zip(
            totals.index,
            values,
            period_coefficients,
            linked,
        ):
            detail_rows.append(
                {
                    period_name: period,
                    "component": component,
                    "period_contribution": float(value),
                    "linking_coefficient": float(
                        period_coefficient / overall
                    ),
                    "linked_contribution": float(linked_value),
                }
            )
    reconciled = float(sum(linked_sums.values()))
    if not math.isclose(
        reconciled,
        active_total,
        abs_tol=max(tolerance, 1e-12),
        rel_tol=0.0,
    ):
        raise AnalyticsValidationError(
            "linked attribution does not reconcile to multi-period active return"
        )
    linked_totals = pd.DataFrame(
        [
            {
                "component": component,
                "linked_contribution": contribution,
            }
            for component, contribution in linked_sums.items()
        ]
        + [
            {
                "component": "total_attribution",
                "linked_contribution": reconciled,
            },
            {
                "component": "multi_period_active_return",
                "linked_contribution": active_total,
            },
        ]
    )
    return pd.DataFrame(detail_rows), linked_totals
