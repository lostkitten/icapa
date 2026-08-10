"""Euler risk contributions from an explicit covariance matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class RiskContributionResult:
    """Portfolio and optional active-risk contribution tables."""

    portfolio: pd.DataFrame
    active: pd.DataFrame
    summary: pd.Series

    def __post_init__(self) -> None:
        object.__setattr__(self, "portfolio", self.portfolio.copy(deep=True))
        object.__setattr__(self, "active", self.active.copy(deep=True))
        object.__setattr__(self, "summary", self.summary.copy(deep=True))


def calculate_risk_contributions(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    benchmark_weights: pd.Series | None = None,
    weight_tolerance: float = 1e-8,
) -> RiskContributionResult:
    """Calculate instrument contributions to volatility and tracking error."""

    portfolio = _weight_series(weights, "weights", weight_tolerance)
    matrix = _covariance(covariance, portfolio.index)
    portfolio_table, portfolio_variance, portfolio_volatility = _contributions(
        portfolio,
        matrix,
        label="portfolio",
    )
    summary: dict[str, float] = {
        "portfolio_variance": portfolio_variance,
        "portfolio_volatility": portfolio_volatility,
    }
    active_table = pd.DataFrame(
        columns=[
            "active_weight",
            "marginal_tracking_error",
            "tracking_error_contribution",
            "tracking_error_contribution_ratio",
        ]
    )
    if benchmark_weights is not None:
        benchmark = _weight_series(
            benchmark_weights,
            "benchmark_weights",
            weight_tolerance,
        )
        if set(benchmark.index) != set(portfolio.index):
            raise ValueError(
                "benchmark_weights must contain the same instruments as weights"
            )
        benchmark = benchmark.reindex(portfolio.index)
        active_weights = portfolio - benchmark
        raw_table, variance, tracking_error = _contributions(
            active_weights,
            matrix,
            label="active",
        )
        active_table = raw_table.rename(
            columns={
                "weight": "active_weight",
                "marginal_risk": "marginal_tracking_error",
                "risk_contribution": "tracking_error_contribution",
                "risk_contribution_ratio": (
                    "tracking_error_contribution_ratio"
                ),
            }
        )
        summary["active_variance"] = variance
        summary["tracking_error"] = tracking_error
    return RiskContributionResult(
        portfolio=portfolio_table,
        active=active_table,
        summary=pd.Series(summary, dtype=float, name="value"),
    )


def _contributions(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    label: str,
) -> tuple[pd.DataFrame, float, float]:
    vector = weights.to_numpy(dtype=float)
    matrix = covariance.to_numpy(dtype=float)
    variance = float(vector @ matrix @ vector)
    if variance < -1e-12:
        raise ValueError(f"{label} variance is negative")
    variance = max(variance, 0.0)
    volatility = float(np.sqrt(variance))
    marginal = (
        matrix @ vector / volatility
        if volatility > 0
        else np.zeros_like(vector)
    )
    contributions = vector * marginal
    ratios = (
        contributions / volatility
        if volatility > 0
        else np.zeros_like(contributions)
    )
    table = pd.DataFrame(
        {
            "weight": vector,
            "marginal_risk": marginal,
            "risk_contribution": contributions,
            "risk_contribution_ratio": ratios,
        },
        index=weights.index.copy(),
    )
    table.index.name = weights.index.name or "instrument_id"
    if not np.isclose(
        float(contributions.sum()),
        volatility,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(f"{label} risk contributions did not reconcile")
    return table, variance, volatility


def _weight_series(
    value: pd.Series,
    label: str,
    tolerance: float,
) -> pd.Series:
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


def _covariance(value: pd.DataFrame, instruments: pd.Index) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise ValueError("covariance must be a non-empty pandas DataFrame")
    if value.index.has_duplicates or value.columns.has_duplicates:
        raise ValueError("covariance labels must be unique")
    missing = sorted(
        set(map(str, instruments))
        - set(map(str, value.index))
    )
    if missing:
        raise ValueError(f"covariance is missing instruments: {missing}")
    try:
        result = value.loc[instruments, instruments].astype(float)
    except KeyError as exc:
        raise ValueError(
            "covariance index and columns must contain the weight instruments"
        ) from exc
    array = result.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("covariance must contain finite numeric values")
    if not np.allclose(array, array.T, atol=1e-12, rtol=0.0):
        raise ValueError("covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(array)
    if float(eigenvalues.min()) < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    return result
