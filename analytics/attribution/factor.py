"""Transparent time-series factor attribution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FactorAttributionSpec:
    """Configuration for a linear factor attribution."""

    include_intercept: bool = True
    ridge_penalty: float = 0.0
    minimum_observations: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.ridge_penalty) or self.ridge_penalty < 0:
            raise ValueError("ridge_penalty must be finite and non-negative")
        if self.minimum_observations is not None and self.minimum_observations < 2:
            raise ValueError("minimum_observations must be at least two")


@dataclass(frozen=True, slots=True)
class FactorAttributionResult:
    """Daily factor contributions, fitted coefficients, and reconciliation."""

    coefficients: pd.Series
    daily: pd.DataFrame
    summary: pd.Series

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficients",
            self.coefficients.copy(deep=True),
        )
        object.__setattr__(self, "daily", self.daily.copy(deep=True))
        object.__setattr__(self, "summary", self.summary.copy(deep=True))


def calculate_factor_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    benchmark_returns: pd.Series | None = None,
    spec: FactorAttributionSpec | None = None,
) -> FactorAttributionResult:
    """Attribute portfolio or active returns to supplied factor returns.

    This is an explanatory linear model. It does not infer a risk model or load
    factor data. The caller must supply point-in-time aligned factor returns.
    """

    selected = spec or FactorAttributionSpec()
    portfolio = _numeric_series(portfolio_returns, "portfolio_returns")
    factors = _numeric_frame(factor_returns, "factor_returns")
    inputs = [portfolio.rename("portfolio_return"), factors]
    if benchmark_returns is not None:
        benchmark = _numeric_series(benchmark_returns, "benchmark_returns")
        inputs.append(benchmark.rename("benchmark_return"))
    aligned = pd.concat(inputs, axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("factor-attribution inputs have no aligned observations")

    factor_columns = list(factors.columns)
    minimum = selected.minimum_observations
    if minimum is None:
        minimum = len(factor_columns) + int(selected.include_intercept) + 1
    if len(aligned) < minimum:
        raise ValueError(
            f"factor attribution requires at least {minimum} aligned observations"
        )

    dependent = aligned["portfolio_return"].to_numpy(dtype=float)
    if "benchmark_return" in aligned:
        dependent = dependent - aligned["benchmark_return"].to_numpy(dtype=float)
    design = aligned[factor_columns].to_numpy(dtype=float)
    coefficient_names = factor_columns.copy()
    if selected.include_intercept:
        design = np.column_stack([np.ones(len(design)), design])
        coefficient_names.insert(0, "intercept")

    if selected.ridge_penalty > 0:
        penalty = np.eye(design.shape[1]) * selected.ridge_penalty
        if selected.include_intercept:
            penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ dependent,
        )
    else:
        coefficients, _, _, _ = np.linalg.lstsq(
            design,
            dependent,
            rcond=None,
        )

    fitted = design @ coefficients
    residual = dependent - fitted
    daily = pd.DataFrame(index=aligned.index)
    daily["attributed_return"] = dependent
    for position, factor in enumerate(factor_columns):
        coefficient_position = position + int(selected.include_intercept)
        daily[f"{factor}_contribution"] = (
            aligned[factor] * coefficients[coefficient_position]
        )
    if selected.include_intercept:
        daily["intercept_contribution"] = float(coefficients[0])
    daily["residual"] = residual
    contribution_columns = [
        column for column in daily if column.endswith("_contribution")
    ]
    daily["reconciled_return"] = (
        daily[contribution_columns].sum(axis=1) + daily["residual"]
    )
    if not np.allclose(
        daily["attributed_return"],
        daily["reconciled_return"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("factor attribution did not reconcile")

    total_sum_squares = float(
        np.square(dependent - dependent.mean()).sum()
    )
    residual_sum_squares = float(np.square(residual).sum())
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else np.nan
    )
    summary_values: dict[str, float] = {
        "observations": float(len(aligned)),
        "r_squared": r_squared,
        "residual_sum_squares": residual_sum_squares,
        "total_attributed_return": float(dependent.sum()),
        "total_residual": float(residual.sum()),
    }
    for column in contribution_columns:
        summary_values[f"total_{column}"] = float(daily[column].sum())
    daily.index.name = aligned.index.name or "business_date"
    return FactorAttributionResult(
        coefficients=pd.Series(
            coefficients,
            index=coefficient_names,
            dtype=float,
            name="coefficient",
        ),
        daily=daily,
        summary=pd.Series(summary_values, dtype=float, name="value"),
    )


def _numeric_series(value: pd.Series, label: str) -> pd.Series:
    if not isinstance(value, pd.Series) or value.empty:
        raise ValueError(f"{label} must be a non-empty pandas Series")
    result = pd.to_numeric(value.copy(deep=True), errors="coerce")
    if result.isna().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must contain finite numeric values")
    if result.index.has_duplicates:
        raise ValueError(f"{label} index must be unique")
    return result.sort_index().astype(float)


def _numeric_frame(value: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise ValueError(f"{label} must be a non-empty pandas DataFrame")
    if len(value.columns) == 0 or value.columns.has_duplicates:
        raise ValueError(f"{label} columns must be non-empty and unique")
    result = value.apply(pd.to_numeric, errors="coerce")
    if result.isna().any().any() or not np.isfinite(
        result.to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{label} must contain finite numeric values")
    if result.index.has_duplicates:
        raise ValueError(f"{label} index must be unique")
    return result.sort_index().astype(float)
