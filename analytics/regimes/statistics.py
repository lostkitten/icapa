"""Provider-neutral regime analysis over aligned return series."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


class RegimeAnalysisDependencyError(ImportError):
    """Raised when optional statistical tests require an unavailable library."""


@dataclass(frozen=True, slots=True)
class RegimeAnalysisResult:
    """Regime summaries, transitions, and optional distribution tests."""

    observations: pd.DataFrame
    summary: pd.DataFrame
    transition_counts: pd.DataFrame
    transition_probabilities: pd.DataFrame
    distribution_tests: pd.DataFrame

    def __post_init__(self) -> None:
        for name in (
            "observations",
            "summary",
            "transition_counts",
            "transition_probabilities",
            "distribution_tests",
        ):
            object.__setattr__(self, name, getattr(self, name).copy(deep=True))


def analyze_regimes(
    returns: pd.Series | pd.DataFrame,
    regimes: pd.Series,
    *,
    annualization_factor: int = 252,
    include_distribution_tests: bool = True,
) -> RegimeAnalysisResult:
    """Calculate per-regime statistics, transitions, and rank-based tests."""

    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    frame = _return_frame(returns)
    regime_series = _regime_series(regimes)
    aligned = frame.join(regime_series.rename("regime"), how="inner").dropna()
    if aligned.empty:
        raise ValueError("returns and regimes have no aligned observations")

    return_columns = list(frame.columns)
    summary_rows: list[dict[str, object]] = []
    for regime, group in aligned.groupby("regime", sort=True):
        for column in return_columns:
            values = group[column]
            total = float((1.0 + values).prod() - 1.0)
            count = len(values)
            annualized_return = float(
                (1.0 + total) ** (annualization_factor / count) - 1.0
            )
            volatility = float(
                values.std(ddof=1) * np.sqrt(annualization_factor)
            )
            downside = values.where(values < 0, 0.0)
            downside_deviation = float(
                np.sqrt(np.square(downside).mean())
                * np.sqrt(annualization_factor)
            )
            levels = (1.0 + values).cumprod()
            drawdown = levels.div(levels.cummax()).sub(1.0)
            summary_rows.append(
                {
                    "regime": regime,
                    "series": column,
                    "observations": count,
                    "mean_return": float(values.mean()),
                    "median_return": float(values.median()),
                    "total_return": total,
                    "annualized_return": annualized_return,
                    "annualized_volatility": volatility,
                    "downside_deviation": downside_deviation,
                    "maximum_drawdown": float(drawdown.min()),
                    "positive_return_ratio": float((values > 0).mean()),
                }
            )

    counts, probabilities = calculate_regime_transitions(aligned["regime"])
    tests = (
        _distribution_tests(aligned, return_columns)
        if include_distribution_tests
        else _empty_distribution_tests()
    )
    observations = aligned.reset_index()
    observations = observations.rename(
        columns={observations.columns[0]: "business_date"}
    )
    return RegimeAnalysisResult(
        observations=observations,
        summary=pd.DataFrame(summary_rows),
        transition_counts=counts,
        transition_probabilities=probabilities,
        distribution_tests=tests,
    )


def calculate_regime_transitions(
    regimes: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return transition counts and row-normalized probabilities."""

    values = _regime_series(regimes).dropna()
    if len(values) < 2:
        labels = sorted(values.astype(str).unique())
        empty = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
        empty.index.name = "from_regime"
        empty.columns.name = "to_regime"
        return empty, empty.astype(float)
    previous = values.iloc[:-1].astype(str).reset_index(drop=True)
    current = values.iloc[1:].astype(str).reset_index(drop=True)
    counts = pd.crosstab(previous, current, dropna=False)
    labels = sorted(set(previous) | set(current))
    counts = counts.reindex(index=labels, columns=labels, fill_value=0)
    counts.index.name = "from_regime"
    counts.columns.name = "to_regime"
    row_totals = counts.sum(axis=1).replace(0, np.nan)
    probabilities = counts.div(row_totals, axis=0).fillna(0.0)
    return counts, probabilities


def _distribution_tests(
    frame: pd.DataFrame,
    return_columns: list[str],
) -> pd.DataFrame:
    try:
        from scipy.stats import kruskal, mannwhitneyu
    except ImportError as exc:
        raise RegimeAnalysisDependencyError(
            "regime distribution tests require scipy; rerun with "
            "include_distribution_tests=False or install the analytics "
            "statistical dependency"
        ) from exc

    rows: list[dict[str, object]] = []
    regimes = tuple(sorted(frame["regime"].astype(str).unique()))
    for column in return_columns:
        samples = [
            frame.loc[frame["regime"].astype(str).eq(regime), column].to_numpy(
                dtype=float
            )
            for regime in regimes
        ]
        if len(samples) >= 2 and all(len(sample) > 0 for sample in samples):
            statistic, p_value = kruskal(*samples)
            rows.append(
                {
                    "series": column,
                    "test": "kruskal_wallis",
                    "regime_a": None,
                    "regime_b": None,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
        for regime_a, regime_b in combinations(regimes, 2):
            sample_a = frame.loc[
                frame["regime"].astype(str).eq(regime_a),
                column,
            ].to_numpy(dtype=float)
            sample_b = frame.loc[
                frame["regime"].astype(str).eq(regime_b),
                column,
            ].to_numpy(dtype=float)
            statistic, p_value = mannwhitneyu(
                sample_a,
                sample_b,
                alternative="two-sided",
            )
            rows.append(
                {
                    "series": column,
                    "test": "mann_whitney_u",
                    "regime_a": regime_a,
                    "regime_b": regime_b,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
    return pd.DataFrame(rows, columns=_distribution_test_columns())


def _empty_distribution_tests() -> pd.DataFrame:
    return pd.DataFrame(columns=_distribution_test_columns())


def _distribution_test_columns() -> list[str]:
    return [
        "series",
        "test",
        "regime_a",
        "regime_b",
        "statistic",
        "p_value",
    ]


def _return_frame(value: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        name = str(value.name or "return")
        frame = value.to_frame(name)
    elif isinstance(value, pd.DataFrame):
        frame = value.copy(deep=True)
    else:
        raise TypeError("returns must be a pandas Series or DataFrame")
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("returns must not be empty")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any() or not np.isfinite(
        frame.to_numpy(dtype=float)
    ).all():
        raise ValueError("returns must contain finite numeric values")
    if (frame < -1.0).any().any():
        raise ValueError("returns cannot contain values below -100%")
    if frame.index.has_duplicates:
        raise ValueError("returns index must be unique")
    return frame.sort_index().astype(float)


def _regime_series(value: pd.Series) -> pd.Series:
    if not isinstance(value, pd.Series) or value.empty:
        raise ValueError("regimes must be a non-empty pandas Series")
    if value.index.has_duplicates:
        raise ValueError("regimes index must be unique")
    result = value.copy(deep=True)
    missing = result.isna() | result.astype(str).str.strip().eq("")
    if missing.any():
        raise ValueError("regimes must not contain missing or empty values")
    return result.sort_index()
