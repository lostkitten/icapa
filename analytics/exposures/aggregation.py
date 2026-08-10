"""Generic weighted exposure aggregation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def calculate_weighted_exposures(
    frame: pd.DataFrame,
    exposure_columns: Sequence[str],
    *,
    group_columns: Sequence[str] = (),
    portfolio_weight_column: str = "index_weight",
    benchmark_weight_column: str = "benchmark_weight",
) -> pd.DataFrame:
    """Aggregate numeric fields under portfolio and benchmark weights."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a non-empty pandas DataFrame")
    fields = tuple(str(column) for column in exposure_columns)
    groups = tuple(str(column) for column in group_columns)
    if not fields:
        raise ValueError("exposure_columns must not be empty")
    required = {
        *fields,
        *groups,
        portfolio_weight_column,
        benchmark_weight_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"exposure input is missing columns: {missing}")

    working = frame.loc[
        :,
        [*groups, *fields, portfolio_weight_column, benchmark_weight_column],
    ].copy(deep=True)
    numeric = [*fields, portfolio_weight_column, benchmark_weight_column]
    for column in numeric:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    if working[numeric].isna().any().any() or not np.isfinite(
        working[numeric].to_numpy(dtype=float)
    ).all():
        raise ValueError("exposure inputs must contain finite numeric values")

    rows: list[dict[str, object]] = []
    grouped = (
        working.groupby(list(groups), dropna=False, sort=True)
        if groups
        else [((), working)]
    )
    for key, group in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(groups, key_values, strict=True))
        portfolio_total = float(group[portfolio_weight_column].sum())
        benchmark_total = float(group[benchmark_weight_column].sum())
        for field in fields:
            portfolio_exposure = float(
                (group[field] * group[portfolio_weight_column]).sum()
            )
            benchmark_exposure = float(
                (group[field] * group[benchmark_weight_column]).sum()
            )
            rows.append(
                {
                    **labels,
                    "field": field,
                    "portfolio_weight": portfolio_total,
                    "benchmark_weight": benchmark_total,
                    "portfolio_exposure": portfolio_exposure,
                    "benchmark_exposure": benchmark_exposure,
                    "active_exposure": portfolio_exposure - benchmark_exposure,
                }
            )
    return pd.DataFrame(rows)
