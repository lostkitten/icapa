"""Exposure and capacity normalization for analytics plugins."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from ..contracts import AnalyticsValidationError
from .support_reviews import flat_input_frame, review_frames

LIQUIDITY_FIELDS = ("average_daily_value_traded", "average_daily_volume", "liquidity", "liquidity_score")
CAPACITY_LIMIT_FIELDS = ("capacity_weight_limit", "weight_capacity", "capacity_limit")

def instrument_research_frame(
    backtest_result: object,
    explicit: pd.DataFrame | None,
    *,
    input_name: str,
) -> pd.DataFrame:
    review_rows: list[pd.DataFrame] = []
    for effective_date, review, frame in review_frames(backtest_result):
        working = frame.reset_index()
        if "effective_date" in working.columns:
            observed = pd.to_datetime(
                working.pop("effective_date"),
                errors="raise",
            ).dt.normalize()
            expected = pd.Timestamp(effective_date).normalize()
            if not observed.eq(expected).all():
                raise AnalyticsValidationError(
                    "constituent effective_date does not match its review"
                )
        working.insert(0, "effective_date", effective_date)
        if "reference_date" in working.columns:
            observed_reference_dates = pd.to_datetime(
                working["reference_date"],
                errors="raise",
            ).dt.normalize()
            expected_reference_date = getattr(
                review,
                "reference_date",
                None,
            )
            if expected_reference_date is not None:
                expected_reference_date = pd.Timestamp(
                    expected_reference_date
                ).normalize()
                if not observed_reference_dates.eq(
                    expected_reference_date
                ).all():
                    raise AnalyticsValidationError(
                        "constituent reference_date does not match its review"
                    )
                working["reference_date"] = expected_reference_date
            else:
                working["reference_date"] = observed_reference_dates
        review_rows.append(working)
    reviews = pd.concat(review_rows, ignore_index=True)
    keys = ["effective_date", "instrument_id"]
    if explicit is None:
        result = reviews
    else:
        supplied = flat_input_frame(explicit)
        missing = sorted(set(keys).difference(supplied.columns))
        if missing:
            raise AnalyticsValidationError(
                f"{input_name} is missing columns: {missing}"
            )
        supplied["effective_date"] = pd.to_datetime(
            supplied["effective_date"],
            errors="raise",
        ).dt.normalize()
        if supplied.duplicated(keys).any():
            raise AnalyticsValidationError(
                f"{input_name} contains duplicate review/instrument rows"
            )
        weights = reviews.loc[
            :,
            keys + ["index_weight", "benchmark_weight"],
        ]
        supplied = supplied.drop(
            columns=[
                column
                for column in ("index_weight", "benchmark_weight")
                if column in supplied
            ]
        )
        result = weights.merge(
            supplied,
            on=keys,
            how="right",
            validate="one_to_one",
        )
        if result[["index_weight", "benchmark_weight"]].isna().any().any():
            raise AnalyticsValidationError(
                f"{input_name} contains rows outside the backtest reviews"
            )
    required = {"effective_date", "instrument_id", "index_weight", "benchmark_weight"}
    missing = sorted(required.difference(result.columns))
    if missing:
        raise AnalyticsValidationError(
            f"research exposure data is missing columns: {missing}"
        )
    if result.duplicated(keys).any():
        raise AnalyticsValidationError(
            "research exposure data contains duplicate review/instrument rows"
        )
    for column in ("index_weight", "benchmark_weight"):
        result[column] = pd.to_numeric(result[column], errors="raise")
        values = result[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise AnalyticsValidationError(
                f"research exposure data contains invalid {column}"
            )
    return result.sort_values(keys, kind="mergesort").reset_index(drop=True)


def is_factor_or_signal_field(column: Any) -> bool:
    name = str(column).lower()
    return (
        name == "factor_score"
        or name.endswith("_factor")
        or name.endswith("_signal")
        or name.endswith("_score")
        or name.endswith("_zscore")
    )


def exposure_type(field_name: str) -> str:
    name = field_name.lower()
    return "signal" if name.endswith("_signal") else "factor"


def weighted_field_statistics(
    group: pd.DataFrame,
    field_name: str,
    tolerance: float,
) -> dict[str, Any]:
    values = pd.to_numeric(group[field_name], errors="coerce")
    available = values.notna() & np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    index_weights = group["index_weight"].astype(float)
    benchmark_weights = group["benchmark_weight"].astype(float)
    index_coverage = float(index_weights.where(available, 0.0).sum())
    benchmark_coverage = float(
        benchmark_weights.where(available, 0.0).sum()
    )
    index_exposure = (
        float((index_weights[available] * values[available]).sum())
        / index_coverage
        if index_coverage > tolerance
        else np.nan
    )
    benchmark_exposure = (
        float(
            (benchmark_weights[available] * values[available]).sum()
        )
        / benchmark_coverage
        if benchmark_coverage > tolerance
        else np.nan
    )
    return {
        "instrument_count": int(len(group)),
        "available_count": int(available.sum()),
        "missing_count": int((~available).sum()),
        "index_weight_coverage": index_coverage,
        "benchmark_weight_coverage": benchmark_coverage,
        "index_exposure": index_exposure,
        "benchmark_exposure": benchmark_exposure,
        "active_exposure": index_exposure - benchmark_exposure,
    }


def capacity_table(
    frame: pd.DataFrame,
    capacity_field: str,
    tolerance: float,
) -> pd.DataFrame:
    columns = [
        "effective_date",
        "instrument_id",
        "index_weight",
        "capacity_limit",
        "capacity_utilisation",
        "capacity_breach",
    ]
    if not capacity_field:
        return pd.DataFrame(columns=columns)
    values = pd.to_numeric(frame[capacity_field], errors="coerce")
    if (values.dropna() < 0).any():
        raise AnalyticsValidationError(
            "capacity limits must be non-negative"
        )
    available = values.notna() & np.isfinite(
        values.to_numpy(dtype=float, na_value=np.nan)
    )
    selected = frame.loc[available, ["effective_date", "instrument_id", "index_weight"]].copy()
    limits = values.loc[available].astype(float)
    selected["capacity_limit"] = limits.to_numpy()
    selected["capacity_utilisation"] = np.where(
        limits.to_numpy() > 0,
        selected["index_weight"].to_numpy(dtype=float) / limits.to_numpy(),
        np.nan,
    )
    selected["capacity_breach"] = (
        selected["index_weight"].to_numpy(dtype=float)
        > limits.to_numpy() + tolerance
    )
    return selected.loc[:, columns].sort_values(
        ["effective_date", "capacity_utilisation", "instrument_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
