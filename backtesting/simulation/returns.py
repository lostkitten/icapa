"""Canonical return arithmetic for fixed-weight and stateful simulations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .enums import DividendTreatment

REQUIRED_RETURN_COLUMNS = {
    "instrument_id",
    "business_date",
    "price_return",
    "gross_dividend",
    "net_dividend",
}


def normalize_weights(weights: pd.Series, label: str = "weights") -> pd.Series:
    """Return finite, non-negative weights normalized to one."""

    result = pd.to_numeric(weights, errors="raise").astype(float)
    if result.index.has_duplicates:
        raise ValueError(f"{label} contains duplicate instrument_id values")
    if (
        not np.isfinite(result).all()
        or (result < 0).any()
        or float(result.sum()) <= 0
    ):
        raise ValueError(f"{label} must be finite, non-negative, and non-zero")
    result = result / float(result.sum())
    result.index.name = "instrument_id"
    return result


def select_held_observations(
    observations: pd.DataFrame,
    instrument_ids: pd.Index,
    business_date: pd.Timestamp,
) -> pd.DataFrame:
    """Align one business day's observations to the held instruments."""

    missing_ids = instrument_ids.difference(observations.index)
    if len(missing_ids):
        raise ValueError(
            "daily market data is missing held instruments on "
            f"{business_date.date()}: {list(missing_ids)}"
        )
    return observations.loc[instrument_ids]


def portfolio_return_factors(
    observations: pd.DataFrame,
    weights: pd.Series,
    dividend_treatment: DividendTreatment | str,
    business_date: pd.Timestamp,
) -> dict[str, float]:
    """Calculate price, gross-total, and net-total factors for one business day."""

    treatment = DividendTreatment(dividend_treatment)
    selected = select_held_observations(
        observations,
        weights.index,
        business_date=business_date,
    )
    price_factor = float((weights * (1.0 + selected["price_return"])).sum())
    gross_dividend = float((weights * selected["gross_dividend"]).sum())
    net_dividend = float((weights * selected["net_dividend"]).sum())

    if treatment is DividendTreatment.STANDARD:
        if gross_dividend >= 1.0 or net_dividend >= 1.0:
            raise ValueError("weighted dividend must be less than one")
        gross_factor = price_factor / (1.0 - gross_dividend)
        net_factor = price_factor / (1.0 - net_dividend)
    else:
        gross_factor = price_factor + gross_dividend
        net_factor = price_factor + net_dividend

    factors = {
        "price": price_factor,
        "gross": gross_factor,
        "net": net_factor,
    }
    if not np.isfinite(list(factors.values())).all() or min(factors.values()) <= 0:
        raise ValueError(
            f"daily portfolio factor is invalid on {business_date.date()}"
        )
    return factors


def calculate_index_returns(
    daily_market_data: pd.DataFrame,
    weights: pd.Series,
    dividend_treatment: DividendTreatment | str = DividendTreatment.STANDARD,
) -> pd.DataFrame:
    """Calculate fixed-weight index returns for every available business day."""

    missing = REQUIRED_RETURN_COLUMNS - set(daily_market_data.columns)
    if missing:
        raise ValueError(
            f"daily market data is missing columns: {sorted(missing)}"
        )
    data = daily_market_data.copy()
    data["business_date"] = pd.to_datetime(
        data["business_date"]
    ).dt.normalize()
    if data.duplicated(["instrument_id", "business_date"]).any():
        raise ValueError(
            "daily market data contains duplicate instrument_id/business_date rows"
        )
    normalized_weights = normalize_weights(weights)
    records: list[dict[str, object]] = []

    for business_date, observations in data.groupby("business_date", sort=True):
        indexed = observations.set_index("instrument_id", verify_integrity=True)
        factors = portfolio_return_factors(
            indexed,
            normalized_weights,
            dividend_treatment,
            business_date=business_date,
        )
        records.append(
            {
                "business_date": business_date,
                "price_return": factors["price"] - 1.0,
                "gross_total_return": factors["gross"] - 1.0,
                "net_total_return": factors["net"] - 1.0,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "price_return",
                "gross_total_return",
                "net_total_return",
            ],
            index=pd.DatetimeIndex([], name="business_date"),
        )
    return pd.DataFrame.from_records(records).set_index("business_date")


__all__ = [
    "calculate_index_returns",
    "normalize_weights",
    "portfolio_return_factors",
    "REQUIRED_RETURN_COLUMNS",
    "select_held_observations",
]
