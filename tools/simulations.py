"""Deterministic index return calculations."""

from __future__ import annotations

import pandas as pd

from icapa.backtesting.simulation_params import DividendTreatment


REQUIRED_DAILY_COLUMNS = {
    "instrument_id",
    "business_date",
    "price_return",
    "gross_dividend",
    "net_dividend",
}


def calculate_index_returns(
    daily_market_data: pd.DataFrame,
    weights: pd.Series,
    dividend_treatment: DividendTreatment = DividendTreatment.NYSE,
) -> pd.DataFrame:
    """Calculate price, gross-total, and net-total index returns by business day."""

    missing = REQUIRED_DAILY_COLUMNS - set(daily_market_data.columns)
    if missing:
        raise ValueError(f"daily market data is missing columns: {sorted(missing)}")
    if weights.index.has_duplicates:
        raise ValueError("weights contain duplicate instrument_id values")
    if weights.isna().any() or (weights < 0).any() or float(weights.sum()) <= 0:
        raise ValueError("weights must be finite, non-negative, and have a positive sum")

    selected_treatment = DividendTreatment(dividend_treatment)
    normalised_weights = weights.astype(float) / float(weights.sum())
    records: list[dict] = []

    data = daily_market_data.copy()
    data["business_date"] = pd.to_datetime(data["business_date"]).dt.normalize()
    for business_date, observations in data.groupby("business_date", sort=True):
        observations = observations.set_index("instrument_id", verify_integrity=True)
        missing_ids = normalised_weights.index.difference(observations.index)
        if len(missing_ids):
            raise ValueError(
                f"daily market data is missing weighted instruments on {business_date.date()}: "
                f"{list(missing_ids)}"
            )
        observations = observations.loc[normalised_weights.index]
        price_factor = float((normalised_weights * (1.0 + observations["price_return"])).sum())

        if selected_treatment is DividendTreatment.NYSE:
            gross_dividend = float((normalised_weights * observations["gross_dividend"]).sum())
            net_dividend = float((normalised_weights * observations["net_dividend"]).sum())
            if gross_dividend >= 1.0 or net_dividend >= 1.0:
                raise ValueError("weighted dividend must be less than one")
            gross_factor = price_factor / (1.0 - gross_dividend)
            net_factor = price_factor / (1.0 - net_dividend)
        else:
            gross_factor = float(
                (
                    normalised_weights
                    * (1.0 + observations["price_return"] + observations["gross_dividend"])
                ).sum()
            )
            net_factor = float(
                (
                    normalised_weights
                    * (1.0 + observations["price_return"] + observations["net_dividend"])
                ).sum()
            )

        records.append(
            {
                "business_date": business_date,
                "price_return": price_factor - 1.0,
                "gross_total_return": gross_factor - 1.0,
                "net_total_return": net_factor - 1.0,
            }
        )

    return pd.DataFrame.from_records(records).set_index("business_date")


__all__ = ["calculate_index_returns"]
