"""Offline smoke test for the canonical data-loading contract."""

import importlib.util
from pathlib import Path

import pandas as pd


# Load the isolated contract module without importing the package root.  The
# smoke test must not require database drivers or optional reporting packages.
contract_path = Path(__file__).parents[2] / "data_sources" / "contracts.py"
spec = importlib.util.spec_from_file_location("icapa_data_contracts", contract_path)
contracts = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(contracts)
validate_daily_market_data = contracts.validate_daily_market_data
validate_universe = contracts.validate_universe
validate_review_dates = contracts.validate_review_dates


def test_synthetic_data_contract():
    reference_date = pd.Timestamp("2026-06-05")
    effective_date = pd.Timestamp("2026-06-22")

    universe = pd.DataFrame(
        {
            "instrument_id": [900001, 900002],
            "name": ["Example Alpha", "Example Beta"],
            "country": ["US", "GB"],
            "industry": [10, 20],
            "shares": [1_000_000.0, 2_000_000.0],
            "free_float": [0.80, 0.75],
            "price": [25.0, 10.0],
            "currency": ["USD", "GBP"],
            "base_currency": ["USD", "USD"],
            "fx_rate": [1.0, 1.25],
            "reference_date": [reference_date, reference_date],
            "effective_date": [effective_date, effective_date],
        }
    )
    universe["market_cap"] = (
        universe["shares"]
        * universe["free_float"]
        * universe["price"]
        * universe["fx_rate"]
    )
    universe["benchmark_weight"] = universe["market_cap"] / universe["market_cap"].sum()

    checked_universe = validate_universe(universe)
    assert checked_universe["benchmark_weight"].sum() == 1.0
    assert set(contracts.UNIVERSE_COLUMNS).issubset(checked_universe.columns)

    daily = pd.DataFrame(
        {
            "instrument_id": [900001, 900002],
            "business_date": [reference_date, reference_date],
            "price_return": [0.01, -0.005],
            "gross_dividend": [0.0, 0.0],
            "net_dividend": [0.0, 0.0],
            "market_cap": checked_universe["market_cap"],
        }
    )
    checked_daily = validate_daily_market_data(daily, reference_date=reference_date)
    assert set(checked_daily["instrument_id"]) == set(checked_universe["instrument_id"])

    try:
        validate_review_dates("2026-06-23", "2026-06-22")
    except ValueError:
        pass
    else:
        raise AssertionError("reference_date after effective_date must be rejected")


if __name__ == "__main__":
    test_synthetic_data_contract()
    print("canonical synthetic-data smoke test passed")
