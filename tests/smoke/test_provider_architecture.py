"""Offline smoke tests for explicit provider selection and typed external data."""

from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd
import pytest

from icapa.data_sources import (
    FactSet,
    FileProvider,
    SnowflakePlaceholder,
    register_provider,
    registry,
)
from icapa.data_sources.exceptions import (
    DataCapabilityNotConfiguredError,
    DataSourceNotConfiguredError,
)
from icapa.portfolio_construction.rules.data_loading import (
    AddIdentifierFacts,
    AddReturns,
    AddThirdPartyData,
    AddUnderlyingIndex,
)
from icapa.tools.container import DataContext
from icapa.tools.enums.data_loading import ThirdPartyDataType


class SyntheticProvider:
    """Deterministic provider containing no credentials or production data."""

    def load_universe(self, universe_id, reference_date, effective_date, **kwargs):
        frame = pd.DataFrame(
            {
                "instrument_id": [1001, 1002],
                "name": ["Synthetic One", "Synthetic Two"],
                "country": ["US", "CA"],
                "industry": ["Technology", "Industrials"],
                "shares": [1_000.0, 2_000.0],
                "free_float": [0.8, 0.6],
                "price": [10.0, 20.0],
                "currency": ["USD", "CAD"],
                "base_currency": ["USD", "USD"],
                "fx_rate": [1.0, 0.75],
                "reference_date": [reference_date, reference_date],
                "effective_date": [effective_date, effective_date],
            }
        )
        frame["market_cap"] = (
            frame["shares"] * frame["free_float"] * frame["price"] * frame["fx_rate"]
        )
        frame["benchmark_weight"] = frame["market_cap"] / frame["market_cap"].sum()
        return frame

    def load_daily_market_data(self, instrument_ids, start_date, end_date, **kwargs):
        instruments = list(instrument_ids)
        return pd.DataFrame(
            {
                "instrument_id": instruments,
                "business_date": [pd.Timestamp(end_date)] * len(instruments),
                "price_return": [0.01, -0.005],
                "gross_dividend": [0.0, 0.001],
                "net_dividend": [0.0, 0.0008],
                "market_cap": [8_000.0, 18_000.0],
            }
        )

    def load_reference_data(self, instrument_ids, reference_date, fields, **kwargs):
        rows = {"instrument_id": list(instrument_ids)}
        for field_name in fields:
            rows[field_name] = [f"{field_name.upper()}-{item}" for item in instrument_ids]
        return pd.DataFrame(rows)

    def load_third_party_data(
        self, data_type, instrument_ids, fields, reference_date, parameters=None
    ):
        rows = {"instrument_id": list(instrument_ids)}
        for position, field_name in enumerate(fields, start=1):
            rows[field_name] = [float(position), float(position + 1)]
        return pd.DataFrame(rows)


@pytest.fixture
def synthetic_provider():
    register_provider("synthetic", SyntheticProvider(), replace=True)
    yield
    registry.unregister("synthetic")


def test_placeholders_have_no_default_connection_or_data():
    for provider, operation in (
        (FactSet(), lambda value: value.query("select 1")),
        (SnowflakePlaceholder(), lambda value: value.query("select 1")),
    ):
        assert provider.configured is False
        with pytest.raises(DataSourceNotConfiguredError):
            operation(provider)


def test_provider_selection_is_never_implicit(synthetic_provider):
    with pytest.raises(DataCapabilityNotConfiguredError, match="provider_name is required"):
        registry.resolve("load_universe")
    assert registry.resolve("load_universe", "synthetic").__class__ is SyntheticProvider


def test_typed_loading_rules_join_synthetic_data(synthetic_provider):
    context = DataContext(
        reference_date="2026-06-05",
        effective_date="2026-06-22",
        index_id="SYNTHETIC_DEMO",
    )
    AddUnderlyingIndex(
        universe_id="SYNTHETIC_UNIVERSE", provider_name="synthetic"
    ).execute(context)
    AddIdentifierFacts(fields=["isin"], provider_name="synthetic").execute(context)
    AddThirdPartyData(
        data_type=ThirdPartyDataType.FACTOR_DATA,
        fields=["quality_signal", "value_signal"],
        provider_name="synthetic",
    ).execute(context)
    AddReturns(provider_name="synthetic").execute(context)

    assert context.universe_id == "SYNTHETIC_UNIVERSE"
    assert {"isin", "quality_signal", "value_signal"}.issubset(context.cons.columns)
    assert context.daily is not None
    assert context.daily.index.names == ["instrument_id", "business_date"]


def test_third_party_data_requires_type_fields_and_provider():
    with pytest.raises(ValueError, match="data_type"):
        AddThirdPartyData(fields=["signal"], provider_name="synthetic")
    with pytest.raises(ValueError, match="fields"):
        AddThirdPartyData(
            data_type=ThirdPartyDataType.FACTOR_DATA,
            provider_name="synthetic",
        )
    with pytest.raises(ValueError, match="provider_name"):
        AddThirdPartyData(
            data_type=ThirdPartyDataType.FACTOR_DATA,
            fields=["signal"],
        )


def test_csv_and_excel_inputs():
    upload = pd.DataFrame(
        {"instrument_id": [1001, 1002], "name": ["Synthetic One", "Synthetic Two"]}
    )
    provider = FileProvider()
    with TemporaryDirectory() as directory:
        csv_path = Path(directory) / "upload.csv"
        excel_path = Path(directory) / "upload.xlsx"
        upload.to_csv(csv_path, index=False)
        upload.to_excel(excel_path, index=False)
        pd.testing.assert_frame_equal(provider.read(csv_path), upload)
        pd.testing.assert_frame_equal(provider.read(excel_path), upload)
