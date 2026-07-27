"""Canonical names used at the provider and data-loading boundary."""

from icapa.tools.base_enum import BaseEnum


class DataSource(str, BaseEnum):
    GIS = "gis"
    FACTSET = "factset"
    ICE_DATA_INDICES = "ice_data_indices"
    SNOWFLAKE = "snowflake"
    FILE = "file"

    @classmethod
    def metadata(cls):
        return {
            "gis": {"configured": False, "kind": "sql_server"},
            "factset": {"configured": False, "kind": "sql_server"},
            "ice_data_indices": {"configured": False, "kind": "library"},
            "snowflake": {"configured": False, "kind": "placeholder"},
            "file": {"configured": True, "kind": "csv_excel"},
        }


class ThirdPartyDataType(str, BaseEnum):
    """Explicit categories for non-canonical external datasets."""

    FACTOR_DATA = "ThirdPartyFactorData"
    EMISSIONS = "ThirdPartyEmissions"
    LIQUIDITY_DATA = "ThirdPartyLiquidityData"


class IdentifierType(str, BaseEnum):
    INSTRUMENT_ID = "instrument_id"
    ISIN = "isin"
    CUSIP = "cusip"


class UnderlyingIndexColumns(str, BaseEnum):
    INSTRUMENT_ID = "instrument_id"
    NAME = "name"
    COUNTRY = "country"
    INDUSTRY = "industry"
    SHARES = "shares"
    FREE_FLOAT = "free_float"
    PRICE = "price"
    CURRENCY = "currency"
    BASE_CURRENCY = "base_currency"
    FX_RATE = "fx_rate"
    MARKET_CAP = "market_cap"
    BENCHMARK_WEIGHT = "benchmark_weight"
    REFERENCE_DATE = "reference_date"
    EFFECTIVE_DATE = "effective_date"


class DailyMarketColumns(str, BaseEnum):
    INSTRUMENT_ID = "instrument_id"
    BUSINESS_DATE = "business_date"
    PRICE_RETURN = "price_return"
    GROSS_DIVIDEND = "gross_dividend"
    NET_DIVIDEND = "net_dividend"
    MARKET_CAP = "market_cap"
