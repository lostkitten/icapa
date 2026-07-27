"""Calculation choices for index return simulation."""

from icapa.tools.base_enum import BaseEnum


class DividendTreatment(str, BaseEnum):
    """Select one of the two explicit dividend calculation variants."""

    NYSE = "NYSE"
    NYSE_ALTERNATIVE = "NYSE_alternative"


class WeightDrift(str, BaseEnum):
    """Select how constituent weights evolve between review dates."""

    PRICE_RETURN = "price_return"
    MARKET_CAP = "market_cap"


class RebalanceTiming(str, BaseEnum):
    """Select how a non-business effective date is applied."""

    NEXT_BUSINESS_DAY = "next_business_day"
    EXACT_DATE = "exact_date"
