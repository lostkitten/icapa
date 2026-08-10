"""Return attribution models for index research."""

from ..contracts import BrinsonAttribution, BrinsonInput
from .brinson import calculate_brinson_attribution
from .factor import (
    FactorAttributionResult,
    FactorAttributionSpec,
    calculate_factor_attribution,
)

__all__ = [
    "BrinsonAttribution",
    "BrinsonInput",
    "FactorAttributionResult",
    "FactorAttributionSpec",
    "calculate_brinson_attribution",
    "calculate_factor_attribution",
]
