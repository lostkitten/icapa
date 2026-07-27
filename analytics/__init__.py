"""Client-neutral, side-effect-free analytics for ICAPA results.

The public API accepts already-produced backtest and simulation results. It
does not load data, call providers, mutate calculations, or write reports.
"""

from .brinson import calculate_brinson_attribution
from .contracts import (
    AnalyticsDiagnostic,
    AnalyticsResult,
    AnalyticsValidationError,
    BrinsonAttribution,
    BrinsonInput,
)
from .engine import AnalyticsEngine, analyze_backtest

__all__ = [
    "AnalyticsDiagnostic",
    "AnalyticsEngine",
    "AnalyticsResult",
    "AnalyticsValidationError",
    "BrinsonAttribution",
    "BrinsonInput",
    "analyze_backtest",
    "calculate_brinson_attribution",
]
