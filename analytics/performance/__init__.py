"""Return, risk, and drawdown analytics for index research."""

from .metrics import (
    PerformanceMetrics,
    calculate_performance_metrics,
    calculate_rolling_risk,
    drawdown_series,
)

__all__ = [
    "PerformanceMetrics",
    "calculate_performance_metrics",
    "calculate_rolling_risk",
    "drawdown_series",
]
