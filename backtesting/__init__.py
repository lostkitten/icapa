"""Provider-neutral backtesting entry points."""

from .backtester import (
    BacktestMetadata,
    BacktestResult,
    Backtester,
    ReviewResultMetadata,
)
from .calendar import Calendar
from .simulation_params import (
    DividendTreatment,
    RebalanceTiming,
    SimulationParams,
    WeightDrift,
)
from .simulator import IndexSimulationResult, IndexSimulator

__all__ = [
    "Backtester",
    "BacktestMetadata",
    "BacktestResult",
    "Calendar",
    "DividendTreatment",
    "IndexSimulationResult",
    "IndexSimulator",
    "RebalanceTiming",
    "ReviewResultMetadata",
    "SimulationParams",
    "WeightDrift",
]
