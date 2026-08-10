"""Index review generation and daily simulation."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "BacktestMetadata": ".reviews",
    "BacktestResult": ".reviews",
    "Backtester": ".reviews",
    "ReviewResultMetadata": ".reviews",
    "Calendar": ".calendar",
    "RebalanceFrequency": ".calendar",
    "AbsoluteMarketCapCompatibilityDrift": ".simulation",
    "CapitalizationDrift": ".simulation",
    "DividendTreatment": ".simulation",
    "IndexSimulationResult": ".simulation",
    "IndexSimulator": ".simulation",
    "PriceReturnDrift": ".simulation",
    "RebalancePhase": ".simulation",
    "RebalanceTiming": ".simulation",
    "RelativeCapitalizationDrift": ".simulation",
    "SimulationCheckpoint": ".simulation",
    "SimulationMaterialization": ".simulation",
    "SimulationParams": ".simulation",
    "WeightDrift": ".simulation",
    "WeightDriftStrategy": ".simulation",
    "WeightSnapshotMode": ".simulation",
    "calculate_index_returns": ".simulation",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Load one backtesting symbol without importing unrelated domains."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
