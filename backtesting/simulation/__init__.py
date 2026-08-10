"""Daily index-simulation domain."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "SimulationMaterialization": ".config",
    "SimulationParams": ".config",
    "AbsoluteMarketCapCompatibilityDrift": ".drift",
    "CapitalizationDrift": ".drift",
    "LegacyAbsoluteMarketCapDrift": ".drift",
    "PriceReturnDrift": ".drift",
    "RelativeCapitalizationDrift": ".drift",
    "WeightDriftStrategy": ".drift",
    "IndexSimulator": ".engine",
    "DividendTreatment": ".enums",
    "RebalancePhase": ".enums",
    "RebalanceTiming": ".enums",
    "WeightDrift": ".enums",
    "WeightSnapshotMode": ".enums",
    "IndexSimulationResult": ".models",
    "SimulationCheckpoint": ".models",
    "calculate_index_returns": ".returns",
    "normalize_weights": ".returns",
    "portfolio_return_factors": ".returns",
    "ImmutableSimulationSegment": ".segments",
    "DefaultSimulationIdentityService": ".cache_contracts",
    "SimulationCacheMissError": ".cache_contracts",
    "SimulationCachePolicy": ".cache_contracts",
    "SimulationCacheStore": ".cache_contracts",
    "SimulationIdentityService": ".cache_contracts",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Load one simulation symbol without eager engine initialization."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
