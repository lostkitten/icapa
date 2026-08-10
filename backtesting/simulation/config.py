"""Configuration contracts for stateful daily index simulation."""

from dataclasses import dataclass, field

from .drift import (
    LegacyAbsoluteMarketCapDrift,
    PriceReturnDrift,
    WeightDriftStrategy,
)
from .enums import (
    DividendTreatment,
    RebalancePhase,
    RebalanceTiming,
    WeightDrift,
    WeightSnapshotMode,
)


@dataclass(frozen=True, slots=True)
class SimulationMaterialization:
    """Control large optional tables without changing index calculations."""

    weight_snapshots: WeightSnapshotMode = WeightSnapshotMode.DAILY
    include_asset_returns: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "weight_snapshots",
            WeightSnapshotMode(self.weight_snapshots),
        )


@dataclass(frozen=True)
class SimulationParams:
    dividend_treatment: DividendTreatment = DividendTreatment.STANDARD
    weight_drift: WeightDrift = WeightDrift.PRICE_RETURN
    rebalance_timing: RebalanceTiming = RebalanceTiming.NEXT_BUSINESS_DAY
    base_value: float = 100.0
    rebalance_phase: RebalancePhase = RebalancePhase.OPEN
    index_drift: WeightDriftStrategy | None = None
    benchmark_drift: WeightDriftStrategy | None = None
    materialization: SimulationMaterialization = field(
        default_factory=SimulationMaterialization
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dividend_treatment",
            DividendTreatment(self.dividend_treatment),
        )
        object.__setattr__(self, "weight_drift", WeightDrift(self.weight_drift))
        object.__setattr__(
            self,
            "rebalance_timing",
            RebalanceTiming(self.rebalance_timing),
        )
        object.__setattr__(
            self,
            "rebalance_phase",
            RebalancePhase(self.rebalance_phase),
        )
        if not isinstance(self.materialization, SimulationMaterialization):
            if isinstance(self.materialization, dict):
                object.__setattr__(
                    self,
                    "materialization",
                    SimulationMaterialization(**self.materialization),
                )
            else:
                raise TypeError(
                    "materialization must be a SimulationMaterialization"
                )
        for name in ("index_drift", "benchmark_drift"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, WeightDriftStrategy):
                raise TypeError(f"{name} must implement WeightDriftStrategy")
        if self.base_value <= 0:
            raise ValueError("base_value must be positive")

    @property
    def resolved_index_drift(self) -> WeightDriftStrategy:
        """Return the explicit index drift or its v1 compatibility adapter."""

        return self.index_drift or self._compatibility_drift()

    @property
    def resolved_benchmark_drift(self) -> WeightDriftStrategy:
        """Return the explicit benchmark drift or its v1 compatibility adapter."""

        return self.benchmark_drift or self._compatibility_drift()

    def _compatibility_drift(self) -> WeightDriftStrategy:
        if self.weight_drift is WeightDrift.PRICE_RETURN:
            return PriceReturnDrift()
        return LegacyAbsoluteMarketCapDrift()

    def _legacy_drift(self) -> WeightDriftStrategy:
        """Retain the v1 private helper name for compatibility."""

        return self._compatibility_drift()
