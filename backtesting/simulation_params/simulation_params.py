"""Parameters for stateful daily index simulation."""

from dataclasses import dataclass

from .enums import DividendTreatment, RebalanceTiming, WeightDrift


@dataclass(frozen=True)
class SimulationParams:
    dividend_treatment: DividendTreatment = DividendTreatment.NYSE
    weight_drift: WeightDrift = WeightDrift.PRICE_RETURN
    rebalance_timing: RebalanceTiming = RebalanceTiming.NEXT_BUSINESS_DAY
    base_value: float = 100.0

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
        if self.base_value <= 0:
            raise ValueError("base_value must be positive")
