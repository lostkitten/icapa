"""Field-based portfolio model specifications for index research.

The compiler turns canonical constituent fields into the existing generic
``OptimizationModelSpec``. It does not prescribe a methodology or solver and
therefore remains usable by built-in recipes, private composite stages, and
researcher-defined objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .constraints import (
    LinearConstraintSpec,
    NonlinearConstraintSpec,
    tracking_error_constraint,
    turnover_constraint,
)
from .models import (
    ObjectiveSpec,
    OptimizationModelSpec,
    SquaredDistanceObjectiveSpec,
    WeightVariableSpec,
)


@dataclass(frozen=True)
class GroupWeightConstraintSpec:
    """Bounds for selected values of a categorical constituent field.

    Bounds are absolute portfolio weights unless ``relative_to_field`` is
    supplied. In relative mode each tuple is interpreted as an asymmetric
    lower/upper deviation from that group's weight in the reference field.
    The same contract covers country, industry, issuer, or custom groupings.
    """

    field: str
    bounds: Mapping[Any, tuple[float, float]]
    relative_to_field: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise ValueError("group field must not be empty")
        if self.relative_to_field is not None and (
            not isinstance(self.relative_to_field, str)
            or not self.relative_to_field.strip()
        ):
            raise ValueError("relative_to_field must not be empty when supplied")
        values: dict[Any, tuple[float, float]] = {}
        for group, pair in dict(self.bounds).items():
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError(
                    "each group bound must contain a lower and upper value"
                )
            lower, upper = map(float, pair)
            if not np.isfinite([lower, upper]).all() or lower > upper:
                raise ValueError(
                    f"group bounds are invalid for {group!r}"
                )
            values[group] = (lower, upper)
        if not values:
            raise ValueError("group bounds must not be empty")
        if self.name is not None and not self.name.strip():
            raise ValueError("group constraint name must not be empty")
        object.__setattr__(self, "bounds", MappingProxyType(values))


@dataclass(frozen=True)
class FieldExposureConstraintSpec:
    """Lower and upper portfolio exposure to one numeric field."""

    field: str
    lower: float = -np.inf
    upper: float = np.inf
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise ValueError("exposure field must not be empty")
        lower = float(self.lower)
        upper = float(self.upper)
        if np.isnan(lower) or np.isnan(upper) or lower > upper:
            raise ValueError("exposure bounds are invalid")
        if self.name is not None and not self.name.strip():
            raise ValueError("exposure constraint name must not be empty")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class LiquidityConstraintSpec:
    """Convert point-in-time capacity into per-instrument weight bounds."""

    capacity_field: str
    portfolio_value: float
    participation_rate: float = 1.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.capacity_field, str)
            or not self.capacity_field.strip()
        ):
            raise ValueError("capacity_field must not be empty")
        if (
            not np.isfinite(self.portfolio_value)
            or self.portfolio_value <= 0
        ):
            raise ValueError("portfolio_value must be finite and positive")
        if (
            not np.isfinite(self.participation_rate)
            or self.participation_rate <= 0
        ):
            raise ValueError(
                "participation_rate must be finite and positive"
            )


@dataclass(frozen=True)
class PortfolioModelSpec:
    """Compile canonical constituent fields into a solver-neutral model.

    ``objective`` is optional. Without one, squared active weight is minimized
    against ``target_weight_field``. Researchers can still supply any
    ``ObjectiveSpec`` or bypass this layer and use ``OptimizationProblem``
    directly.
    """

    name: str
    constituents: pd.DataFrame
    objective: ObjectiveSpec | None = None
    instrument_id_field: str = "instrument_id"
    target_weight_field: str = "benchmark_weight"
    initial_weight_field: str | None = None
    previous_weight_field: str | None = None
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    lower_bound_field: str | None = None
    upper_bound_field: str | None = None
    investment_level: float = 1.0
    groups: tuple[GroupWeightConstraintSpec, ...] = ()
    exposures: tuple[FieldExposureConstraintSpec, ...] = ()
    liquidity: LiquidityConstraintSpec | None = None
    maximum_one_way_turnover: float | None = None
    tracking_error_covariance: pd.DataFrame | np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    maximum_tracking_error: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("portfolio model name must not be empty")
        if not isinstance(self.constituents, pd.DataFrame):
            raise TypeError("constituents must be a pandas DataFrame")
        if self.constituents.empty:
            raise ValueError("constituents must not be empty")
        if self.objective is not None and not isinstance(
            self.objective,
            ObjectiveSpec,
        ):
            raise TypeError("objective must implement ObjectiveSpec")
        for value, label in (
            (self.lower_bound, "lower_bound"),
            (self.upper_bound, "upper_bound"),
            (self.investment_level, "investment_level"),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must not exceed upper_bound")
        if self.maximum_one_way_turnover is not None and (
            not np.isfinite(self.maximum_one_way_turnover)
            or self.maximum_one_way_turnover < 0
        ):
            raise ValueError(
                "maximum_one_way_turnover must be finite and non-negative"
            )
        if (self.tracking_error_covariance is None) != (
            self.maximum_tracking_error is None
        ):
            raise ValueError(
                "tracking-error covariance and maximum must be supplied together"
            )
        if self.maximum_tracking_error is not None and (
            not np.isfinite(self.maximum_tracking_error)
            or self.maximum_tracking_error < 0
        ):
            raise ValueError(
                "maximum_tracking_error must be finite and non-negative"
            )
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "exposures", tuple(self.exposures))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    def compile(self) -> OptimizationModelSpec:
        """Build an inspectable generic optimization model."""

        frame = _instrument_frame(
            self.constituents,
            self.instrument_id_field,
        )
        instrument_ids = tuple(frame.index)
        target = _numeric_field(frame, self.target_weight_field)
        initial_field = self.initial_weight_field or self.target_weight_field
        initial = _numeric_field(frame, initial_field)
        lower = _bound_vector(
            frame,
            field=self.lower_bound_field,
            fallback=self.lower_bound,
            label="lower bound",
        )
        upper = _bound_vector(
            frame,
            field=self.upper_bound_field,
            fallback=self.upper_bound,
            label="upper bound",
        )
        if self.liquidity is not None:
            capacity = _numeric_field(
                frame,
                self.liquidity.capacity_field,
            )
            if (capacity < 0).any():
                raise ValueError("liquidity capacity must be non-negative")
            capacity_weight = (
                capacity
                * float(self.liquidity.participation_rate)
                / float(self.liquidity.portfolio_value)
            )
            upper = np.minimum(upper, capacity_weight)
        if np.any(lower > upper):
            raise ValueError(
                "instrument, field, and liquidity bounds are infeasible"
            )

        linear: list[LinearConstraintSpec] = []
        for group_spec in self.groups:
            _require_columns(
                frame,
                [
                    group_spec.field,
                    *(
                        ()
                        if group_spec.relative_to_field is None
                        else (group_spec.relative_to_field,)
                    ),
                ],
            )
            groups = frame[group_spec.field]
            available = set(groups.dropna().unique())
            missing = set(group_spec.bounds).difference(available)
            if missing:
                raise ValueError(
                    f"group values are absent from {group_spec.field}: "
                    f"{sorted(map(str, missing))}"
                )
            reference = (
                None
                if group_spec.relative_to_field is None
                else _numeric_field(frame, group_spec.relative_to_field)
            )
            for group, (configured_lower, configured_upper) in sorted(
                group_spec.bounds.items(),
                key=lambda item: str(item[0]),
            ):
                coefficients = (groups == group).to_numpy(dtype=float)
                if reference is None:
                    constraint_lower = configured_lower
                    constraint_upper = configured_upper
                else:
                    reference_weight = float(
                        reference[groups.to_numpy() == group].sum()
                    )
                    constraint_lower = reference_weight + configured_lower
                    constraint_upper = reference_weight + configured_upper
                linear.append(
                    LinearConstraintSpec(
                        coefficients=coefficients,
                        lower=constraint_lower,
                        upper=constraint_upper,
                        name=(
                            f"{group_spec.name or group_spec.field}:{group}"
                        ),
                    )
                )

        for exposure in self.exposures:
            coefficients = _numeric_field(frame, exposure.field)
            linear.append(
                LinearConstraintSpec(
                    coefficients=coefficients,
                    lower=exposure.lower,
                    upper=exposure.upper,
                    name=exposure.name or f"exposure:{exposure.field}",
                )
            )

        nonlinear: list[NonlinearConstraintSpec] = []
        if self.maximum_one_way_turnover is not None:
            if self.previous_weight_field is None:
                raise ValueError(
                    "previous_weight_field is required for a turnover limit"
                )
            nonlinear.append(
                turnover_constraint(
                    _numeric_field(frame, self.previous_weight_field),
                    self.maximum_one_way_turnover,
                )
            )
        if self.maximum_tracking_error is not None:
            covariance = _aligned_covariance(
                self.tracking_error_covariance,
                instrument_ids,
            )
            nonlinear.append(
                tracking_error_constraint(
                    covariance,
                    target,
                    self.maximum_tracking_error,
                )
            )

        objective = self.objective or SquaredDistanceObjectiveSpec(target)
        return OptimizationModelSpec(
            name=self.name,
            variables=WeightVariableSpec(
                instrument_ids=instrument_ids,
                initial_weights=initial,
                lower_bounds=lower,
                upper_bounds=upper,
                investment_level=self.investment_level,
            ),
            objective=objective,
            linear_constraints=tuple(linear),
            nonlinear_constraints=tuple(nonlinear),
            metadata={
                **dict(self.metadata),
                "field_model": {
                    "instrument_id_field": self.instrument_id_field,
                    "target_weight_field": self.target_weight_field,
                    "initial_weight_field": initial_field,
                    "group_constraint_count": len(self.groups),
                    "exposure_constraint_count": len(self.exposures),
                    "has_liquidity_bounds": self.liquidity is not None,
                    "has_turnover_limit": (
                        self.maximum_one_way_turnover is not None
                    ),
                    "has_tracking_error_limit": (
                        self.maximum_tracking_error is not None
                    ),
                },
            },
        )


def _instrument_frame(
    frame: pd.DataFrame,
    instrument_id_field: str,
) -> pd.DataFrame:
    result = frame.copy(deep=True)
    if instrument_id_field in result.columns:
        result = result.set_index(instrument_id_field, verify_integrity=True)
    elif result.index.name != instrument_id_field:
        raise KeyError(
            f"instrument identifier field is missing: {instrument_id_field}"
        )
    if not result.index.is_unique or result.index.hasnans:
        raise ValueError("instrument identifiers must be unique and non-null")
    return result


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise KeyError(f"constituent fields are missing: {sorted(missing)}")


def _numeric_field(frame: pd.DataFrame, field: str) -> np.ndarray:
    _require_columns(frame, (field,))
    values = pd.to_numeric(frame[field], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"constituent field must be finite: {field}")
    return values


def _bound_vector(
    frame: pd.DataFrame,
    *,
    field: str | None,
    fallback: float,
    label: str,
) -> np.ndarray:
    if field is None:
        return np.full(len(frame), float(fallback))
    try:
        return _numeric_field(frame, field)
    except (KeyError, ValueError) as exc:
        raise type(exc)(f"{label} field is invalid: {field}") from exc


def _aligned_covariance(
    value: pd.DataFrame | np.ndarray | None,
    instrument_ids: Sequence[Any],
) -> np.ndarray:
    if value is None:
        raise ValueError("tracking-error covariance is required")
    if isinstance(value, pd.DataFrame):
        missing_rows = set(instrument_ids).difference(value.index)
        missing_columns = set(instrument_ids).difference(value.columns)
        if missing_rows or missing_columns:
            raise ValueError(
                "tracking-error covariance is missing instruments"
            )
        matrix = value.loc[list(instrument_ids), list(instrument_ids)].to_numpy(
            dtype=float
        )
    else:
        matrix = np.asarray(value, dtype=float)
    expected = (len(instrument_ids), len(instrument_ids))
    if matrix.shape != expected or not np.isfinite(matrix).all():
        raise ValueError(
            "tracking-error covariance must be a finite square matrix"
        )
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError("tracking-error covariance must be symmetric")
    return matrix


__all__ = [
    "FieldExposureConstraintSpec",
    "GroupWeightConstraintSpec",
    "LiquidityConstraintSpec",
    "PortfolioModelSpec",
]
