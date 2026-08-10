"""Rolling return-window and covariance-estimation extension contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd


class ReturnWindowKind(str, Enum):
    """How a point-in-time return lookback is measured."""

    OBSERVATIONS = "observations"
    CALENDAR_DAYS = "calendar_days"


class CovarianceMissingDataPolicy(str, Enum):
    """How sample covariance handles partially observed return histories."""

    PAIRWISE = "pairwise"
    COMPLETE_CASE = "complete_case"


class CovarianceShrinkageTarget(str, Enum):
    """Deterministic target used by shrinkage covariance estimation."""

    DIAGONAL = "diagonal"
    SCALED_IDENTITY = "scaled_identity"


@dataclass(frozen=True)
class ResolvedReturnWindow:
    """Concrete business dates selected for one reference date."""

    reference_date: pd.Timestamp
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    business_dates: tuple[pd.Timestamp, ...]

    @property
    def observation_count(self) -> int:
        return len(self.business_dates)


@dataclass(frozen=True)
class ReturnWindowSpec:
    """Point-in-time rolling return window with an explicit observation policy."""

    lookback: int = 252
    kind: ReturnWindowKind = ReturnWindowKind.OBSERVATIONS
    minimum_observations: int = 20
    end_lag_observations: int = 0

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")
        if self.minimum_observations < 2:
            raise ValueError("minimum_observations must be at least two")
        if self.end_lag_observations < 0:
            raise ValueError("end_lag_observations must be non-negative")
        object.__setattr__(self, "kind", ReturnWindowKind(self.kind))

    def resolve(
        self,
        available_business_dates: Sequence[object] | pd.Index,
        reference_date: object,
    ) -> ResolvedReturnWindow:
        """Resolve a no-lookahead window from available business dates."""

        reference = pd.Timestamp(reference_date).normalize()
        dates = pd.DatetimeIndex(available_business_dates)
        dates = dates[~dates.isna()].normalize().unique().sort_values()
        dates = dates[dates <= reference]
        if len(dates) <= self.end_lag_observations:
            raise ValueError("no business date is available after applying the end lag")
        if self.end_lag_observations:
            dates = dates[: -self.end_lag_observations]
        if self.kind is ReturnWindowKind.OBSERVATIONS:
            selected = dates[-self.lookback :]
        else:
            cutoff = dates[-1] - pd.Timedelta(days=self.lookback)
            selected = dates[dates >= cutoff]
        if len(selected) < self.minimum_observations:
            raise ValueError(
                "resolved return window has fewer observations than "
                "minimum_observations"
            )
        business_dates = tuple(pd.Timestamp(item).normalize() for item in selected)
        return ResolvedReturnWindow(
            reference_date=reference,
            start_date=business_dates[0],
            end_date=business_dates[-1],
            business_dates=business_dates,
        )


@dataclass(frozen=True)
class CovarianceEstimate:
    """Validated covariance matrix plus estimator and sample diagnostics."""

    matrix: np.ndarray
    instrument_ids: Sequence[Any]
    observations: int
    estimator_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifiers = tuple(self.instrument_ids)
        matrix = np.asarray(self.matrix, dtype=float).copy()
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("covariance instrument_ids must be non-empty and unique")
        if matrix.shape != (len(identifiers), len(identifiers)):
            raise ValueError("covariance matrix shape must match instrument_ids")
        if not np.isfinite(matrix).all():
            raise ValueError("covariance matrix must be finite")
        if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
            raise ValueError("covariance matrix must be symmetric")
        if self.observations < 2:
            raise ValueError("covariance estimate requires at least two observations")
        if not isinstance(self.estimator_name, str) or not self.estimator_name.strip():
            raise ValueError("estimator_name must be a non-empty string")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "instrument_ids", identifiers)
        object.__setattr__(self, "observations", int(self.observations))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def minimum_eigenvalue(self) -> float:
        return float(np.linalg.eigvalsh(self.matrix).min())


@runtime_checkable
class CovarianceEstimator(Protocol):
    """Extension point for sample, shrinkage, factor, or custom risk models."""

    @property
    def estimator_name(self) -> str: ...

    def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate: ...


@dataclass(frozen=True)
class SampleCovarianceEstimator:
    """Deterministic sample covariance with explicit missing-data and PSD policy."""

    minimum_observations: int = 20
    ridge: float = 1e-8
    ensure_positive_semidefinite: bool = True
    missing_data_policy: CovarianceMissingDataPolicy = (
        CovarianceMissingDataPolicy.PAIRWISE
    )

    def __post_init__(self) -> None:
        if self.minimum_observations < 2:
            raise ValueError("minimum_observations must be at least two")
        if self.ridge < 0 or not np.isfinite(self.ridge):
            raise ValueError("ridge must be finite and non-negative")
        object.__setattr__(
            self,
            "missing_data_policy",
            CovarianceMissingDataPolicy(self.missing_data_policy),
        )

    @property
    def estimator_name(self) -> str:
        return "sample_covariance"

    def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
        if not isinstance(returns, pd.DataFrame) or returns.empty:
            raise ValueError("returns must be a non-empty pandas DataFrame")
        if not returns.columns.is_unique:
            raise ValueError("return columns must contain unique instrument IDs")
        numeric = returns.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        counts = numeric.notna().sum()
        insufficient = counts[counts < self.minimum_observations]
        if not insufficient.empty:
            raise ValueError(
                "insufficient return history for instruments: "
                + ", ".join(map(str, insufficient.index.tolist()))
            )
        if self.missing_data_policy is CovarianceMissingDataPolicy.COMPLETE_CASE:
            estimation_sample = numeric.dropna(axis=0, how="any")
            if len(estimation_sample) < self.minimum_observations:
                raise ValueError(
                    "complete-case return sample has insufficient observations"
                )
            covariance_frame = estimation_sample.cov()
        else:
            estimation_sample = numeric
            covariance_frame = numeric.cov(
                min_periods=self.minimum_observations
            )
        covariance = covariance_frame.reindex(
            index=numeric.columns,
            columns=numeric.columns,
        ).to_numpy(dtype=float)
        if not np.isfinite(covariance).all():
            raise ValueError(
                "return histories do not have enough overlapping observations"
            )
        covariance = (covariance + covariance.T) / 2.0
        raw_minimum_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
        diagonal_adjustment = float(self.ridge)
        if self.ensure_positive_semidefinite:
            diagonal_adjustment += max(0.0, -raw_minimum_eigenvalue)
        covariance = covariance + np.eye(len(covariance)) * diagonal_adjustment
        return CovarianceEstimate(
            matrix=covariance,
            instrument_ids=tuple(numeric.columns),
            observations=int(estimation_sample.dropna(how="all").shape[0]),
            estimator_name=self.estimator_name,
            metadata={
                "missing_data_policy": self.missing_data_policy.value,
                "minimum_instrument_observations": int(counts.min()),
                "complete_case_observations": int(numeric.dropna(how="any").shape[0]),
                "raw_minimum_eigenvalue": raw_minimum_eigenvalue,
                "diagonal_adjustment": diagonal_adjustment,
            },
        )


@dataclass(frozen=True)
class ShrinkageCovarianceEstimator:
    """Shrink sample covariance toward a deterministic structured target.

    The configured intensity is explicit and therefore becomes part of the
    estimator's cache identity. A value of zero keeps the sample covariance,
    while a value of one returns the selected target.
    """

    shrinkage: float = 0.1
    target: CovarianceShrinkageTarget = CovarianceShrinkageTarget.DIAGONAL
    minimum_observations: int = 20
    ridge: float = 1e-8
    ensure_positive_semidefinite: bool = True
    missing_data_policy: CovarianceMissingDataPolicy = (
        CovarianceMissingDataPolicy.PAIRWISE
    )

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.shrinkage)
            or self.shrinkage < 0
            or self.shrinkage > 1
        ):
            raise ValueError("shrinkage must be finite and between zero and one")
        if self.minimum_observations < 2:
            raise ValueError("minimum_observations must be at least two")
        if self.ridge < 0 or not np.isfinite(self.ridge):
            raise ValueError("ridge must be finite and non-negative")
        object.__setattr__(self, "target", CovarianceShrinkageTarget(self.target))
        object.__setattr__(
            self,
            "missing_data_policy",
            CovarianceMissingDataPolicy(self.missing_data_policy),
        )

    @property
    def estimator_name(self) -> str:
        return "shrinkage_covariance"

    def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
        sample = SampleCovarianceEstimator(
            minimum_observations=self.minimum_observations,
            ridge=0.0,
            ensure_positive_semidefinite=False,
            missing_data_policy=self.missing_data_policy,
        ).estimate(returns)
        sample_matrix = np.asarray(sample.matrix)
        diagonal = np.diag(sample_matrix)
        if self.target is CovarianceShrinkageTarget.DIAGONAL:
            target_matrix = np.diag(diagonal)
        else:
            average_variance = float(diagonal.mean())
            target_matrix = np.eye(len(diagonal)) * average_variance
        covariance = (
            (1.0 - self.shrinkage) * sample_matrix
            + self.shrinkage * target_matrix
        )
        covariance = (covariance + covariance.T) / 2.0
        raw_minimum_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
        diagonal_adjustment = float(self.ridge)
        if self.ensure_positive_semidefinite:
            diagonal_adjustment += max(0.0, -raw_minimum_eigenvalue)
        covariance = covariance + np.eye(len(covariance)) * diagonal_adjustment
        metadata = dict(sample.metadata)
        metadata.update(
            {
                "shrinkage": float(self.shrinkage),
                "shrinkage_target": self.target.value,
                "pre_adjustment_minimum_eigenvalue": raw_minimum_eigenvalue,
                "diagonal_adjustment": diagonal_adjustment,
            }
        )
        return CovarianceEstimate(
            matrix=covariance,
            instrument_ids=sample.instrument_ids,
            observations=sample.observations,
            estimator_name=self.estimator_name,
            metadata=metadata,
        )


@dataclass(frozen=True)
class FactorCovarianceEstimator:
    """Estimate a deterministic statistical factor covariance model.

    Principal covariance components represent common factor risk. The
    non-negative diagonal residual preserves each instrument's sample variance
    as specific risk. Complete observations are required so the factor
    decomposition is based on one coherent point-in-time sample.
    """

    factor_count: int = 1
    minimum_observations: int = 20
    ridge: float = 1e-8

    def __post_init__(self) -> None:
        if (
            isinstance(self.factor_count, bool)
            or not isinstance(self.factor_count, (int, np.integer))
            or self.factor_count <= 0
        ):
            raise ValueError("factor_count must be a positive integer")
        if self.minimum_observations < 2:
            raise ValueError("minimum_observations must be at least two")
        if self.ridge < 0 or not np.isfinite(self.ridge):
            raise ValueError("ridge must be finite and non-negative")
        object.__setattr__(self, "factor_count", int(self.factor_count))

    @property
    def estimator_name(self) -> str:
        return "factor_covariance"

    def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
        if not isinstance(returns, pd.DataFrame) or returns.empty:
            raise ValueError("returns must be a non-empty pandas DataFrame")
        if not returns.columns.is_unique:
            raise ValueError("return columns must contain unique instrument IDs")
        numeric = returns.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        counts = numeric.notna().sum()
        insufficient = counts[counts < self.minimum_observations]
        if not insufficient.empty:
            raise ValueError(
                "insufficient return history for instruments: "
                + ", ".join(map(str, insufficient.index.tolist()))
            )
        estimation_sample = numeric.dropna(axis=0, how="any")
        observations = len(estimation_sample)
        if observations < self.minimum_observations:
            raise ValueError(
                "complete-case return sample has insufficient observations"
            )
        maximum_factors = min(len(numeric.columns), observations - 1)
        if self.factor_count > maximum_factors:
            raise ValueError(
                "factor_count cannot exceed the number of instruments or "
                "complete observations minus one"
            )

        values = estimation_sample.to_numpy(dtype=float)
        centered = values - values.mean(axis=0, keepdims=True)
        sample_covariance = centered.T @ centered / float(observations - 1)
        sample_covariance = (sample_covariance + sample_covariance.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(sample_covariance)
        order = np.argsort(eigenvalues, kind="stable")[::-1]
        selected_indices = order[: self.factor_count]
        selected_eigenvalues = np.maximum(eigenvalues[selected_indices], 0.0)
        selected_vectors = eigenvectors[:, selected_indices]
        common_covariance = (
            selected_vectors * selected_eigenvalues
        ) @ selected_vectors.T
        specific_variances = np.maximum(
            np.diag(sample_covariance - common_covariance),
            0.0,
        )
        covariance = common_covariance + np.diag(specific_variances)
        covariance = (covariance + covariance.T) / 2.0
        raw_minimum_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
        diagonal_adjustment = float(self.ridge) + max(
            0.0,
            -raw_minimum_eigenvalue,
        )
        covariance = covariance + np.eye(len(covariance)) * diagonal_adjustment
        total_variance = float(np.maximum(eigenvalues, 0.0).sum())
        explained_variance_ratio = (
            float(selected_eigenvalues.sum() / total_variance)
            if total_variance > 0
            else 0.0
        )
        return CovarianceEstimate(
            matrix=covariance,
            instrument_ids=tuple(numeric.columns),
            observations=observations,
            estimator_name=self.estimator_name,
            metadata={
                "factor_count": self.factor_count,
                "factor_eigenvalues": tuple(
                    float(value) for value in selected_eigenvalues
                ),
                "explained_variance_ratio": explained_variance_ratio,
                "minimum_instrument_observations": int(counts.min()),
                "complete_case_observations": observations,
                "minimum_specific_variance": float(specific_variances.min()),
                "maximum_specific_variance": float(specific_variances.max()),
                "pre_adjustment_minimum_eigenvalue": raw_minimum_eigenvalue,
                "diagonal_adjustment": diagonal_adjustment,
            },
        )


def estimate_covariance_for_window(
    estimator: CovarianceEstimator,
    returns: pd.DataFrame,
    window: ReturnWindowSpec,
    reference_date: object,
) -> tuple[ResolvedReturnWindow, CovarianceEstimate]:
    """Select a point-in-time rolling sample and estimate its covariance."""

    if not isinstance(estimator, CovarianceEstimator):
        raise TypeError("estimator must implement CovarianceEstimator")
    if not isinstance(returns.index, pd.DatetimeIndex):
        try:
            dates = pd.DatetimeIndex(returns.index)
        except Exception as exc:
            raise TypeError("returns index must be convertible to business dates") from exc
        returns = returns.copy()
        returns.index = dates
    resolved = window.resolve(returns.index, reference_date)
    selected = returns.loc[list(resolved.business_dates)]
    estimate = estimator.estimate(selected)
    metadata = dict(estimate.metadata)
    metadata.update(
        {
            "window_kind": window.kind.value,
            "window_lookback": window.lookback,
            "window_start_date": resolved.start_date.isoformat(),
            "window_end_date": resolved.end_date.isoformat(),
        }
    )
    enriched = CovarianceEstimate(
        matrix=estimate.matrix,
        instrument_ids=estimate.instrument_ids,
        observations=estimate.observations,
        estimator_name=estimate.estimator_name,
        metadata=metadata,
    )
    return resolved, enriched


__all__ = [
    "CovarianceEstimate",
    "CovarianceEstimator",
    "CovarianceMissingDataPolicy",
    "CovarianceShrinkageTarget",
    "FactorCovarianceEstimator",
    "ResolvedReturnWindow",
    "ReturnWindowKind",
    "ReturnWindowSpec",
    "SampleCovarianceEstimator",
    "ShrinkageCovarianceEstimator",
    "estimate_covariance_for_window",
]
