"""Deterministic perturbations for methodology sensitivity research."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd


class NoiseDistribution(StrEnum):
    """Supported pseudo-random noise distributions."""

    NORMAL = "normal"
    UNIFORM = "uniform"


@dataclass(frozen=True, slots=True)
class NoiseSpec:
    """Configuration for reproducible field perturbation."""

    columns: tuple[str, ...]
    scale: float
    seed: int
    distribution: NoiseDistribution = NoiseDistribution.NORMAL
    relative: bool = False

    def __post_init__(self) -> None:
        if not self.columns or any(not str(column).strip() for column in self.columns):
            raise ValueError("columns must contain non-empty field names")
        if not np.isfinite(float(self.scale)) or float(self.scale) < 0:
            raise ValueError("scale must be finite and non-negative")
        object.__setattr__(self, "columns", tuple(map(str, self.columns)))
        object.__setattr__(self, "distribution", NoiseDistribution(self.distribution))


def add_noise(frame: pd.DataFrame, spec: NoiseSpec) -> pd.DataFrame:
    """Return a reproducibly perturbed copy of selected numeric fields."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not isinstance(spec, NoiseSpec):
        raise TypeError("spec must be a NoiseSpec")
    missing = set(spec.columns) - set(frame.columns)
    if missing:
        raise KeyError(f"noise fields are missing: {sorted(missing)}")
    result = frame.copy(deep=True)
    rng = np.random.default_rng(spec.seed)
    for column in spec.columns:
        values = pd.to_numeric(result[column], errors="raise").astype(float)
        if spec.distribution is NoiseDistribution.NORMAL:
            noise = rng.normal(0.0, spec.scale, size=len(values))
        else:
            noise = rng.uniform(-spec.scale, spec.scale, size=len(values))
        result[column] = values * (1.0 + noise) if spec.relative else values + noise
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"noise produced non-finite values in {column!r}")
    return result


def bootstrap_rows(
    frame: pd.DataFrame,
    *,
    seed: int,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Sample rows with replacement while preserving the original schema."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("cannot bootstrap an empty frame")
    size = len(frame) if sample_size is None else int(sample_size)
    if size <= 0:
        raise ValueError("sample_size must be positive")
    rng = np.random.default_rng(seed)
    positions = rng.integers(0, len(frame), size=size)
    result = frame.iloc[positions].copy()
    result.index = pd.RangeIndex(size)
    return result


__all__ = [
    "NoiseDistribution",
    "NoiseSpec",
    "add_noise",
    "bootstrap_rows",
]
