"""Scenario transformations for canonical constituent tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .models import Scenario, ScenarioShock, ShockOperation


@dataclass(frozen=True, slots=True)
class ScenarioApplication:
    """Transformed data and an auditable summary of applied shocks."""

    scenario: Scenario
    frame: pd.DataFrame
    diagnostics: pd.DataFrame


def apply_scenario(frame: pd.DataFrame, scenario: Scenario) -> ScenarioApplication:
    """Apply a scenario without mutating the supplied constituent table."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    transformed = frame.copy(deep=True)
    records: list[dict[str, Any]] = []
    for order, shock in enumerate(scenario.shocks):
        selected = _selection_mask(transformed, shock)
        count = int(selected.sum())
        if count == 0 and not shock.allow_empty:
            raise ValueError(
                f"scenario {scenario.name!r} shock for {shock.field!r} "
                "selected no instruments"
            )
        before = transformed.loc[selected, shock.field].copy()
        transformed.loc[selected, shock.field] = _apply_values(before, shock)
        after = transformed.loc[selected, shock.field]
        records.append(
            {
                "scenario": scenario.name,
                "shock_order": order,
                "field": shock.field,
                "operation": shock.operation.value,
                "selected_count": count,
                "before_mean": _numeric_mean(before),
                "after_mean": _numeric_mean(after),
            }
        )
    diagnostics = pd.DataFrame.from_records(records)
    return ScenarioApplication(scenario, transformed, diagnostics)


def _selection_mask(frame: pd.DataFrame, shock: ScenarioShock) -> pd.Series:
    if shock.field not in frame.columns:
        raise KeyError(f"scenario field is missing: {shock.field}")
    selected = pd.Series(True, index=frame.index, dtype=bool)
    if shock.instrument_ids:
        if "instrument_id" in frame.columns:
            identifiers = frame["instrument_id"].astype(str)
        else:
            identifiers = frame.index.to_series().astype(str)
        selected &= identifiers.isin(shock.instrument_ids)
    for column, values in shock.where.items():
        if column not in frame.columns:
            raise KeyError(f"scenario selector field is missing: {column}")
        selected &= frame[column].isin(values)
    return selected


def _apply_values(values: pd.Series, shock: ScenarioShock) -> pd.Series:
    if shock.operation is ShockOperation.REPLACE:
        return pd.Series(shock.value, index=values.index)
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    scalar = float(shock.value)
    if not np.isfinite(scalar):
        raise ValueError("scenario shock value must be finite")
    if shock.operation is ShockOperation.ADD:
        result = numeric + scalar
    else:
        result = numeric * scalar
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("scenario shock produced non-finite values")
    return result


def _numeric_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if not numeric.notna().any():
        return None
    return float(numeric.mean())


__all__ = ["ScenarioApplication", "apply_scenario"]
