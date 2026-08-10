"""Sensitivity evaluation orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

import pandas as pd


ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class SensitivityEvaluation(Generic[ResultT]):
    """One named perturbation and its evaluated result."""

    name: str
    result: ResultT


def evaluate_perturbations(
    base_frame: pd.DataFrame,
    perturbations: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]],
    evaluator: Callable[[pd.DataFrame], ResultT],
) -> tuple[SensitivityEvaluation[ResultT], ...]:
    """Evaluate independent perturbations without mutating the base input."""

    if not isinstance(base_frame, pd.DataFrame):
        raise TypeError("base_frame must be a pandas DataFrame")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    results: list[SensitivityEvaluation[ResultT]] = []
    for name, perturbation in perturbations.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("perturbation names must be non-empty strings")
        if not callable(perturbation):
            raise TypeError(f"perturbation {name!r} must be callable")
        candidate = perturbation(base_frame.copy(deep=True))
        if not isinstance(candidate, pd.DataFrame):
            raise TypeError(f"perturbation {name!r} must return a DataFrame")
        results.append(
            SensitivityEvaluation(
                name=name,
                result=evaluator(candidate.copy(deep=True)),
            )
        )
    return tuple(results)


__all__ = ["SensitivityEvaluation", "evaluate_perturbations"]
