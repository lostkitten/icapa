"""Scenario evaluation orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import pandas as pd

from .models import Scenario
from .transforms import ScenarioApplication, apply_scenario


ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class ScenarioEvaluation(Generic[ResultT]):
    """One transformed input and its evaluator result."""

    application: ScenarioApplication
    result: ResultT


def evaluate_scenarios(
    frame: pd.DataFrame,
    scenarios: Sequence[Scenario],
    evaluator: Callable[[pd.DataFrame], ResultT],
) -> tuple[ScenarioEvaluation[ResultT], ...]:
    """Apply and evaluate scenarios independently in their supplied order."""

    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    evaluations: list[ScenarioEvaluation[ResultT]] = []
    for scenario in scenarios:
        application = apply_scenario(frame, scenario)
        evaluations.append(
            ScenarioEvaluation(
                application=application,
                result=evaluator(application.frame.copy(deep=True)),
            )
        )
    return tuple(evaluations)


__all__ = ["ScenarioEvaluation", "evaluate_scenarios"]
