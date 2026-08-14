"""Generic quantile-selection portfolio engine.

Standardized (cross-sectionally z-scored) signal exposures are combined
into a composite selection score; the top score quantile is selected
within each selection scope and deterministically reweighted into a
fully invested portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np
import pandas as pd

from icapa.portfolio_construction.recipes.factors import (
    factor_output_name,
)


class SelectionCriterion(str, Enum):
    """Select how the top quantile is measured on the score ranking."""

    COUNT = "count"
    WEIGHT_COVERAGE = "weight_coverage"


class SelectionScope(str, Enum):
    """Select whether the quantile is taken over the universe or within groups."""

    UNIVERSE = "universe"
    WITHIN_GROUP = "within_group"


class SelectionWeighting(str, Enum):
    """Select how the chosen instruments are reweighted to full investment."""

    PROPORTIONAL = "proportional"
    EQUAL = "equal"


@dataclass
class QuantileSelectionEngine:
    """Select the top score quantile per selection scope and reweight it deterministically."""

    signal_weights: Mapping[str, float]
    selection_fraction: float = 0.4
    selection_criterion: SelectionCriterion = SelectionCriterion.COUNT
    selection_scope: SelectionScope = SelectionScope.UNIVERSE
    selection_weighting: SelectionWeighting = SelectionWeighting.PROPORTIONAL
    group_column: str = "industry"
    input_weight_column: str = "benchmark_weight"
    output_weight_column: str = "index_weight"
    score_column: str = "selection_score"
    selected_column: str = "selected"
    signal_suffix: str = "_zscore"
    liquidity_field: str | None = None
    minimum_liquidity: float | None = None

    def execute(self, data_context):
        if not self.signal_weights:
            raise ValueError(
                "signal_weights must map at least one signal field to a combination weight"
            )
        if not 0 < self.selection_fraction <= 1:
            raise ValueError("selection_fraction must be in the interval (0, 1]")
        criterion = SelectionCriterion(self.selection_criterion)
        scope = SelectionScope(self.selection_scope)
        weighting = SelectionWeighting(self.selection_weighting)
        if (self.liquidity_field is None) != (self.minimum_liquidity is None):
            raise ValueError(
                "liquidity_field and minimum_liquidity must be supplied together"
            )
        signal_columns = [
            factor_output_name(field, self.signal_suffix)
            for field in self.signal_weights
        ]
        columns = [self.input_weight_column, *signal_columns]
        if scope is SelectionScope.WITHIN_GROUP:
            columns.append(self.group_column)
        if self.liquidity_field:
            columns.append(self.liquidity_field)
        columns = list(dict.fromkeys(columns))
        frame = data_context.get_dataframe(
            columns, include_excluded_instruments=False
        )
        if frame.empty or not frame.index.is_unique:
            raise ValueError("the investable universe must be non-empty with a unique index")
        benchmark = frame[self.input_weight_column].to_numpy(dtype=float)
        if not np.isfinite(benchmark).all() or np.any(benchmark < 0) or benchmark.sum() <= 0:
            raise ValueError("benchmark weights must be finite, non-negative, and non-zero")
        frame = frame.copy()
        frame[self.input_weight_column] = benchmark / benchmark.sum()

        score = np.zeros(len(frame), dtype=float)
        for (field, coefficient), column in zip(
            self.signal_weights.items(), signal_columns
        ):
            if not np.isfinite(float(coefficient)):
                raise ValueError(f"signal weight must be finite: {field}")
            values = frame[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"standardized signal contains non-finite values: {field}"
                )
            score += float(coefficient) * values
        frame[self.score_column] = score

        eligible = pd.Series(True, index=frame.index)
        if self.liquidity_field:
            liquidity = pd.to_numeric(frame[self.liquidity_field], errors="coerce")
            eligible &= liquidity.ge(float(self.minimum_liquidity)).fillna(False)
        selected = pd.Series(False, index=frame.index)
        weights = pd.Series(0.0, index=frame.index)

        if scope is SelectionScope.UNIVERSE:
            selected_index = self._select(frame.loc[eligible], criterion)
            if len(selected_index) == 0:
                raise ValueError("no instruments passed the eligibility filters")
            selected.loc[selected_index] = True
            weights.loc[selected_index] = self._selection_weights(
                weighting,
                frame.loc[selected_index, self.input_weight_column],
                1.0,
            )
        else:
            grouped = frame.groupby(self.group_column, sort=True, dropna=False)
            for _, group in grouped:
                group_total = float(group[self.input_weight_column].sum())
                candidates = group.loc[eligible.loc[group.index]]
                if candidates.empty and group_total > 0:
                    raise ValueError(
                        "a positive-weight selection group has no eligible instruments"
                    )
                selected_index = self._select(candidates, criterion)
                if len(selected_index) == 0:
                    continue
                selected.loc[selected_index] = True
                weights.loc[selected_index] = self._selection_weights(
                    weighting,
                    frame.loc[selected_index, self.input_weight_column],
                    group_total,
                )

        result = frame.iloc[:, 0:0].copy()
        result[self.score_column] = frame[self.score_column]
        result[self.selected_column] = selected.astype(bool)
        result[self.output_weight_column] = weights.astype(float)
        if not np.isclose(result[self.output_weight_column].sum(), 1.0, atol=1e-10):
            raise ValueError(
                "quantile selection could not construct a fully invested portfolio"
            )
        data_context.set_dataframe(
            result,
            columns=[self.score_column, self.selected_column, self.output_weight_column],
        )
        return data_context

    def _select(
        self, candidates: pd.DataFrame, criterion: SelectionCriterion
    ) -> pd.Index:
        if candidates.empty:
            return candidates.index
        ranked = candidates.assign(_tie_break=candidates.index.map(str)).sort_values(
            [self.score_column, "_tie_break"],
            ascending=[False, True],
            kind="mergesort",
        )
        if criterion is SelectionCriterion.COUNT:
            count = max(1, int(np.ceil(self.selection_fraction * len(ranked))))
        else:
            target_weight = self.selection_fraction * float(
                ranked[self.input_weight_column].sum()
            )
            cumulative = ranked[self.input_weight_column].cumsum().to_numpy()
            count = min(
                len(ranked),
                int(np.searchsorted(cumulative, target_weight, side="left")) + 1,
            )
        return ranked.index[:count]

    def _selection_weights(
        self,
        weighting: SelectionWeighting,
        source: pd.Series,
        target_total: float,
    ) -> pd.Series:
        if weighting is SelectionWeighting.EQUAL:
            return pd.Series(target_total / len(source), index=source.index)
        return self._normalized_weights(source, target_total)

    @staticmethod
    def _normalized_weights(source: pd.Series, target_total: float) -> pd.Series:
        source_total = float(source.sum())
        if source_total > 0:
            return source * (target_total / source_total)
        return pd.Series(target_total / len(source), index=source.index)

    @staticmethod
    def _normalised_weights(
        source: pd.Series,
        target_total: float,
    ) -> pd.Series:
        """Retain ``_normalised_weights`` as a compatibility spelling."""

        return QuantileSelectionEngine._normalized_weights(source, target_total)


__all__ = [
    "QuantileSelectionEngine",
    "SelectionCriterion",
    "SelectionScope",
    "SelectionWeighting",
]
