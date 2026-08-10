"""Provider-neutral cross-sectional factor standardization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

import numpy as np
import pandas as pd


class FactorStandardizationVariant(StrEnum):
    """Select a factor-standardization formula without provider branding."""

    STANDARD = "standard"
    ALTERNATIVE = "alternative"


def factor_output_name(field: str, suffix: str = "_zscore") -> str:
    """Return the canonical output name for a standardized factor."""

    if not field:
        raise ValueError("factor field names cannot be empty")
    return f"{field}{suffix}"


def standardize_factors(
    frame: pd.DataFrame,
    factor_fields: Sequence[str],
    *,
    output_suffix: str = "_zscore",
    zscore_clip: float | None = 3.0,
    missing_zscore: float = 0.0,
) -> pd.DataFrame:
    """Calculate deterministic cross-sectional z-scores for explicit fields."""

    fields = tuple(factor_fields)
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("factor_fields must contain unique field names")
    if zscore_clip is not None and zscore_clip <= 0:
        raise ValueError("zscore_clip must be positive when supplied")

    result = pd.DataFrame(index=frame.index)
    for field in fields:
        if field not in frame:
            raise ValueError(f"factor field is missing: {field}")
        values = pd.to_numeric(frame[field], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        finite_count = int(values.notna().sum())
        standard_deviation = float(values.std(ddof=0))
        if (
            finite_count < 2
            or not np.isfinite(standard_deviation)
            or standard_deviation <= 0
        ):
            zscores = pd.Series(0.0, index=values.index)
        else:
            zscores = (
                values - float(values.mean())
            ) / standard_deviation
            if zscore_clip is not None:
                zscores = zscores.clip(-zscore_clip, zscore_clip)
            zscores = zscores.fillna(float(missing_zscore))
        result[factor_output_name(field, output_suffix)] = zscores.astype(float)
    return result


@dataclass
class StandardizeFactors:
    """Standardize explicit factor fields in a data context."""

    factor_fields: Sequence[str]
    output_suffix: str = "_zscore"
    zscore_clip: float | None = 3.0
    missing_zscore: float = 0.0
    standardization_variant: FactorStandardizationVariant = (
        FactorStandardizationVariant.STANDARD
    )

    @property
    def output_columns(self) -> list[str]:
        return [
            factor_output_name(field, self.output_suffix)
            for field in self.factor_fields
        ]

    def execute(self, data_context):
        variant = FactorStandardizationVariant(self.standardization_variant)
        if variant is not FactorStandardizationVariant.STANDARD:
            raise ValueError(
                "factor standardization currently has only the STANDARD variant"
            )
        frame = data_context.get_dataframe(
            list(self.factor_fields),
            include_excluded_instruments=False,
        )
        result = standardize_factors(
            frame,
            self.factor_fields,
            output_suffix=self.output_suffix,
            zscore_clip=self.zscore_clip,
            missing_zscore=self.missing_zscore,
        )
        data_context.set_dataframe(
            result,
            columns=self.output_columns,
        )
        return data_context


__all__ = [
    "FactorStandardizationVariant",
    "StandardizeFactors",
    "factor_output_name",
    "standardize_factors",
]
