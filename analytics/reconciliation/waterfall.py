"""Field-level reconciliation across ordered calculation-stage tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class DataWaterfallResult:
    """Detailed consecutive-stage differences and aggregate diagnostics."""

    detail: pd.DataFrame
    summary: pd.DataFrame
    reconciled: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", self.detail.copy(deep=True))
        object.__setattr__(self, "summary", self.summary.copy(deep=True))


def compare_data_stages(
    stages: Mapping[str, pd.DataFrame],
    *,
    key_columns: Sequence[str],
    value_columns: Sequence[str] | None = None,
    tolerance: float = 1e-12,
) -> DataWaterfallResult:
    """Compare every consecutive pair in an ordered stage mapping."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    stage_items = tuple(stages.items())
    if len(stage_items) < 2:
        raise ValueError("stages must contain at least two ordered tables")
    keys = tuple(str(column) for column in key_columns)
    if not keys:
        raise ValueError("key_columns must not be empty")
    normalized = [
        (name, _validated_stage(name, frame, keys))
        for name, frame in stage_items
    ]
    selected_values = (
        tuple(str(column) for column in value_columns)
        if value_columns is not None
        else _common_numeric_columns(normalized, keys)
    )
    if not selected_values:
        raise ValueError("no value columns are available for reconciliation")
    for name, frame in normalized:
        missing = sorted(set(selected_values).difference(frame.columns))
        if missing:
            raise ValueError(f"stage {name!r} is missing value columns: {missing}")

    detail_frames: list[pd.DataFrame] = []
    for (previous_name, previous), (current_name, current) in zip(
        normalized,
        normalized[1:],
    ):
        detail_frames.append(
            _compare_pair(
                previous_name,
                previous,
                current_name,
                current,
                keys,
                selected_values,
                tolerance,
            )
        )
    detail = pd.concat(detail_frames, ignore_index=True)
    summary = (
        detail.groupby(
            ["previous_stage", "current_stage", "field"],
            sort=False,
            dropna=False,
        )
        .agg(
            added=("status", lambda values: int(values.eq("added").sum())),
            removed=("status", lambda values: int(values.eq("removed").sum())),
            changed=("status", lambda values: int(values.eq("changed").sum())),
            unchanged=(
                "status",
                lambda values: int(values.eq("unchanged").sum()),
            ),
            absolute_difference=("absolute_difference", "sum"),
            maximum_absolute_difference=("absolute_difference", "max"),
        )
        .reset_index()
    )
    reconciled = not detail["status"].isin({"added", "removed", "changed"}).any()
    return DataWaterfallResult(
        detail=detail,
        summary=summary,
        reconciled=bool(reconciled),
    )


def reconcile_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    value_columns: Sequence[str] | None = None,
    tolerance: float = 1e-12,
) -> DataWaterfallResult:
    """Reconcile two tables through the same stage-waterfall contract."""

    return compare_data_stages(
        {"baseline": baseline, "candidate": candidate},
        key_columns=key_columns,
        value_columns=value_columns,
        tolerance=tolerance,
    )


def _compare_pair(
    previous_name: str,
    previous: pd.DataFrame,
    current_name: str,
    current: pd.DataFrame,
    keys: tuple[str, ...],
    values: tuple[str, ...],
    tolerance: float,
) -> pd.DataFrame:
    merged = previous.loc[:, [*keys, *values]].merge(
        current.loc[:, [*keys, *values]],
        on=list(keys),
        how="outer",
        suffixes=("_previous", "_current"),
        indicator=True,
        validate="one_to_one",
        sort=True,
    )
    rows: list[pd.DataFrame] = []
    for field in values:
        previous_column = f"{field}_previous"
        current_column = f"{field}_current"
        left = pd.to_numeric(merged[previous_column], errors="coerce")
        right = pd.to_numeric(merged[current_column], errors="coerce")
        invalid_left = merged["_merge"].ne("right_only") & left.isna()
        invalid_right = merged["_merge"].ne("left_only") & right.isna()
        if invalid_left.any() or invalid_right.any():
            raise ValueError(
                f"field {field!r} must contain finite numeric values"
            )
        difference = right - left
        status = np.select(
            [
                merged["_merge"].eq("right_only"),
                merged["_merge"].eq("left_only"),
                difference.abs().gt(tolerance),
            ],
            ["added", "removed", "changed"],
            default="unchanged",
        )
        result = merged.loc[:, list(keys)].copy()
        result.insert(0, "current_stage", current_name)
        result.insert(0, "previous_stage", previous_name)
        result["field"] = field
        result["previous_value"] = left
        result["current_value"] = right
        result["difference"] = difference
        result["absolute_difference"] = difference.abs().fillna(0.0)
        result["status"] = status
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _validated_stage(
    name: str,
    frame: pd.DataFrame,
    keys: tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("stage names must be non-empty strings")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"stage {name!r} must be a pandas DataFrame")
    missing = sorted(set(keys).difference(frame.columns))
    if missing:
        raise ValueError(f"stage {name!r} is missing key columns: {missing}")
    if frame.duplicated(list(keys)).any():
        raise ValueError(f"stage {name!r} contains duplicate keys")
    return frame.copy(deep=True)


def _common_numeric_columns(
    stages: list[tuple[str, pd.DataFrame]],
    keys: tuple[str, ...],
) -> tuple[str, ...]:
    common = set(stages[0][1].columns).difference(keys)
    for _, frame in stages[1:]:
        common.intersection_update(frame.columns)
    return tuple(
        column
        for column in stages[0][1].columns
        if column in common
        and all(
            pd.api.types.is_numeric_dtype(frame[column])
            for _, frame in stages
        )
    )
