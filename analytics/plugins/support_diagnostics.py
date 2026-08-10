"""Diagnostic-table normalization for analytics plugins."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
import pandas as pd
from ..contracts import AnalyticsValidationError
from .api import MissingAnalyticsInput
from .support_reviews import copy_alias, flat_input_frame, review_frames

def diagnostic_table(
    backtest_result: object,
    *,
    keys: Sequence[str],
) -> pd.DataFrame | None:
    wanted = set(keys)
    rows: list[dict[str, Any]] = []
    for effective_date, review, _ in review_frames(backtest_result):
        diagnostics = getattr(review, "diagnostics", None)
        if not isinstance(diagnostics, Mapping):
            continue
        collect_diagnostic_rows(
            diagnostics,
            wanted=wanted,
            effective_date=effective_date,
            rows=rows,
        )
    return pd.DataFrame(rows) if rows else None


def collect_diagnostic_rows(
    value: Mapping[str, Any],
    *,
    wanted: set[str],
    effective_date: pd.Timestamp,
    rows: list[dict[str, Any]],
) -> None:
    for raw_key, item in value.items():
        key = str(raw_key)
        if key in wanted:
            rows.extend(
                diagnostic_value_records(
                    item,
                    effective_date=effective_date,
                )
            )
            continue
        if isinstance(item, Mapping):
            collect_diagnostic_rows(
                item,
                wanted=wanted,
                effective_date=effective_date,
                rows=rows,
            )


def diagnostic_value_records(
    value: Any,
    *,
    effective_date: pd.Timestamp,
) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        records = value.to_dict(orient="records")
    elif isinstance(value, Mapping):
        scalar_keys = {
            "name",
            "field",
            "target_name",
            "constraint_name",
            "requested",
            "requested_value",
            "achieved",
            "achieved_value",
            "value",
            "lower",
            "lower_bound",
            "upper",
            "upper_bound",
            "violation",
            "status",
        }
        if scalar_keys.intersection(value):
            records = [dict(value)]
        else:
            records = []
            for name, item in value.items():
                if isinstance(item, Mapping):
                    records.append({"name": str(name), **dict(item)})
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        records = [dict(item) for item in value if isinstance(item, Mapping)]
    else:
        records = []
    return [
        {"effective_date": effective_date, **record}
        for record in records
    ]


def normalise_target_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    working = flat_input_frame(frame)
    working = copy_alias(
        working,
        "target_name",
        ("name", "field"),
        required=True,
    )
    working = copy_alias(
        working,
        "requested_value",
        ("requested", "target_value", "target"),
        required=True,
    )
    working = copy_alias(
        working,
        "achieved_value",
        ("achieved", "value"),
        required=True,
    )
    if "effective_date" not in working:
        raise AnalyticsValidationError(
            "target diagnostics require effective_date"
        )
    working["effective_date"] = pd.to_datetime(
        working["effective_date"],
        errors="raise",
    ).dt.normalize()
    if working["effective_date"].isna().any():
        raise AnalyticsValidationError(
            "target diagnostics contain a null effective_date"
        )
    working["target_name"] = working["target_name"].astype("string").str.strip()
    if working["target_name"].isna().any() or working["target_name"].eq("").any():
        raise AnalyticsValidationError(
            "target diagnostic names must not be empty"
        )
    for column in ("requested_value", "achieved_value"):
        working[column] = pd.to_numeric(working[column], errors="raise")
        if not np.isfinite(working[column].to_numpy(dtype=float)).all():
            raise AnalyticsValidationError(
                f"target diagnostics contain non-finite {column}"
            )
    if "direction" not in working:
        working["direction"] = "equal"
    working["direction"] = (
        working["direction"].fillna("equal").astype(str).str.lower()
    )
    allowed = {"equal", "at_least", "at_most"}
    invalid_directions = sorted(set(working["direction"]).difference(allowed))
    if invalid_directions:
        raise AnalyticsValidationError(
            f"target diagnostics contain invalid directions: {invalid_directions}"
        )
    if "tolerance" not in working:
        working["tolerance"] = np.nan
    working["tolerance"] = pd.to_numeric(
        working["tolerance"],
        errors="coerce",
    )
    if (working["tolerance"].dropna() < 0).any():
        raise AnalyticsValidationError(
            "target diagnostic tolerance must be non-negative"
        )
    for target, aliases in (
        ("lower_bound", ("lower",)),
        ("upper_bound", ("upper",)),
    ):
        working = copy_alias(
            working,
            target,
            aliases,
            required=False,
            default=np.nan,
        )
        working[target] = pd.to_numeric(working[target], errors="coerce")

    has_tolerance = working["tolerance"].notna()
    equal = working["direction"].eq("equal")
    at_least = working["direction"].eq("at_least")
    at_most = working["direction"].eq("at_most")
    lower_missing = working["lower_bound"].isna()
    upper_missing = working["upper_bound"].isna()
    working.loc[
        lower_missing & has_tolerance & (equal | at_least),
        "lower_bound",
    ] = (
        working["requested_value"] - working["tolerance"]
    )
    working.loc[
        upper_missing & has_tolerance & (equal | at_most),
        "upper_bound",
    ] = (
        working["requested_value"] + working["tolerance"]
    )
    working["deviation"] = (
        working["achieved_value"] - working["requested_value"]
    )
    working["absolute_deviation"] = working["deviation"].abs()
    assessed = (
        working["lower_bound"].notna()
        | working["upper_bound"].notna()
    )
    within = pd.Series(pd.NA, index=working.index, dtype="boolean")
    evaluated = pd.Series(True, index=working.index)
    evaluated &= (
        working["lower_bound"].isna()
        | (working["achieved_value"] >= working["lower_bound"])
    )
    evaluated &= (
        working["upper_bound"].isna()
        | (working["achieved_value"] <= working["upper_bound"])
    )
    within.loc[assessed] = evaluated.loc[assessed]
    working["within_bounds"] = within
    return working.loc[
        :,
        [
            "effective_date",
            "target_name",
            "direction",
            "requested_value",
            "achieved_value",
            "deviation",
            "absolute_deviation",
            "tolerance",
            "lower_bound",
            "upper_bound",
            "within_bounds",
        ],
    ].sort_values(
        ["effective_date", "target_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def normalise_constraint_diagnostics(
    frame: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    working = flat_input_frame(frame)
    working = copy_alias(
        working,
        "constraint_name",
        ("name",),
        required=True,
    )
    working = copy_alias(
        working,
        "achieved_value",
        ("value", "achieved"),
        required=True,
    )
    if "effective_date" not in working:
        raise AnalyticsValidationError(
            "constraint diagnostics require effective_date"
        )
    working["effective_date"] = pd.to_datetime(
        working["effective_date"],
        errors="raise",
    ).dt.normalize()
    if working["effective_date"].isna().any():
        raise AnalyticsValidationError(
            "constraint diagnostics contain a null effective_date"
        )
    working["constraint_name"] = (
        working["constraint_name"].astype("string").str.strip()
    )
    if (
        working["constraint_name"].isna().any()
        or working["constraint_name"].eq("").any()
    ):
        raise AnalyticsValidationError(
            "constraint diagnostic names must not be empty"
        )
    working["achieved_value"] = pd.to_numeric(
        working["achieved_value"],
        errors="raise",
    )
    for target, aliases in (
        ("lower_bound", ("lower",)),
        ("upper_bound", ("upper",)),
        ("lower_slack", ()),
        ("upper_slack", ()),
        ("violation", ("maximum_violation",)),
        ("dual_value", ()),
    ):
        working = copy_alias(
            working,
            target,
            aliases,
            required=False,
            default=np.nan,
        )
        working[target] = pd.to_numeric(working[target], errors="coerce")
    usable = (
        working["lower_bound"].notna()
        | working["upper_bound"].notna()
        | working["violation"].notna()
        | ("status" in working and working["status"].notna())
    )
    working = working.loc[usable].copy()
    if working.empty:
        raise MissingAnalyticsInput(
            "constraint diagnostics do not contain bounds, status, or violations"
        )
    missing_lower_slack = (
        working["lower_slack"].isna()
        & working["lower_bound"].notna()
    )
    missing_upper_slack = (
        working["upper_slack"].isna()
        & working["upper_bound"].notna()
    )
    working.loc[missing_lower_slack, "lower_slack"] = (
        working["achieved_value"] - working["lower_bound"]
    )
    working.loc[missing_upper_slack, "upper_slack"] = (
        working["upper_bound"] - working["achieved_value"]
    )
    calculated_violation = pd.concat(
        [
            (working["lower_bound"] - working["achieved_value"]).clip(
                lower=0
            ),
            (working["achieved_value"] - working["upper_bound"]).clip(
                lower=0
            ),
            pd.Series(0.0, index=working.index),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    working["violation"] = working["violation"].fillna(
        calculated_violation
    )
    if "status" not in working:
        working["status"] = ""
    status = working["status"].fillna("").astype(str).str.lower()
    working["violated"] = (
        status.isin({"violated", "invalid"})
        | (working["violation"] > tolerance)
    )
    finite_slacks = pd.concat(
        [
            working["lower_slack"].abs(),
            working["upper_slack"].abs(),
        ],
        axis=1,
    ).min(axis=1, skipna=True)
    working["binding"] = (
        status.eq("binding")
        | (
            ~working["violated"]
            & finite_slacks.notna()
            & (finite_slacks <= tolerance)
        )
    )
    working.loc[status.eq(""), "status"] = np.where(
        working.loc[status.eq(""), "violated"],
        "violated",
        np.where(
            working.loc[status.eq(""), "binding"],
            "binding",
            "satisfied",
        ),
    )
    return working.loc[
        :,
        [
            "effective_date",
            "constraint_name",
            "achieved_value",
            "lower_bound",
            "upper_bound",
            "lower_slack",
            "upper_slack",
            "violation",
            "status",
            "binding",
            "violated",
            "dual_value",
        ],
    ].sort_values(
        ["effective_date", "constraint_name"],
        kind="mergesort",
    ).reset_index(drop=True)
