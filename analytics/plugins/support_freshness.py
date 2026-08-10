"""Point-in-time freshness normalization for analytics plugins."""
from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import pandas as pd
from ..contracts import AnalyticsValidationError
from .support_reviews import flat_input_frame, review_frames

CANONICAL_DATE_FIELDS = {"business_date", "effective_date", "reference_date"}

def normalise_freshness_input(frame: pd.DataFrame) -> pd.DataFrame:
    working = flat_input_frame(frame)
    required = {
        "effective_date",
        "reference_date",
        "source",
        "observation_date",
    }
    missing = sorted(required.difference(working.columns))
    if missing:
        raise AnalyticsValidationError(
            f"freshness_data is missing columns: {missing}"
        )
    for column in ("effective_date", "reference_date"):
        working[column] = pd.to_datetime(
            working[column],
            errors="raise",
        ).dt.normalize()
        if working[column].isna().any():
            raise AnalyticsValidationError(
                f"freshness_data contains a null {column}"
            )
    if (working["reference_date"] > working["effective_date"]).any():
        raise AnalyticsValidationError(
            "freshness_data reference_date must not be after effective_date"
        )
    original = working["observation_date"]
    working["observation_date"] = pd.to_datetime(
        original,
        errors="coerce",
    ).dt.normalize()
    invalid = original.notna() & working["observation_date"].isna()
    if invalid.any():
        raise AnalyticsValidationError(
            "freshness_data contains invalid observation_date values"
        )
    working["source"] = working["source"].astype(str)
    return working.loc[
        :,
        [
            "effective_date",
            "reference_date",
            "source",
            "observation_date",
        ],
    ]


def freshness_records_from_reviews(
    backtest_result: object,
    requested: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for effective_date, review, frame in review_frames(backtest_result):
        if requested:
            missing = sorted(set(requested).difference(frame.columns))
            if missing:
                raise AnalyticsValidationError(
                    f"point-in-time date fields are missing: {missing}"
                )
            date_fields = tuple(requested)
        else:
            date_fields = tuple(
                column
                for column in frame.columns
                if str(column).lower().endswith("_date")
                and str(column).lower() not in CANONICAL_DATE_FIELDS
            )
        reference_date = getattr(review, "reference_date", None)
        if reference_date is None:
            raise AnalyticsValidationError(
                "review context must define reference_date for freshness analytics"
            )
        cutoff = pd.Timestamp(reference_date).normalize()
        for field_name in date_fields:
            original = frame[field_name]
            observations = pd.to_datetime(original, errors="coerce")
            invalid = original.notna() & observations.isna()
            if invalid.any():
                raise AnalyticsValidationError(
                    f"point-in-time field contains invalid dates: {field_name}"
                )
            for observation in observations:
                rows.append(
                    {
                        "effective_date": effective_date,
                        "reference_date": cutoff,
                        "source": str(field_name),
                        "observation_date": (
                            pd.Timestamp(observation).normalize()
                            if not pd.isna(observation)
                            else pd.NaT
                        ),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "effective_date",
            "reference_date",
            "source",
            "observation_date",
        ],
    )
