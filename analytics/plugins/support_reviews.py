"""Review-table normalization for analytics plugins."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import pandas as pd
from ..contracts import AnalyticsValidationError
from .support_common import json_scalar

def review_frames(
    backtest_result: object,
) -> tuple[tuple[pd.Timestamp, object, pd.DataFrame], ...]:
    reviews = getattr(backtest_result, "reviews", None)
    if not isinstance(reviews, Mapping):
        raise AnalyticsValidationError(
            "backtest_result.reviews must be a mapping"
        )
    result: list[tuple[pd.Timestamp, object, pd.DataFrame]] = []
    for raw_date, review in sorted(
        reviews.items(),
        key=lambda item: pd.Timestamp(item[0]),
    ):
        effective_date = pd.Timestamp(raw_date).normalize()
        frame = getattr(review, "cons", None)
        if not isinstance(frame, pd.DataFrame):
            raise AnalyticsValidationError(
                "every review context must expose a pandas DataFrame as cons"
            )
        working = frame.copy(deep=True)
        if working.index.name != "instrument_id":
            if "instrument_id" not in working.columns:
                raise AnalyticsValidationError(
                    "review constituent data requires instrument_id"
                )
            working = working.set_index(
                "instrument_id",
                verify_integrity=True,
            )
        elif working.index.has_duplicates:
            raise AnalyticsValidationError(
                "review constituent data contains duplicate instrument_id values"
            )
        result.append((effective_date, review, working))
    return tuple(result)


def normalise_reason_input(frame: pd.DataFrame) -> pd.DataFrame:
    working = flat_input_frame(frame)
    required = {
        "effective_date",
        "instrument_id",
        "decision",
        "reason",
    }
    missing = sorted(required.difference(working.columns))
    if missing:
        raise AnalyticsValidationError(
            f"selection_reasons is missing columns: {missing}"
        )
    rows: list[dict[str, Any]] = []
    for record in working.to_dict(orient="records"):
        for reason in reason_values(record["reason"]):
            decision = str(record["decision"]).strip()
            if not decision:
                raise AnalyticsValidationError(
                    "selection reason decisions must not be empty"
                )
            effective_date = pd.Timestamp(record["effective_date"])
            if pd.isna(effective_date):
                raise AnalyticsValidationError(
                    "selection reasons contain a null effective_date"
                )
            rows.append(
                {
                    "effective_date": effective_date.normalize(),
                    "instrument_id": record["instrument_id"],
                    "decision": decision,
                    "reason": reason,
                    "source": str(
                        record.get("source", "explicit_input")
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "effective_date",
            "instrument_id",
            "decision",
            "reason",
            "source",
        ],
    )


def reason_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        text = str(
            {
                str(key): json_scalar(item)
                for key, item in sorted(
                    value.items(),
                    key=lambda item: str(item[0]),
                )
            }
        )
        return (text,)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        result: list[str] = []
        for item in value:
            result.extend(reason_values(item))
        return tuple(result)
    try:
        if bool(pd.isna(value)):
            return ()
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return (text,) if text else ()


def flat_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy(deep=True)
    if isinstance(working.index, pd.RangeIndex) and working.index.name is None:
        return working.reset_index(drop=True)
    return working.reset_index()


def copy_alias(
    frame: pd.DataFrame,
    target: str,
    aliases: Sequence[str],
    *,
    required: bool,
    default: Any = None,
) -> pd.DataFrame:
    result = frame.copy()
    candidates = (target, *aliases)
    available = [column for column in candidates if column in result]
    if not available and required:
        raise AnalyticsValidationError(
            f"analytics diagnostics require column {target!r}"
        )
    if target not in result:
        result[target] = default
    for alias in aliases:
        if alias in result:
            result[target] = result[target].where(
                result[target].notna(),
                result[alias],
            )
    return result


def review_weights(backtest_result: object) -> pd.Series:
    weights = getattr(backtest_result, "weights", None)
    if not isinstance(weights, pd.DataFrame) or "index_weight" not in weights:
        raise AnalyticsValidationError(
            "backtest_result.weights must contain index_weight"
        )
    frame = weights.copy(deep=True)
    if not isinstance(frame.index, pd.MultiIndex) or set(frame.index.names) != {
        "effective_date",
        "instrument_id",
    }:
        required = {"effective_date", "instrument_id", "index_weight"}
        if not required.issubset(frame.reset_index().columns):
            raise AnalyticsValidationError(
                "review weights require effective_date and instrument_id"
            )
        frame = frame.reset_index().set_index(
            ["effective_date", "instrument_id"],
            verify_integrity=True,
        )
    frame = frame.sort_index()
    values = pd.to_numeric(frame["index_weight"], errors="raise").astype(float)
    values.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(values.index.get_level_values("effective_date")).normalize(),
            values.index.get_level_values("instrument_id"),
        ],
        names=["effective_date", "instrument_id"],
    )
    return values.sort_index()
