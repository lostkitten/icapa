"""Rebalance arithmetic and tabular materialization helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .enums import RebalancePhase


HOLDING_COLUMNS = (
    "index_opening_weight",
    "index_closing_weight",
    "benchmark_opening_weight",
    "benchmark_closing_weight",
)

REBALANCE_COLUMNS = (
    "scheduled_effective_date",
    "applied_business_date",
    "reference_date",
    "one_way_turnover",
    "benchmark_one_way_turnover",
    "index_turnover",
    "benchmark_turnover",
    "instrument_count",
    "review_source",
)

REBALANCE_WEIGHT_SNAPSHOT_INDEX = (
    "applied_business_date",
    "snapshot",
    "instrument_id",
)

REBALANCE_WEIGHT_SNAPSHOT_COLUMNS = (
    "scheduled_effective_date",
    "reference_date",
    "rebalance_phase",
    "index_weight",
    "benchmark_weight",
)


def one_way_turnover(previous: pd.Series, target: pd.Series) -> float:
    """Calculate one-way turnover between two constituent-weight vectors."""

    instruments = previous.index.union(target.index)
    aligned_previous = previous.reindex(instruments, fill_value=0.0)
    aligned_target = target.reindex(instruments, fill_value=0.0)
    return 0.5 * float((aligned_target - aligned_previous).abs().sum())


def append_holding_rows(
    rows: list[dict[str, Any]],
    *,
    business_date: pd.Timestamp,
    index_open: pd.Series,
    index_close: pd.Series,
    benchmark_open: pd.Series,
    benchmark_close: pd.Series,
) -> None:
    """Append constituent opening and closing weights for one business day."""

    all_ids = (
        index_open.index.union(index_close.index)
        .union(benchmark_open.index)
        .union(benchmark_close.index)
    )
    for instrument_id in all_ids:
        rows.append(
            {
                "business_date": business_date,
                "instrument_id": instrument_id,
                "index_opening_weight": float(index_open.get(instrument_id, 0.0)),
                "index_closing_weight": float(index_close.get(instrument_id, 0.0)),
                "benchmark_opening_weight": float(
                    benchmark_open.get(instrument_id, 0.0)
                ),
                "benchmark_closing_weight": float(
                    benchmark_close.get(instrument_id, 0.0)
                ),
            }
        )


def append_rebalance_weight_snapshot_rows(
    rows: list[dict[str, Any]],
    *,
    business_date: pd.Timestamp,
    application: dict[str, Any],
    rebalance_phase: RebalancePhase,
    pre_index: pd.Series | None,
    target_index: pd.Series,
    end_index: pd.Series,
    pre_benchmark: pd.Series | None,
    target_benchmark: pd.Series,
    end_benchmark: pd.Series,
) -> None:
    """Append pre-rebalance, target, and end-of-day weight snapshots."""

    all_ids = target_index.index.union(end_index.index)
    all_ids = all_ids.union(target_benchmark.index).union(end_benchmark.index)
    if pre_index is not None:
        all_ids = all_ids.union(pre_index.index)
    if pre_benchmark is not None:
        all_ids = all_ids.union(pre_benchmark.index)

    snapshots = (
        ("pre_rebalance", pre_index, pre_benchmark),
        ("target", target_index, target_benchmark),
        ("end_of_day", end_index, end_benchmark),
    )
    for snapshot, index_weights, benchmark_weights in snapshots:
        for instrument_id in all_ids:
            rows.append(
                {
                    "applied_business_date": business_date,
                    "snapshot": snapshot,
                    "instrument_id": instrument_id,
                    "scheduled_effective_date": application[
                        "scheduled_effective_date"
                    ],
                    "reference_date": application["reference_date"],
                    "rebalance_phase": rebalance_phase.value,
                    "index_weight": snapshot_weight(
                        index_weights,
                        instrument_id,
                    ),
                    "benchmark_weight": snapshot_weight(
                        benchmark_weights,
                        instrument_id,
                    ),
                }
            )


def snapshot_weight(
    weights: pd.Series | None,
    instrument_id: Any,
) -> float:
    """Return one constituent weight or NaN for a missing lifecycle state."""

    if weights is None:
        return float("nan")
    return float(weights.get(instrument_id, 0.0))


def holding_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the fixed-schema daily holdings table."""

    if rows:
        return pd.DataFrame.from_records(rows).set_index(
            ["business_date", "instrument_id"]
        )
    index = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex([], name="business_date"),
            pd.Index([], name="instrument_id"),
        ],
        names=["business_date", "instrument_id"],
    )
    return pd.DataFrame(columns=HOLDING_COLUMNS, index=index, dtype=float)


def rebalance_weight_snapshot_frame(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build the fixed-schema effective-date weight snapshot table."""

    record_columns = (
        *REBALANCE_WEIGHT_SNAPSHOT_INDEX,
        *REBALANCE_WEIGHT_SNAPSHOT_COLUMNS,
    )
    if rows:
        frame = pd.DataFrame.from_records(rows, columns=record_columns)
        frame["applied_business_date"] = pd.to_datetime(
            frame["applied_business_date"]
        )
        frame["scheduled_effective_date"] = pd.to_datetime(
            frame["scheduled_effective_date"]
        )
        frame["reference_date"] = pd.to_datetime(frame["reference_date"])
        return frame.set_index(
            list(REBALANCE_WEIGHT_SNAPSHOT_INDEX)
        ).sort_index()

    index = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex([], name="applied_business_date"),
            pd.Index([], name="snapshot", dtype=object),
            pd.Index([], name="instrument_id", dtype=object),
        ],
        names=list(REBALANCE_WEIGHT_SNAPSHOT_INDEX),
    )
    return pd.DataFrame(
        {
            "scheduled_effective_date": pd.Series(
                index=index,
                dtype="datetime64[ns]",
            ),
            "reference_date": pd.Series(index=index, dtype="datetime64[ns]"),
            "rebalance_phase": pd.Series(index=index, dtype=object),
            "index_weight": pd.Series(index=index, dtype=float),
            "benchmark_weight": pd.Series(index=index, dtype=float),
        },
        columns=REBALANCE_WEIGHT_SNAPSHOT_COLUMNS,
        index=index,
    )


def rebalance_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the fixed-schema rebalance event table."""

    return with_turnover_aliases(
        pd.DataFrame.from_records(rows, columns=REBALANCE_COLUMNS)
    )


def with_turnover_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose canonical and historical turnover columns without dropping either."""

    result = frame.copy(deep=True)
    for canonical, historical in (
        ("one_way_turnover", "index_turnover"),
        ("benchmark_one_way_turnover", "benchmark_turnover"),
    ):
        if canonical not in result and historical in result:
            result[canonical] = result[historical]
        if historical not in result and canonical in result:
            result[historical] = result[canonical]
    return result


__all__ = [
    "append_holding_rows",
    "append_rebalance_weight_snapshot_rows",
    "holding_frame",
    "one_way_turnover",
    "rebalance_frame",
    "rebalance_weight_snapshot_frame",
    "snapshot_weight",
    "with_turnover_aliases",
]
