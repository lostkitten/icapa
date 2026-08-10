"""Result assembly, slicing, checkpoint, and cache-payload helpers."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import json
from typing import Any

import numpy as np
import pandas as pd

from .config import SimulationParams
from .models import IndexSimulationResult, SimulationCheckpoint


def empty_asset_return_frame(columns) -> pd.DataFrame:
    """Return an empty fixed-schema asset-return table."""

    value_columns = [
        column for column in columns if column not in {"instrument_id", "business_date"}
    ]
    index = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex([], name="business_date"),
            pd.Index([], name="instrument_id"),
        ],
        names=["business_date", "instrument_id"],
    )
    return pd.DataFrame(columns=value_columns, index=index)


def latest_observations_before(
    market_data: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.DataFrame | None:
    """Return the latest cross-section strictly before a date."""

    prior = market_data.loc[market_data["business_date"] < date]
    if prior.empty:
        return None
    previous_date = prior["business_date"].max()
    return prior.loc[prior["business_date"] == previous_date].set_index(
        "instrument_id",
        verify_integrity=True,
    )


def simulation_params_payload(params: SimulationParams) -> dict[str, Any]:
    """Return the canonical calculation configuration payload."""

    return {
        item.name: configuration_value(getattr(params, item.name))
        for item in fields(params)
    } | {
        "resolved_index_drift": configuration_value(params.resolved_index_drift),
        "resolved_benchmark_drift": configuration_value(
            params.resolved_benchmark_drift
        ),
    }


def legacy_simulation_params_payload(
    params: SimulationParams,
) -> dict[str, Any]:
    """Return the v1 parameter payload used for compatible cache reads."""

    return {
        "dividend_treatment": params.dividend_treatment.value,
        "weight_drift": params.weight_drift.value,
        "rebalance_timing": params.rebalance_timing.value,
        "base_value": params.base_value,
    }


def configuration_value(value: Any) -> Any:
    """Convert nested simulation configuration to deterministic primitives."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "configuration": {
                item.name: configuration_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, dict):
        return {
            str(key): configuration_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [configuration_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "configuration": {
                key: configuration_value(item)
                for key, item in sorted(attributes.items())
                if not key.startswith("_")
            },
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def checkpoint_frame(checkpoint: SimulationCheckpoint) -> pd.DataFrame:
    """Encode an extension checkpoint as a constituent-indexed table."""

    instruments = checkpoint.index_weights.index.union(
        checkpoint.benchmark_weights.index
    ).union(checkpoint.previous_observations.index)
    frame = pd.DataFrame(index=instruments)
    frame.index.name = "instrument_id"
    frame["index_weight"] = checkpoint.index_weights.reindex(
        instruments,
        fill_value=0.0,
    )
    frame["benchmark_weight"] = checkpoint.benchmark_weights.reindex(
        instruments,
        fill_value=0.0,
    )
    for column in checkpoint.previous_observations.columns:
        frame[f"previous_observation__{column}"] = checkpoint.previous_observations[
            column
        ].reindex(instruments)
    return frame


def decode_checkpoint(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> SimulationCheckpoint:
    """Decode a persisted constituent table and metadata into a checkpoint."""

    prefix = "previous_observation__"
    previous_columns = [column for column in frame.columns if column.startswith(prefix)]
    previous = frame[previous_columns].rename(
        columns=lambda column: column.removeprefix(prefix)
    )
    previous = previous.dropna(how="all")
    return SimulationCheckpoint(
        business_date=metadata["business_date"],
        index_weights=frame["index_weight"].rename("index_weight"),
        benchmark_weights=frame["benchmark_weight"].rename("benchmark_weight"),
        levels={
            str(key): float(value) for key, value in dict(metadata["levels"]).items()
        },
        previous_observations=previous,
    )


def checkpoint_with_levels(
    checkpoint: SimulationCheckpoint,
    levels: dict[str, float],
) -> SimulationCheckpoint:
    """Copy a checkpoint with replacement level state."""

    return SimulationCheckpoint(
        business_date=checkpoint.business_date,
        index_weights=checkpoint.index_weights,
        benchmark_weights=checkpoint.benchmark_weights,
        levels=levels,
        previous_observations=checkpoint.previous_observations,
    )


def rescale_simulation_levels(
    result: IndexSimulationResult,
    *,
    desired_opening: dict[str, float],
) -> IndexSimulationResult:
    """Scale a base-neutral segment onto an accumulated opening level."""

    if result.daily.empty:
        return result
    daily = result.daily.copy()
    scales: dict[str, float] = {}
    first = daily.iloc[0]
    for level_column in (
        column for column in daily.columns if column.endswith("_level")
    ):
        return_column = level_column.removesuffix("_level") + "_return"
        if return_column not in daily:
            raise ValueError(f"simulation segment is missing {return_column!r}")
        first_factor = 1.0 + float(first[return_column])
        if not np.isfinite(first_factor) or first_factor <= 0.0:
            raise ValueError(f"simulation segment has an invalid {return_column!r}")
        local_opening = float(first[level_column]) / first_factor
        target_opening = float(desired_opening.get(level_column, local_opening))
        if (
            not np.isfinite(local_opening)
            or local_opening <= 0.0
            or not np.isfinite(target_opening)
            or target_opening <= 0.0
        ):
            raise ValueError(
                f"simulation segment has an invalid {level_column!r} opening"
            )
        scale = target_opening / local_opening
        daily.loc[:, level_column] = daily[level_column] * scale
        scales[level_column] = scale
    checkpoint = result.checkpoint
    if checkpoint is not None:
        checkpoint = checkpoint_with_levels(
            checkpoint,
            {
                name: value * scales.get(name, 1.0)
                for name, value in checkpoint.levels.items()
            },
        )
    return IndexSimulationResult(
        daily=daily,
        holdings=result.holdings,
        rebalances=result.rebalances,
        asset_returns=result.asset_returns,
        metadata=result.metadata,
        checkpoint=checkpoint,
        weight_snapshots=result.weight_snapshots,
    )


def checkpoint_from_result(
    result: IndexSimulationResult,
    business_date: pd.Timestamp,
) -> SimulationCheckpoint | None:
    """Recover continuation state from a materialized simulation date."""

    target_date = pd.Timestamp(business_date).normalize()
    if result.checkpoint is not None and result.checkpoint.business_date == target_date:
        return result.checkpoint
    if result.holdings.empty or result.asset_returns.empty:
        return None
    try:
        holdings = result.holdings.xs(
            target_date,
            level="business_date",
        )
        observations = result.asset_returns.xs(
            target_date,
            level="business_date",
        )
        daily = result.daily.loc[target_date]
    except KeyError:
        return None
    level_columns = [
        column for column in result.daily.columns if column.endswith("_level")
    ]
    if not level_columns:
        return None
    return SimulationCheckpoint(
        business_date=target_date,
        index_weights=holdings["index_closing_weight"].rename("index_weight"),
        benchmark_weights=holdings["benchmark_closing_weight"].rename(
            "benchmark_weight"
        ),
        levels={column: float(daily[column]) for column in level_columns},
        previous_observations=observations,
    )


def concat_frames(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Concatenate non-overlapping simulation frames."""

    if first.empty:
        return second.copy()
    if second.empty:
        return first.copy()
    combined = pd.concat([first, second], axis=0)
    if combined.index.has_duplicates:
        raise ValueError("simulation segments contain overlapping rows")
    return combined.sort_index()


def slice_frame_from_date(
    frame: pd.DataFrame,
    start_date: pd.Timestamp,
    *,
    level_name: str,
) -> pd.DataFrame:
    """Slice a materialized simulation table without changing its schema."""

    if frame.empty:
        return frame.copy()
    if level_name not in frame.index.names:
        raise ValueError(f"simulation table index is missing {level_name!r}")
    dates = pd.to_datetime(frame.index.get_level_values(level_name))
    return frame.loc[dates >= pd.Timestamp(start_date).normalize()].copy()


def combine_simulation_results(
    results: list[IndexSimulationResult],
    *,
    cache_key: str,
) -> IndexSimulationResult:
    """Assemble non-overlapping simulation partitions in date order."""

    if not results:
        raise ValueError("at least one simulation partition is required")
    rebalances = concat_rebalance_sequence([result.rebalances for result in results])
    if not rebalances.empty and rebalances["applied_business_date"].duplicated().any():
        raise ValueError(
            "simulation partitions apply more than one rebalance on the same "
            "business date"
        )
    final = results[-1]
    return IndexSimulationResult(
        daily=concat_frame_sequence([result.daily for result in results]),
        holdings=concat_frame_sequence([result.holdings for result in results]),
        rebalances=rebalances,
        asset_returns=concat_frame_sequence(
            [result.asset_returns for result in results]
        ),
        metadata={**final.metadata, "cache_key": cache_key},
        checkpoint=final.checkpoint,
        weight_snapshots=concat_frame_sequence(
            [result.weight_snapshots for result in results]
        ),
    )


def merged_source_data_records(
    results: list[IndexSimulationResult],
) -> list[dict[str, Any]]:
    """Deduplicate source-lineage records across simulation partitions."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        for raw in result.metadata.get("source_data_records", ()) or ():
            record = dict(raw)
            marker = json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            if marker in seen:
                continue
            seen.add(marker)
            records.append(record)
    return records


def concat_frame_sequence(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a sequence of non-overlapping indexed tables."""

    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return frames[-1].copy()
    combined = pd.concat(nonempty, axis=0)
    if combined.index.has_duplicates:
        raise ValueError("simulation partitions contain overlapping rows")
    return combined.sort_index()


def concat_rebalances(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate two rebalance event tables."""

    if first.empty:
        return second.copy().reset_index(drop=True)
    if second.empty:
        return first.copy().reset_index(drop=True)
    return pd.concat([first, second], ignore_index=True)


def concat_rebalance_sequence(
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate rebalance event tables in sequence."""

    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return frames[-1].copy().reset_index(drop=True)
    return pd.concat(nonempty, ignore_index=True)


__all__ = [
    "checkpoint_frame",
    "checkpoint_from_result",
    "checkpoint_with_levels",
    "combine_simulation_results",
    "concat_frame_sequence",
    "concat_frames",
    "concat_rebalance_sequence",
    "concat_rebalances",
    "configuration_value",
    "decode_checkpoint",
    "empty_asset_return_frame",
    "latest_observations_before",
    "legacy_simulation_params_payload",
    "merged_source_data_records",
    "rescale_simulation_levels",
    "simulation_params_payload",
    "slice_frame_from_date",
]
