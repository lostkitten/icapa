"""Display-friendly recipe DAG tables."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def recipe_graph_frame(recipe: object) -> pd.DataFrame:
    """Return one row per recipe stage dependency."""

    stages = _recipe_stages(recipe)
    rows: list[dict[str, Any]] = []
    for order, stage in enumerate(stages):
        descriptor = getattr(stage, "descriptor", stage)
        stage_id = _first_value(descriptor, "stage_id", "name", default=f"stage_{order}")
        dependencies = _as_tuple(
            _first_value(
                descriptor,
                "dependencies",
                "depends_on",
                "input_stages",
                default=(),
            )
        )
        if not dependencies:
            rows.append(
                {
                    "stage_order": order,
                    "stage_id": stage_id,
                    "depends_on": None,
                    "deterministic": _first_value(
                        descriptor,
                        "deterministic",
                        default=True,
                    ),
                }
            )
            continue
        for dependency in dependencies:
            rows.append(
                {
                    "stage_order": order,
                    "stage_id": stage_id,
                    "depends_on": str(dependency),
                    "deterministic": _first_value(
                        descriptor,
                        "deterministic",
                        default=True,
                    ),
                }
            )
    return pd.DataFrame.from_records(
        rows,
        columns=["stage_order", "stage_id", "depends_on", "deterministic"],
    )


def _recipe_stages(recipe: object) -> tuple[object, ...]:
    for attribute in ("stages", "ordered_stages", "nodes"):
        value = getattr(recipe, attribute, None)
        if value is not None and not isinstance(value, (str, bytes)):
            return tuple(value)
    raise TypeError("recipe must expose stages, ordered_stages, or nodes")


def _first_value(target: object, *names: str, default: Any) -> Any:
    for name in names:
        if hasattr(target, name):
            return getattr(target, name)
    return default


def _as_tuple(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


__all__ = ["recipe_graph_frame"]
