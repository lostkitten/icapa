"""Provider-neutral scenario definitions for index research."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class ShockOperation(StrEnum):
    """Supported deterministic field transformations."""

    ADD = "add"
    MULTIPLY = "multiply"
    REPLACE = "replace"


@dataclass(frozen=True, slots=True)
class ScenarioShock:
    """Apply one field transformation to a selected set of instruments."""

    field: str
    operation: ShockOperation
    value: Any
    instrument_ids: tuple[str, ...] = ()
    where: Mapping[str, tuple[Any, ...]] = dataclass_field(default_factory=dict)
    allow_empty: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise ValueError("field must be a non-empty string")
        object.__setattr__(self, "field", self.field.strip())
        object.__setattr__(self, "operation", ShockOperation(self.operation))
        ids = tuple(str(value).strip() for value in self.instrument_ids)
        if any(not value for value in ids):
            raise ValueError("instrument_ids must not contain empty values")
        object.__setattr__(self, "instrument_ids", ids)
        selectors: dict[str, tuple[Any, ...]] = {}
        for column, values in self.where.items():
            if not isinstance(column, str) or not column.strip():
                raise ValueError("scenario selector fields must be non-empty strings")
            selected = tuple(values)
            if not selected:
                raise ValueError(
                    f"scenario selector {column!r} must contain at least one value"
                )
            selectors[column.strip()] = selected
        object.__setattr__(self, "where", MappingProxyType(selectors))


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named, ordered collection of deterministic shocks."""

    name: str
    shocks: tuple[ScenarioShock, ...]
    description: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scenario name must be a non-empty string")
        if not self.shocks:
            raise ValueError("scenario must contain at least one shock")
        if not all(isinstance(shock, ScenarioShock) for shock in self.shocks):
            raise TypeError("scenario shocks must be ScenarioShock instances")
        object.__setattr__(self, "name", self.name.strip())


__all__ = ["Scenario", "ScenarioShock", "ShockOperation"]
