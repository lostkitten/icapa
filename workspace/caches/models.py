"""Researcher-selected cache behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


class CacheMode(StrEnum):
    """Control reusable artifacts without changing calculation identity."""

    OFF = "off"
    REUSE = "reuse"
    REFRESH = "refresh"
    READ_ONLY = "read_only"


class CacheStage(StrEnum):
    """Independently controllable stages in an index-research run."""

    SOURCE_DATA = "source_data"
    REVIEWS = "reviews"
    SIMULATION = "simulation"
    ANALYTICS = "analytics"


@dataclass(frozen=True)
class CacheOptions:
    """Global cache behavior with optional per-stage overrides."""

    mode: CacheMode | str = CacheMode.OFF
    stage_modes: Mapping[CacheStage | str, CacheMode | str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", CacheMode(self.mode))
        normalized = {
            CacheStage(stage): CacheMode(mode)
            for stage, mode in dict(self.stage_modes).items()
        }
        object.__setattr__(self, "stage_modes", MappingProxyType(normalized))

    def mode_for(self, stage: CacheStage | str) -> CacheMode:
        return self.stage_modes.get(CacheStage(stage), self.mode)

    @classmethod
    def off(cls) -> "CacheOptions":
        return cls(CacheMode.OFF)

    @classmethod
    def reuse(cls) -> "CacheOptions":
        return cls(CacheMode.REUSE)

    @classmethod
    def refresh(cls) -> "CacheOptions":
        return cls(CacheMode.REFRESH)

    @classmethod
    def read_only(cls) -> "CacheOptions":
        return cls(CacheMode.READ_ONLY)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "stage_modes": {
                stage.value: mode.value
                for stage, mode in sorted(
                    self.stage_modes.items(),
                    key=lambda item: item[0].value,
                )
            },
        }


__all__ = ["CacheMode", "CacheOptions", "CacheStage"]
