"""Explicit provider-neutral universe configuration."""

from .mapping import (
    UniverseMappingMatch,
    UniverseMappingNotConfiguredError,
    UniverseMappingRegistry,
    UniverseProfile,
)

__all__ = [
    "UniverseMappingMatch",
    "UniverseMappingNotConfiguredError",
    "UniverseMappingRegistry",
    "UniverseProfile",
]
