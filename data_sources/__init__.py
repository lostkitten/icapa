"""Small, side-effect-free facade for canonical data contracts and providers."""

from .contracts import (
    DailyMarketColumns,
    DataSource,
    IdentifierType,
    ThirdPartyDataType,
    UniverseColumns,
)
from .provenance import ProvenanceRecorder
from .providers import (
    FactSet,
    FileProvider,
    SnapshotAwareProvider,
    SnowflakePlaceholder,
    get_provider,
    register_provider,
    registry,
)
from .universes import (
    UniverseMappingMatch,
    UniverseMappingNotConfiguredError,
    UniverseMappingRegistry,
    UniverseProfile,
)

__all__ = [
    "FactSet",
    "FileProvider",
    "DailyMarketColumns",
    "DataSource",
    "IdentifierType",
    "ProvenanceRecorder",
    "SnowflakePlaceholder",
    "SnapshotAwareProvider",
    "ThirdPartyDataType",
    "UniverseMappingMatch",
    "UniverseMappingNotConfiguredError",
    "UniverseMappingRegistry",
    "UniverseProfile",
    "UniverseColumns",
    "get_provider",
    "register_provider",
    "registry",
]
