"""Provider protocols, registry, and public integration adapters."""

from .factset import FactSet
from .exceptions import (
    DataCapabilityNotConfiguredError,
    DataSourceError,
    DataSourceNotConfiguredError,
)
from .file import FileProvider
from .interfaces import (
    CalendarProvider,
    MarketDataProvider,
    MembershipProvider,
    ReferenceDataProvider,
    SnapshotAwareProvider,
    ThirdPartyDataProvider,
    UniverseProvider,
)
from .registry import (
    ProviderRegistry,
    get_provider,
    register_provider,
    registry,
)
from .snowflake import SnowflakePlaceholder
from .sql_server import (
    SQLQueryExecutor,
    SQLServerPlaceholder,
)

__all__ = [
    "CalendarProvider",
    "DataCapabilityNotConfiguredError",
    "DataSourceError",
    "DataSourceNotConfiguredError",
    "FactSet",
    "FileProvider",
    "MarketDataProvider",
    "MembershipProvider",
    "ProviderRegistry",
    "ReferenceDataProvider",
    "SQLQueryExecutor",
    "SQLServerPlaceholder",
    "SnapshotAwareProvider",
    "SnowflakePlaceholder",
    "ThirdPartyDataProvider",
    "UniverseProvider",
    "get_provider",
    "register_provider",
    "registry",
]
