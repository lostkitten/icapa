class DataSourceError(RuntimeError):
    """Base error for data-source integrations."""


class DataSourceNotConfiguredError(DataSourceError):
    """Raised when a placeholder is used before its connection is configured."""


class DataCapabilityNotConfiguredError(DataSourceError):
    """Raised when no registered provider implements a requested capability."""
