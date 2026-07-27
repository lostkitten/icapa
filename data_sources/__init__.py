"""Public data-source entry points.

FactSet and Snowflake are unconfigured integration examples. The generic file
provider has no database, account, credential, or default dataset.
"""

from .factset import FactSet
from .file_provider import FileProvider
from .registry import get_provider, register_provider, registry
from .snowflake import SnowflakePlaceholder


def register_default_placeholders(*, replace: bool = False):
    placeholders = {
        "factset": FactSet(),
        "snowflake": SnowflakePlaceholder(),
        "file": FileProvider(),
    }
    for name, provider in placeholders.items():
        if replace:
            register_provider(name, provider, replace=True)
        else:
            try:
                register_provider(name, provider)
            except KeyError:
                pass
    return placeholders


register_default_placeholders()

__all__ = [
    "FactSet",
    "FileProvider",
    "SnowflakePlaceholder",
    "get_provider",
    "register_default_placeholders",
    "register_provider",
    "registry",
]
