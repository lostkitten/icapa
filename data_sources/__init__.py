"""Data-source entry points.

Bundled service providers are unconfigured placeholders. The local file
provider has no database, account, credential, or default dataset.
"""

from .factset import FactSet
from .file_based import FileProvider
from .gis import GIS
from .ice_data_indices import ICEDataIndicesLibrary
from .registry import get_provider, register_provider, registry
from .snowflake import SnowflakePlaceholder


def register_default_placeholders(*, replace: bool = False):
    placeholders = {
        "gis": GIS(),
        "factset": FactSet(),
        "ice_data_indices": ICEDataIndicesLibrary(),
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
    "GIS",
    "ICEDataIndicesLibrary",
    "SnowflakePlaceholder",
    "get_provider",
    "register_default_placeholders",
    "register_provider",
    "registry",
]
