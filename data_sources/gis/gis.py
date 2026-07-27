from dataclasses import dataclass

from icapa.data_sources.sql_server import SqlServerPlaceholder


@dataclass
class GIS(SqlServerPlaceholder):
    """GIS SQL Server placeholder; schema and connection are not yet known."""

    name: str = "gis"
