from dataclasses import dataclass

from icapa.data_sources.sql_server import SqlServerPlaceholder


@dataclass
class FactSet(SqlServerPlaceholder):
    """FactSet SQL Server placeholder; schema and connection are not yet known."""

    name: str = "factset"
