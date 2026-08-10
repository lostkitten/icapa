from dataclasses import dataclass

from ..sql_server import SQLServerPlaceholder


@dataclass
class FactSet(SQLServerPlaceholder):
    """FactSet SQL Server placeholder; schema and connection are not yet known."""

    name: str = "factset"
