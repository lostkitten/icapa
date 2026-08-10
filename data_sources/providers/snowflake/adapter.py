from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..sql_server import SQLQueryExecutor
from ..exceptions import DataSourceNotConfiguredError


@dataclass
class SnowflakePlaceholder:
    """Optional Snowflake hook with no account, role, warehouse, or default data."""

    name: str = "snowflake"
    account: str | None = None
    database: str | None = None
    schema: str | None = None
    warehouse: str | None = None
    role: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    executor: SQLQueryExecutor | None = None

    @property
    def configured(self) -> bool:
        return self.executor is not None

    def configure(self, executor: SQLQueryExecutor, **metadata: Any):
        self.executor = executor
        self.options.update(metadata)
        return self

    def query(self, sql: str, parameters=None):
        if self.executor is None:
            raise DataSourceNotConfiguredError(
                "Snowflake is an optional placeholder and has no configured "
                "connection or default dataset"
            )
        return self.executor(sql, parameters)
