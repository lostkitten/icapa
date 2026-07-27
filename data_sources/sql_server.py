"""Connection-neutral SQL Server placeholder.

Authentication, network, database, and schema settings are intentionally
unset. Inject an executor only after the integration has been configured.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .exceptions import DataSourceNotConfiguredError


QueryExecutor = Callable[[str, Mapping[str, Any] | None], pd.DataFrame]


@dataclass
class SqlServerPlaceholder:
    name: str
    server: str | None = None
    database: str | None = None
    connection_options: dict[str, Any] = field(default_factory=dict)
    executor: QueryExecutor | None = None

    @property
    def configured(self) -> bool:
        return self.executor is not None

    def configure(self, executor: QueryExecutor, **connection_metadata: Any):
        self.executor = executor
        self.connection_options.update(connection_metadata)
        return self

    def query(self, sql: str, parameters: Mapping[str, Any] | None = None) -> pd.DataFrame:
        if self.executor is None:
            raise DataSourceNotConfiguredError(
                f"{self.name} SQL Server connection is a placeholder; "
                "inject an executor after the connection strategy is configured"
            )
        return self.executor(sql, parameters)
