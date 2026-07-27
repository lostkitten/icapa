from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from icapa.data_sources.exceptions import DataSourceNotConfiguredError


@dataclass
class ICEDataIndicesLibrary:
    """Placeholder for the ICE Data Indices Library integration.

    No SDK, credentials, fields, or datasets are assumed.  Inject an operation
    executor once the library and entitlement model are configured.
    """

    name: str = "ice_data_indices"
    executor: Callable[..., Any] | None = None

    @property
    def configured(self) -> bool:
        return self.executor is not None

    def configure(self, executor: Callable[..., Any]):
        self.executor = executor
        return self

    def execute(self, operation: str, **kwargs: Any):
        if self.executor is None:
            raise DataSourceNotConfiguredError(
                "ICE Data Indices Library is a placeholder; no SDK or connection is configured"
            )
        return self.executor(operation=operation, **kwargs)
