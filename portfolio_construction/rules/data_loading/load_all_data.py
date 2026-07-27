from dataclasses import dataclass, field
from typing import List

from icapa.portfolio_construction.rules.data_loading.base import DataLoadingRule
from icapa.portfolio_construction.rules.data_loading.add_underlying_index import AddUnderlyingIndex


@dataclass
class LoadAllData(DataLoadingRule):
    """Minimal canonical loading pipeline.

    The universe rule is mandatory.  Additional facts, file imports, or market
    data rules must be supplied explicitly; no dataset is inferred.
    """

    universe: AddUnderlyingIndex | None = None
    commands: List[DataLoadingRule] = field(default_factory=list)
    command: str = "LoadAllData"

    def get_output_fact_names(self):
        rules = ([self.universe] if self.universe else []) + list(self.commands)
        return sorted({column for rule in rules for column in rule.get_output_fact_names()})

    def execute(self, data_context):
        if self.universe is None:
            raise ValueError("LoadAllData requires an explicit AddUnderlyingIndex rule")
        result = self.universe.execute(data_context)
        for rule in self.commands:
            result = rule.execute(result)
        return result
