from copy import deepcopy
from dataclasses import dataclass, field
from typing import List

from icapa.portfolio_construction.rules.data_loading.base import DataLoadingRule


@dataclass
class AddFacts(DataLoadingRule):
    """Execute an explicit list of provider-neutral data-loading rules."""

    add_facts: List[DataLoadingRule] = field(default_factory=list)
    command: str = "AddFacts"

    def get_output_field_names(self):
        return sorted(
            {
                column
                for rule in self.add_facts
                for column in rule.get_output_field_names()
            }
        )

    def execute(self, data_context):
        result = deepcopy(data_context)
        for rule in self.add_facts:
            result = rule.execute(result)
        return result

    @staticmethod
    def get_required_fields(pipe):
        return sorted(
            {
                field_name
                for command in pipe.commands
                for field_name in command.get_input_field_names()
            }
        )

    @staticmethod
    def get_required_factors(pipe):
        """Return required fields through the compatibility method name."""

        return AddFacts.get_required_fields(pipe)
