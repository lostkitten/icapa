"""Minimal base class for canonical data-loading rules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataLoadingRule:
    description: str | None = None
    command: str = "DataLoadingRule"

    def __post_init__(self):
        self.keys = ["instrument_id"]

    def get_input_field_names(self):
        """Return canonical input field names declared by this rule."""

        compatibility_method = getattr(type(self), "get_input_fact_names")
        if compatibility_method is not DataLoadingRule.get_input_fact_names:
            return compatibility_method(self)
        return []

    def get_input_fact_names(self):
        """Return input fields through the compatibility method name."""

        canonical_method = getattr(type(self), "get_input_field_names")
        if canonical_method is not DataLoadingRule.get_input_field_names:
            return canonical_method(self)
        return []

    def get_output_field_names(self):
        """Return canonical output field names declared by this rule."""

        compatibility_method = getattr(type(self), "get_output_fact_names")
        if compatibility_method is not DataLoadingRule.get_output_fact_names:
            return compatibility_method(self)
        return []

    def get_output_fact_names(self):
        """Return output fields through the compatibility method name."""

        canonical_method = getattr(type(self), "get_output_field_names")
        if canonical_method is not DataLoadingRule.get_output_field_names:
            return canonical_method(self)
        return []

    def execute_within_pipeline(self, pipeline, data_context):
        return self.execute(data_context)

    def execute(self, data_context):
        raise NotImplementedError
