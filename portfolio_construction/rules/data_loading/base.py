"""Minimal base class for canonical data-loading rules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataLoadingRule:
    description: str | None = None
    command: str = "DataLoadingRule"

    def __post_init__(self):
        self.keys = ["instrument_id"]

    def get_input_fact_names(self):
        return []

    def execute_within_pipeline(self, pipeline, data_context):
        return self.execute(data_context)

    def execute(self, data_context):
        raise NotImplementedError
