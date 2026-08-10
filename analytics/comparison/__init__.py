"""Baseline-to-candidate research comparison APIs."""
from .engine import (
    ComparisonEngine, ComparisonInput, ComparisonSpec, CompatibilityPolicy,
    DateAlignment, InstrumentAlignment, ResearchComparison, ReviewAlignment,
    compare_research_results,
)
__all__ = [
    "ComparisonEngine", "ComparisonInput", "ComparisonSpec",
    "CompatibilityPolicy", "DateAlignment", "InstrumentAlignment",
    "ResearchComparison", "ReviewAlignment", "compare_research_results",
]
