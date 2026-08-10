"""Event-window analytics over index and benchmark returns."""

from .study import (
    EventStudyResult,
    EventStudySpec,
    calculate_event_non_event_returns,
    run_event_study,
)

__all__ = [
    "EventStudyResult",
    "EventStudySpec",
    "calculate_event_non_event_returns",
    "run_event_study",
]
