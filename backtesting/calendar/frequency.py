"""Explicit frequency helpers for constructing review schedules."""

from __future__ import annotations

from enum import StrEnum


class RebalanceFrequency(StrEnum):
    """Supported periodic anchors for explicitly generated review schedules."""

    CUSTOM = "custom"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"


__all__ = ["RebalanceFrequency"]
