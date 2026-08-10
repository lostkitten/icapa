"""Shared contracts for persisted review diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


DIAGNOSTIC_TAG = "__icapa_review_diagnostic_type__"
MAX_DIAGNOSTIC_DEPTH = 64
DIAGNOSTIC_ENUM_REGISTRY: dict[str, type[Enum]] = {}


class ReviewDiagnosticSerializationError(ValueError):
    """Raised when review diagnostics cannot be persisted deterministically."""


def diagnostic_enum_id(enum_type: type[Enum]) -> str:
    """Return the stable persistence identifier for a diagnostic Enum."""

    return f"{enum_type.__module__}:{enum_type.__qualname__}"


def register_review_diagnostic_enum(
    enum_type: type[Enum],
) -> type[Enum]:
    """Register one stable Enum type for safe diagnostic restoration."""

    if not isinstance(enum_type, type) or not issubclass(enum_type, Enum):
        raise TypeError("enum_type must be an Enum class")
    module_name = enum_type.__module__
    qualified_name = enum_type.__qualname__
    if (
        not module_name
        or module_name == "__main__"
        or not qualified_name
        or "<locals>" in qualified_name
    ):
        raise ValueError(
            "diagnostic Enum classes must have a stable module-qualified name"
        )
    enum_id = diagnostic_enum_id(enum_type)
    existing = DIAGNOSTIC_ENUM_REGISTRY.get(enum_id)
    if existing is not None and existing is not enum_type:
        raise ValueError(
            f"a different diagnostic Enum is already registered as {enum_id}"
        )
    DIAGNOSTIC_ENUM_REGISTRY[enum_id] = enum_type
    return enum_type


@dataclass(slots=True)
class DiagnosticTable:
    """One nested diagnostic table awaiting immutable persistence."""

    value: pd.DataFrame | pd.Series
    node: dict[str, Any]
    path: str


__all__ = [
    "DiagnosticTable",
    "ReviewDiagnosticSerializationError",
    "register_review_diagnostic_enum",
]
