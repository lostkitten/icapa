"""Batch orchestration over the public research workspace."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BatchItem:
    """One labeled research request."""

    label: str
    spec: object

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("batch item label must be a non-empty string")


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    """Completed result or captured failure for one batch item."""

    label: str
    result: object | None = None
    error: BaseException | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def run_batch(
    workspace: object,
    items: Iterable[BatchItem],
    *,
    continue_on_error: bool = False,
) -> tuple[BatchOutcome, ...]:
    """Run independent research requests in deterministic input order."""

    run = getattr(workspace, "run", None)
    if not callable(run):
        raise TypeError("workspace must implement run(spec)")
    outcomes: list[BatchOutcome] = []
    labels: set[str] = set()
    for item in items:
        if not isinstance(item, BatchItem):
            raise TypeError("items must contain BatchItem instances")
        if item.label in labels:
            raise ValueError(f"duplicate batch item label: {item.label}")
        labels.add(item.label)
        try:
            outcomes.append(BatchOutcome(item.label, result=run(item.spec)))
        except BaseException as error:
            outcomes.append(BatchOutcome(item.label, error=error))
            if not continue_on_error:
                raise
    return tuple(outcomes)


__all__ = ["BatchItem", "BatchOutcome", "run_batch"]
