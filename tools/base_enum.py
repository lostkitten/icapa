"""Shared string-friendly enumeration helpers."""

from enum import Enum


class BaseEnum(Enum):
    """Enum with optional metadata and strict value parsing."""

    @classmethod
    def metadata(cls) -> dict:
        return {}

    @property
    def description(self):
        return self.metadata().get(self.value, {}).get("description")

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    @classmethod
    def from_str(cls, value: str):
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"invalid {cls.__name__} value: {value!r}") from exc
