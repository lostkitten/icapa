"""Explicit client-neutral configuration for underlying universes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


class UnderlyingMappingNotConfiguredError(KeyError):
    """Raised when no exact or prefix mapping exists for an underlying identifier."""


@dataclass(frozen=True)
class UnderlyingProfile:
    """All external names needed to load and simulate one underlying universe."""

    profile_name: str
    universe_id: str
    universe_provider_name: str
    calendar_id: str
    calendar_provider_name: str
    market_data_provider_name: str
    base_currency: str
    fx_provider_name: str | None = None
    tax_provider_name: str | None = None
    dividend_provider_name: str | None = None
    universe_provider_parameters: Mapping[str, Any] = field(default_factory=dict)
    calendar_provider_parameters: Mapping[str, Any] = field(default_factory=dict)
    market_data_provider_parameters: Mapping[str, Any] = field(default_factory=dict)
    fx_parameters: Mapping[str, Any] = field(default_factory=dict)
    tax_parameters: Mapping[str, Any] = field(default_factory=dict)
    dividend_parameters: Mapping[str, Any] = field(default_factory=dict)
    simulation_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required_names = (
            "profile_name",
            "universe_id",
            "universe_provider_name",
            "calendar_id",
            "calendar_provider_name",
            "market_data_provider_name",
            "base_currency",
        )
        for name in required_names:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        optional_provider_names = (
            "fx_provider_name",
            "tax_provider_name",
            "dividend_provider_name",
        )
        for name in optional_provider_names:
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be None or a non-empty string")

        parameter_names = (
            "universe_provider_parameters",
            "calendar_provider_parameters",
            "market_data_provider_parameters",
            "fx_parameters",
            "tax_parameters",
            "dividend_parameters",
            "simulation_parameters",
        )
        for name in parameter_names:
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, MappingProxyType(dict(value)))


@dataclass(frozen=True)
class UnderlyingMappingMatch:
    """A resolved profile together with the rule that matched it."""

    underlying_id: str
    match_type: str
    pattern: str
    profile: UnderlyingProfile


class UnderlyingMappingRegistry:
    """Resolve explicit exact mappings before longest-prefix mappings."""

    def __init__(self) -> None:
        self._exact: dict[str, UnderlyingProfile] = {}
        self._prefix: dict[str, UnderlyingProfile] = {}

    def register_exact(
        self,
        underlying_id: str,
        profile: UnderlyingProfile,
        *,
        replace: bool = False,
    ) -> UnderlyingProfile:
        """Register one exact identifier without installing any implicit defaults."""

        key = self._normalise_identifier(underlying_id)
        self._register(self._exact, key, profile, replace=replace, label="exact")
        return profile

    def register_prefix(
        self,
        prefix: str,
        profile: UnderlyingProfile,
        *,
        replace: bool = False,
    ) -> UnderlyingProfile:
        """Register one prefix; the longest matching prefix wins."""

        key = self._normalise_identifier(prefix)
        self._register(self._prefix, key, profile, replace=replace, label="prefix")
        return profile

    def resolve(self, underlying_id: str) -> UnderlyingProfile:
        """Return a configured profile, preferring exact then longest-prefix matches."""

        return self.resolve_match(underlying_id).profile

    def resolve_match(self, underlying_id: str) -> UnderlyingMappingMatch:
        """Return the profile and matching rule for diagnostics."""

        key = self._normalise_identifier(underlying_id)
        if key in self._exact:
            return UnderlyingMappingMatch(
                underlying_id=key,
                match_type="exact",
                pattern=key,
                profile=self._exact[key],
            )
        matches = [prefix for prefix in self._prefix if key.startswith(prefix)]
        if not matches:
            raise UnderlyingMappingNotConfiguredError(
                f"no underlying mapping is configured for {underlying_id!r}"
            )
        prefix = max(matches, key=len)
        return UnderlyingMappingMatch(
            underlying_id=key,
            match_type="prefix",
            pattern=prefix,
            profile=self._prefix[prefix],
        )

    def unregister_exact(self, underlying_id: str) -> None:
        """Remove an exact mapping if present."""

        self._exact.pop(self._normalise_identifier(underlying_id), None)

    def unregister_prefix(self, prefix: str) -> None:
        """Remove a prefix mapping if present."""

        self._prefix.pop(self._normalise_identifier(prefix), None)

    def clear(self) -> None:
        """Remove every explicit mapping."""

        self._exact.clear()
        self._prefix.clear()

    def items(self) -> Iterator[UnderlyingMappingMatch]:
        """Iterate over exact mappings followed by prefixes, in stable order."""

        for key in sorted(self._exact):
            yield UnderlyingMappingMatch(key, "exact", key, self._exact[key])
        for key in sorted(self._prefix):
            yield UnderlyingMappingMatch(key, "prefix", key, self._prefix[key])

    @staticmethod
    def _register(
        target: dict[str, UnderlyingProfile],
        key: str,
        profile: UnderlyingProfile,
        *,
        replace: bool,
        label: str,
    ) -> None:
        if not isinstance(profile, UnderlyingProfile):
            raise TypeError("profile must be an UnderlyingProfile")
        if key in target and not replace:
            raise KeyError(f"{label} underlying mapping is already registered: {key}")
        target[key] = profile

    @staticmethod
    def _normalise_identifier(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("underlying identifiers must be non-empty strings")
        normalised = value.strip().upper()
        if any(character.isspace() for character in normalised):
            raise ValueError("underlying identifiers must not contain whitespace")
        return normalised


__all__ = [
    "UnderlyingMappingMatch",
    "UnderlyingMappingNotConfiguredError",
    "UnderlyingMappingRegistry",
    "UnderlyingProfile",
]
