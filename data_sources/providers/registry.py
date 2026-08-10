"""Runtime registry for explicitly selected provider implementations."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .exceptions import DataCapabilityNotConfiguredError


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}

    def register(self, name: str, provider: Any, *, replace: bool = False) -> Any:
        key = name.strip().lower()
        if not key:
            raise ValueError("provider name must not be empty")
        if key in self._providers and not replace:
            raise KeyError(f"provider already registered: {key}")
        self._providers[key] = provider
        return provider

    def unregister(self, name: str) -> None:
        self._providers.pop(name.strip().lower(), None)

    def get(self, name: str) -> Any:
        key = name.strip().lower()
        try:
            return self._providers[key]
        except KeyError as exc:
            raise DataCapabilityNotConfiguredError(f"provider is not registered: {key}") from exc

    def resolve(self, capability: str, provider_name: str | None = None) -> Any:
        """Resolve one capability without selecting a provider implicitly."""
        if not provider_name or not provider_name.strip():
            raise DataCapabilityNotConfiguredError(
                f"provider_name is required for capability {capability!r}"
            )
        provider = self.get(provider_name)
        if not callable(getattr(provider, capability, None)):
            raise DataCapabilityNotConfiguredError(
                f"provider {provider_name!r} does not implement {capability!r}"
            )
        return provider

    def items(self) -> Iterator[tuple[str, Any]]:
        return iter(self._providers.items())

    def clear(self) -> None:
        self._providers.clear()


registry = ProviderRegistry()


def get_provider(name: str):
    return registry.get(name)


def register_provider(name: str, provider: Any, *, replace: bool = False):
    return registry.register(name, provider, replace=replace)
