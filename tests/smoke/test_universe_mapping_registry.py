"""Focused tests for explicit universe configuration mappings."""

from __future__ import annotations

import pytest

from icapa.data_sources.universes import (
    UniverseMappingNotConfiguredError,
    UniverseMappingRegistry,
    UniverseProfile,
)


def _profile(label: str) -> UniverseProfile:
    return UniverseProfile(
        profile_name=f"{label}_profile",
        universe_id=f"{label}_UNIVERSE",
        universe_provider_name=f"{label}_universe_provider",
        calendar_id=f"{label}_calendar",
        calendar_provider_name=f"{label}_calendar_provider",
        market_data_provider_name=f"{label}_market_provider",
        base_currency="USD",
        fx_provider_name=f"{label}_fx_provider",
        tax_provider_name=f"{label}_tax_provider",
        dividend_provider_name=f"{label}_dividend_provider",
        universe_provider_parameters={"dataset": label},
        calendar_provider_parameters={"region": label},
        market_data_provider_parameters={"frequency": "daily"},
        fx_parameters={"base_currency": "USD"},
        tax_parameters={"treatment": "net"},
        dividend_parameters={"treatment": "gross"},
        simulation_parameters={
            "weight_drift": "price_return",
            "dividend_treatment": "standard",
        },
    )


def test_exact_mapping_wins_before_longest_prefix():
    registry = UniverseMappingRegistry()
    broad = _profile("broad")
    regional = _profile("regional")
    exact = _profile("exact")

    registry.register_prefix("EQ", broad)
    registry.register_prefix("EQ-US", regional)
    registry.register_exact("EQ-US-LARGE", exact)

    exact_match = registry.resolve_match("eq-us-large")
    assert exact_match.match_type == "exact"
    assert exact_match.profile is exact

    prefix_match = registry.resolve_match("EQ-US-SMALL")
    assert prefix_match.match_type == "prefix"
    assert prefix_match.pattern == "EQ-US"
    assert prefix_match.profile is regional
    assert registry.resolve("EQ-EU") is broad


def test_registry_has_no_default_and_replacement_is_explicit():
    registry = UniverseMappingRegistry()
    first = _profile("first")
    second = _profile("second")

    with pytest.raises(UniverseMappingNotConfiguredError):
        registry.resolve("UNCONFIGURED")

    registry.register_exact("DEMO", first)
    with pytest.raises(KeyError):
        registry.register_exact("DEMO", second)
    registry.register_exact("DEMO", second, replace=True)
    assert registry.resolve("DEMO") is second

    registry.clear()
    with pytest.raises(UniverseMappingNotConfiguredError):
        registry.resolve("DEMO")


def test_profile_requires_every_external_name_and_mapping_parameters():
    with pytest.raises(ValueError, match="calendar_id"):
        UniverseProfile(
            profile_name="incomplete_profile",
            universe_id="DEMO_UNIVERSE",
            universe_provider_name="universe_provider",
            calendar_id="",
            calendar_provider_name="calendar_provider",
            market_data_provider_name="market_provider",
            base_currency="USD",
        )

    profile = _profile("complete")
    with pytest.raises(TypeError):
        profile.fx_parameters["base_currency"] = "EUR"
    with pytest.raises(TypeError):
        profile.simulation_parameters["weight_drift"] = "none"


def test_component_providers_are_optional_but_non_empty_when_supplied():
    profile = UniverseProfile(
        profile_name="adjusted_returns",
        universe_id="ADJUSTED_UNIVERSE",
        universe_provider_name="universe_provider",
        calendar_id="primary_calendar",
        calendar_provider_name="calendar_provider",
        market_data_provider_name="adjusted_market_provider",
        base_currency="USD",
    )
    assert profile.fx_provider_name is None
    assert profile.tax_provider_name is None
    assert profile.dividend_provider_name is None

    with pytest.raises(ValueError, match="fx_provider_name"):
        UniverseProfile(
            profile_name="invalid_component",
            universe_id="DEMO_UNIVERSE",
            universe_provider_name="universe_provider",
            calendar_id="primary_calendar",
            calendar_provider_name="calendar_provider",
            market_data_provider_name="market_provider",
            base_currency="USD",
            fx_provider_name="",
        )
