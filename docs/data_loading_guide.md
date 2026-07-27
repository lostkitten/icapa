# ICAPA Data Loading Guide

## Purpose

ICAPA separates data access from weight production. Provider adapters obtain
data from an external system and translate it into canonical tables.
Data-loading rules place those tables in a `DataContext`; an external weight
producer consumes that context and writes `index_weight`.

```text
Provider -> canonical contract -> external weight producer -> cached target weights -> simulation -> analytics -> report
```

External weight producers must not contain SQL, credentials, physical table
names, provider field names, or implicit provider selection.

## Date model

ICAPA uses three dates, each with one meaning:

| Field | Meaning | Required rule |
| --- | --- | --- |
| `reference_date` | Cutoff date for information available to a review | Point-in-time inputs and historical observations used to calculate review weights must not be later than this date. |
| `effective_date` | First date on which the calculated weights apply | Must be on or after `reference_date`. |
| `business_date` | Observation date of a daily market-data row | Historical rows used by a weight producer must respect `reference_date`; realized rows used after `effective_date` for simulation may be later. |

There is no generic date field. An adapter must map each source date to the appropriate canonical date according to its business meaning.

The distinction between historical calculation data and realized simulation
data is intentional. An external weight producer may use observations through
`reference_date` to calculate weights, while the daily index simulator then
uses market observations on and after `effective_date` to measure how those
frozen weights perform. The later observations are simulation outputs, not
information that was available to the review.

## Canonical contracts

### Point-in-time universe

Every universe row represents one instrument for one review and must contain all of the following columns:

| Column | Meaning |
| --- | --- |
| `instrument_id` | Stable identifier used for all joins inside ICAPA |
| `name` | Display name of the instrument |
| `country` | Canonical country classification |
| `industry` | Canonical industry classification |
| `shares` | Shares used in the investable market-cap calculation |
| `free_float` | Investable proportion of shares, expressed as a decimal |
| `price` | Reference price in the instrument currency |
| `currency` | Currency of `price` |
| `base_currency` | Currency used by the index calculation |
| `fx_rate` | Multiplier converting the instrument currency to `base_currency` |
| `market_cap` | Investable market capitalization in `base_currency` |
| `benchmark_weight` | Starting weight in the underlying universe |
| `reference_date` | Information cutoff for the review |
| `effective_date` | Date on which the calculated weights apply |

`instrument_id` must be non-null and unique. The provider adapter is responsible for identifier mapping, classification mapping, unit conversion, and the final canonical column names. If `market_cap` is derived, the standard formula is:

```text
market_cap = shares * free_float * price * fx_rate
```

`benchmark_weight` must use the same eligible universe and base-currency convention as `market_cap` and should sum to one, subject only to normal floating-point tolerance.

### Daily market data

Daily rows must contain:

| Column | Meaning |
| --- | --- |
| `instrument_id` | Canonical instrument identifier |
| `business_date` | Observation date |
| `price_return` | Price-only return for the observation period |
| `gross_dividend` | Gross dividend return component |
| `net_dividend` | Net dividend return component |
| `market_cap` | Market capitalization for that business date |

The canonical loading rule derives gross and net total-return components from
these fields. A row dated after the review cutoff is rejected when it is loaded
as weight-production input. A simulator can request a later realized date
range after the review weights have already been fixed.

### Review schedules

A review schedule contains `reference_date` and `effective_date`. Business-day calendars are separate provider data and use explicit `calendar_id`, `start_date`, and `end_date` parameters.

## Provider capabilities

Adapters implement only the capabilities they can supply:

| Capability | Data supplied | Main parameters |
| --- | --- | --- |
| `load_universe` | Canonical point-in-time universe | `universe_id`, `reference_date`, `effective_date` |
| `load_business_days` | Valid business dates | `calendar_id`, `start_date`, `end_date` |
| `load_review_schedule` | Review cutoff and effective dates | `calendar_id`, `start_date`, `end_date` |
| `load_daily_market_data` | Canonical daily market rows | `instrument_ids`, `start_date`, `end_date` |
| `load_membership` | Instrument membership flags | `index_id`, `start_date`, `end_date` |
| `load_reference_data` | Identifiers and classifications | `instrument_ids`, `reference_date`, `fields` |
| `load_third_party_data` | Explicitly typed third-party fields | `data_type`, `instrument_ids`, `fields`, `reference_date`, `parameters` |

The registry resolves providers by name and capability. Universe, market-data, membership, reference-data, and third-party loading rules require an explicit `provider_name`; the controlled file rule uses the registered `file` provider unless another file provider is named. ICAPA never selects a provider by fallback order.

### Available integration points

| Integration | Current role | Default connection or dataset |
| --- | --- | --- |
| FactSet | SQL Server adapter placeholder | None |
| Snowflake | Optional future placeholder | None |
| CSV/Excel | Controlled ad-hoc file input | Local file supplied explicitly |

FactSet requires an injected SQL executor after its connection technology, authentication, schemas, and datasets are defined. Snowflake remains inactive until its account and dataset design are provided. No placeholder contains credentials, connection strings, schemas, queries, or sample production data.

## Typed third-party data

Specialized external data is deliberately visible in configuration. `ThirdPartyDataType` supports only:

- `ThirdPartyFactorData`
- `ThirdPartyEmissions`
- `ThirdPartyLiquidityData`

`AddThirdPartyData` requires all of the following parameters:

| Parameter | Purpose |
| --- | --- |
| `data_type` | Declares the external data category |
| `fields` | Exact canonical fields requested by the external weight producer |
| `provider_name` | Explicit registered adapter name |
| `provider_parameters` | Optional source-specific query controls interpreted only by the adapter |

The rule calls the provider's `load_third_party_data` capability at `reference_date` and joins the result to the constituent table by `instrument_id`. It does not infer fields, providers, families, or fallback datasets. The adapter must reject an unsupported `data_type` or field rather than substitute another dataset.

## Data-loading rules and destinations

| Rule | Input source | Destination |
| --- | --- | --- |
| `AddUnderlyingIndex` | `load_universe` | Canonical constituent table in `DataContext` |
| `AddReturns` | `load_daily_market_data` | Instrument-by-`business_date` market-data table |
| `AddIdentifierFacts` | `load_reference_data` | Columns joined to constituents by `instrument_id` |
| `AddIndexMemberships` | `load_membership` | Boolean membership columns on constituents |
| `AddThirdPartyData` | Typed `load_third_party_data` request | Explicit specialized fields joined to constituents |
| `ApplyExclusions` | Existing canonical columns or membership data | Eligibility flags, reasons, and renormalized starting weights |
| `ImportData` | Registered CSV/Excel file provider | Explicitly selected columns merged on configured keys |
| `LoadAllData` | One universe rule followed by an ordered rule list | Fully prepared `DataContext` |

The constituent table is the point-in-time input to the external weight
producer. Daily market data remains a separate time series. The producer writes
its final result as `index_weight`; it must not overwrite `benchmark_weight`.

## Configuration parameters

Parameters fall into four groups and should be kept separate:

1. **Review parameters:** `reference_date`, `effective_date`, and any explicit schedule or `calendar_id`.
2. **Provider parameters:** `provider_name` plus adapter-specific `provider_parameters`. Secrets belong in the deployment environment or a secret manager, never in weight-production configuration.
3. **Dataset parameters:** `universe_id`, membership `index_id`, requested `fields`, date ranges, file paths, merge keys, and third-party data type.
4. **Weight-production parameters:** objective settings, constraints, tolerances, weight bounds, and other calculation settings owned by the external producer.

Provider parameters determine where data comes from. Dataset parameters
determine what is requested. Weight-production parameters determine how
canonical data is transformed. A weight-production parameter must never
silently change the provider or dataset.

Unknown configuration fields, missing providers, missing capabilities, ambiguous capability resolution, missing canonical columns, invalid date relationships, duplicate identifiers, and data beyond the cutoff must fail with a readable exception.

## Adapter implementation checklist

1. Implement only the required provider protocols.
2. Map physical identifiers and columns to the canonical contracts inside the adapter.
3. Normalize types, currencies, units, classifications, and dates before returning data.
4. Enforce point-in-time availability at `reference_date`.
5. Register the adapter under an explicit, deployment-controlled name.
6. Configure each loading rule with that provider name and the minimum required fields.
7. Test the adapter with synthetic rows before using controlled external data.
8. Verify weight totals, identifier uniqueness, date cutoffs, and unavailable-data failures.

This boundary allows a provider implementation to change without changing the
external weight producer.

## Public extension boundary

`portfolio_construction` contains generic optimisation contracts, objective and
constraint builders, and solver interfaces. The private extension folders
`portfolio_construction/methodologies/` and
`portfolio_construction/rules/engines/` intentionally contain only `.gitkeep`
in the public repository. Weight-production implementations are supplied
outside the public core.

The named-workspace, simulation, analytics, and reporting workflow is described
in [the index research workflow](index_research_workflow.md).
