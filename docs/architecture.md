# ICAPA Architecture

ICAPA is an index-research platform. It constructs target weights at review
dates, simulates the index between effective dates, evaluates the resulting
research output, and writes review-ready deliverables. It is not an execution,
order-management, or portfolio-manager book system.

## Domain ownership

| Package | Owns | Does not own |
| --- | --- | --- |
| `data_sources` | Canonical data contracts, provider protocols, provider registration, snapshot evidence, universe profiles, and data services | Methodology logic, simulation, analytics, or physical workspace storage |
| `portfolio_construction` | Review `DataContext`, methodologies, reusable engines, construction rules, recipes, constraints, and optimization | Daily index simulation or physical cache files |
| `backtesting` | Review orchestration, calendars, rebalance schedules, daily simulation, drift, return arithmetic, and simulation results | Provider implementations, analytics, or report rendering |
| `analytics` | Performance, constituents, exposures, attribution, risk, events, regimes, comparisons, reconciliation, and data quality | Research run orchestration or persistent cache implementation |
| `workspace` | Automatic identity, manifests, SQLite catalog, immutable artifacts, cache adapters, and persisted-format readers | Index methodology or analytics calculations |
| `research` | User-facing specifications, workspace sessions, run lifecycle, batch runs, scenarios, sensitivity, and notebook presentation | Low-level database or artifact mechanics |
| `reporting` | Report contracts, payload builders, Excel renderers, dashboards, templates, and multi-format bundles | Construction, simulation, or analytics calculations |

The repository deliberately has no top-level `tools` or `helpers` package.
Reusable code belongs to the business domain that defines its behavior.
Package-private helpers use a local `_utils.py` only when no more specific
business name applies.

The source tree follows the same ownership model:

```text
icapa/
├── data_sources/
│   ├── contracts.py
│   ├── providers/
│   ├── provenance/
│   ├── services/
│   └── universes/
├── portfolio_construction/
│   ├── context.py
│   ├── methodologies/
│   ├── engines/
│   ├── rules/
│   ├── recipes/
│   └── optimization/
├── backtesting/
│   ├── backtester.py
│   ├── calendar/
│   ├── reviews/
│   └── simulation/
├── analytics/
│   ├── performance/
│   ├── constituents/
│   ├── exposures/
│   ├── attribution/
│   ├── risk/
│   ├── events/
│   ├── regimes/
│   ├── comparisons/
│   ├── reconciliation/
│   └── quality/
├── workspace/
│   ├── repository.py
│   ├── manifests.py
│   ├── identity.py
│   ├── catalog.py
│   ├── artifacts.py
│   ├── caches/
│   └── readers/
├── research/
│   ├── models.py
│   ├── results.py
│   ├── workspace.py
│   ├── runners/
│   ├── scenarios/
│   ├── sensitivity/
│   └── notebook/
└── reporting/
    ├── contracts.py
    ├── builders/
    ├── excel/
    ├── dashboards/
    ├── bundle.py
    └── templates/
```

## Dependency direction

The calculation domains follow one direction:

```text
data_sources
    |
    v
portfolio_construction
    |
    v
backtesting
    |
    v
analytics
```

`workspace` may serialize stable contracts from these domains, but calculation
domains do not import workspace implementations. `research` composes the
domains and workspace adapters. `reporting` consumes completed result
contracts. This rule prevents circular imports and keeps calculations usable
without a named workspace.

## Research pipeline

### 1. Load point-in-time data

A provider is selected explicitly for each capability. The provider returns a
canonical universe or market-data table and records secret-safe snapshot
evidence. Provider credentials and queries never enter a run manifest or
report.

### 2. Construct review target weights

At each `reference_date`, a methodology or `IndexRecipe` receives a
`DataContext`. Data-loading, transformation, eligibility, selection, weighting,
optimization, constraint, and validation stages produce canonical
`index_weight` values for the corresponding `effective_date`.

Methodologies produce target weights only. They do not calculate daily index
levels.

### 3. Simulate effective-date segments

The backtester applies target weights on the authoritative effective-date
schedule. The simulation engine evolves them between effective dates according
to the configured drift model, dividend treatment, and rebalance phase.
Simulation segments reference shared market-data artifacts instead of copying
the same observations into every parameter run.

### 4. Analyze the research result

Analytics modules consume completed review and simulation contracts. A research
profile can calculate performance, drawdowns, turnover, constituent changes,
exposures, attribution, risk, liquidity, capacity, events, regimes, target
attainment, constraint diagnostics, data freshness, and candidate differences.
Expensive or specialized analyses run only when selected.

### 5. Compare candidates

A named research workspace can compare one baseline with one or more
candidates. Comparison aligns review dates and business dates explicitly and
reports definition, parameter, membership, weight, exposure, turnover,
performance, validation, and lineage differences.

### 6. Write deliverables

Reporting converts completed results into whitelisted report tables. The Excel
renderer preserves the established workbook contract, while the report bundle
can add JSON metadata and complete Parquet tables. Large tables are split
deterministically in Excel and remain complete in Parquet.

## Extension points

- Add a provider under `data_sources/providers` and implement only the
  capabilities it supplies.
- Add a reusable construction operation to the matching rule category.
- Add a private methodology under `portfolio_construction/methodologies`.
- Add a private reusable calculation engine under
  `portfolio_construction/engines`.
- Add a recipe stage under `portfolio_construction/recipes`.
- Add an optimizer backend under `portfolio_construction/optimization`.
- Add an analysis under the matching `analytics` research question.
- Add a report section through a `reporting` builder rather than calculating
  it in the renderer.

Private methodology, engine, and selected processing implementations remain
available in development checkouts but are excluded from the public source
distribution and wheel.

## Public entry point

The primary user interface is:

```python
from icapa.research import ResearchSpec, ResearchWorkspace

workspace = ResearchWorkspace.open("example_workspace")
run = workspace.run(spec)
```

The root package exposes only the primary research objects and `IndexRecipe`.
Advanced contracts are imported from the domain that owns them. Importing
`icapa` performs no provider registration, file access, or workspace creation.
