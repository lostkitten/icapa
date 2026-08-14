# ICAPA

<img src="assets/icapa.png" alt="ICAPA" width="360">

ICAPA is a provider-neutral platform for index construction research. It
creates effective-date target weights, simulates their daily evolution,
analyzes and compares research candidates, and writes review-ready
deliverables. It is not an order-management, execution, or portfolio-manager
book system.

## Research pipeline

```text
ResearchSpec / IndexRecipe
    -> canonical provider data and automatic snapshot identity
    -> effective-date target weights
    -> segmented daily index simulation
    -> analytics and baseline/candidate comparison
    -> Excel, JSON, and Parquet deliverables
```

Researchers supply methodology logic, parameters, providers, dates, and cache
policy. ICAPA automatically records executable source identity, provider
adapter identity, data snapshot evidence, runtime versions, request identity,
and immutable output checksums.

## Package map

| Package | Responsibility |
| --- | --- |
| `data_sources` | Canonical data contracts, provider protocols, explicit registration, universe profiles, and data services |
| `portfolio_construction` | Review context, methodologies, engines, rules, recipes, constraints, and optimization |
| `backtesting` | Calendars, review orchestration, drift, effective-date segments, and daily index simulation |
| `analytics` | Performance, constituents, exposures, attribution, risk, events, regimes, comparisons, reconciliation, and data quality |
| `workspace` | Automatic identity, manifests, catalog, immutable artifacts, typed caches, and persisted-format readers |
| `research` | User-facing workspace, specifications, results, batch runs, scenarios, sensitivity, and notebook presentation |
| `reporting` | Report contracts, builders, Excel rendering, dashboards, templates, and multi-format bundles |

There is no generic `tools` or `helpers` package. Code belongs to the domain
that defines its behavior.

See [Architecture](docs/architecture.md) for the dependency rules, extension
points, and end-to-end data flow.

## Quick start

ICAPA requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,qp]'
pytest -q
```

The primary API is the research workspace:

```python
from icapa.research import ResearchSpec, ResearchWorkspace

# `spec` references a deployment-supplied methodology or IndexRecipe,
# explicit providers, review dates, simulation settings, and analytics.
workspace = ResearchWorkspace.open("example_workspace")
run = workspace.run(spec)
```

Named workspaces live under `~/.icapa/workspaces/<workspace-name>` by default.
Deployments may set `ICAPA_WORKSPACE_ROOT` to select one controlled root.
Researchers choose whether reusable artifacts are disabled, reused, refreshed,
or read-only.

## Optional research features

```bash
python -m pip install -e '.[analytics]'
python -m pip install -e '.[explainability]'
python -m pip install -e '.[notebook]'
```

Optional packages are imported only when their feature is requested. Missing
optional dependencies produce an actionable error and never change the chosen
calculation silently.

## Public construction methodologies

Public distributions include the provider-neutral Factor Tilt, Minimum
Variance, and Quantile Selection methodologies together with their reusable
engines. The public engine package also includes `EntropyExposureEngine`,
backed by the EGMU solvers. Import these APIs from
`icapa.portfolio_construction.methodologies` and
`icapa.portfolio_construction.engines`.

All methodology and engine modules in these packages ship in both source and
wheel distributions:

```text
portfolio_construction/methodologies/
portfolio_construction/engines/
```

Deployment-specific provider adapters and selected processing extensions may
still be supplied outside the public package; they are separate from the
public methodology API.

## Data and reports

The [data-loading guide](docs/data_loading_guide.md) defines canonical fields,
date semantics, provider capabilities, and adapter parameters. The
[research workflow guide](docs/index_research_workflow.md) explains automatic
identity, cache control, simulation, analytics, comparison, and reporting.

Excel v1 output remains available for established report workflows. The
current report bundle adds sanitized JSON metadata and complete Parquet tables.
Large tables are split deterministically in Excel rather than silently
truncated.

## License

ICAPA is available under the MIT License.
