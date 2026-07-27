# ICAPA Index Research Workflow

## Purpose

ICAPA separates index research into reusable stages:

```text
Provider
    -> canonical contract
    -> external weight producer
    -> cached target weights
    -> simulation
    -> analytics
    -> report
```

The backtester invokes an external weight producer for each review and collects
its target weights. It is not a trade execution system. Daily returns, weight
drift, rebalances, index levels, formal turnover, analytics, and reporting are
separate downstream stages.

## Named workspaces and retention

A `workspace_name` gives one research project a stable on-disk location. The
default root is:

```text
~/.icapa/workspaces
```

Deployments may override only the root through `ICAPA_WORKSPACE_ROOT`. Individual
callers cannot direct artifacts to arbitrary folders. A fingerprint derived
from the index ID, external weight-producer identity, calculation parameters,
provider labels, calendar semantics, and data revision creates an immutable run
beneath the name:

```text
<workspace root>/
  <workspace name>/
    runs/
      <calculation fingerprint>/
        manifest.json
        reviews/
        artifacts/
          simulation/
          analytics/
        reports/
```

Disk artifacts have no automatic expiry. They remain until an operator applies
the deployment's explicit retention policy. This is deliberate: a completed
research run should remain reproducible. The in-process cache lasts until the
Python process exits or `clear_memory_cache()` is called.

`CachePolicy.REUSE` reads a valid memory or disk artifact and calculates only a
missing review. `CachePolicy.REFRESH` recalculates requested reviews.
`CachePolicy.READ_ONLY` fails if any requested artifact is absent.

The requested overall backtest start and end dates are not part of a review
cache key. Therefore:

- a shorter run reuses the overlapping review weights;
- a longer run reuses existing reviews and calculates only the additional ones;
- a changed weight-production parameter, provider configuration, index ID, or
  data revision receives a different fingerprint;
- corrupted or incomplete artifacts fail checksum validation and are not reused.

Production adapters should provide a stable data revision or snapshot label.
When a provider does not expose one, the run must be refreshed whenever its
source data changes.

## Underlying configuration

Underlying identifiers must not control behavior through hard-coded product
codes or deployment-specific conditionals. An `UnderlyingProfile` contains the
external settings for one coherent universe. An
`UnderlyingMappingRegistry` maps an exact identifier or a prefix to that
profile.

```python
from icapa.helpers import UnderlyingMappingRegistry, UnderlyingProfile

us_equity = UnderlyingProfile(
    profile_name="us_equity_research",
    universe_id="US_EQUITY_UNIVERSE",
    universe_provider_name="universe_provider",
    calendar_id="US",
    calendar_provider_name="calendar_provider",
    market_data_provider_name="market_data_provider",
    base_currency="USD",
    fx_provider_name="fx_provider",
    tax_provider_name="tax_provider",
    dividend_provider_name="dividend_provider",
    universe_provider_parameters={"dataset": "broad_equity"},
    calendar_provider_parameters={},
    market_data_provider_parameters={},
    simulation_parameters={
        "base_value": 100.0,
        "weight_drift": "price_return",
        "dividend_treatment": "NYSE",
    },
)

mappings = UnderlyingMappingRegistry()
mappings.register_prefix("US-EQUITY-", us_equity)
mappings.register_exact("US-EQUITY-SPECIAL", us_equity)

match = mappings.resolve_match("US-EQUITY-LARGE")
assert match.profile.calendar_id == "US"
assert match.profile.market_data_provider_name == "market_data_provider"
```

Exact mappings take precedence over prefixes. Among prefixes, the longest match
wins. The registry contains no built-in identifiers and no default profile.
Unknown identifiers fail explicitly.

The provider names in a profile are deployment-controlled registry keys, not
connection strings. Credentials, schemas, SQL, and physical field mappings stay
inside provider adapters or deployment configuration. Optional FX, tax, or
dividend providers may be omitted only when the canonical market-data provider
already supplies the required adjusted observations.

This replaces family-style enums with direct settings. If a deployment calls a
calendar `US`, configure `calendar_id="US"`. If it uses another label, store
that exact deployment label instead of adding a new hard-coded branch.

## Constructing and reusing review weights

The deployment supplies `weight_producer`. It must implement
`execute(data_context)` and write a finite, non-negative `index_weight` that
sums to one. The public core does not provide a concrete weight-production
implementation.

```python
from icapa.backtesting import Backtester, Calendar
from icapa.workspace import CachePolicy

calendar = Calendar(
    start_date="2026-01-01",
    end_date="2026-12-31",
    calendar_id=match.profile.calendar_id,
    provider_name=match.profile.calendar_provider_name,
    provider_parameters=dict(match.profile.calendar_provider_parameters),
)

backtester = Backtester(
    index_id="RESEARCH_INDEX",
    calendar=calendar,
    methodology=weight_producer,
    workspace_name="research_workspace_20260410",
    data_revision="review-snapshot-2026-04-10",
    cache_policy=CachePolicy.REUSE,
)
backtest_result = backtester.run()
```

Each `effective_date` produces one canonical constituent frame with
`index_weight`. `backtest_result.metadata.reviews` records whether each review
was computed, read from process memory, or read from disk. Reusing the same name
and calculation fingerprint with another calendar range reuses the overlapping
reviews.

## Daily index simulation

`IndexSimulator` consumes the completed review weights and realized canonical
market data. It does not rerun the external weight producer.

```python
from icapa.backtesting import IndexSimulator, SimulationParams

simulation = IndexSimulator(
    backtest_result=backtest_result,
    market_data_provider_name=match.profile.market_data_provider_name,
    provider_parameters=dict(match.profile.market_data_provider_parameters),
    start_date="2026-01-20",
    end_date="2026-12-31",
    data_revision="daily-market-snapshot-2027-01-05",
    workspace=backtester.workspace_store,
    params=SimulationParams(**dict(match.profile.simulation_parameters)),
).run()
```

The simulator:

- applies target weights on `effective_date`, or on the next observed business
  day when configured;
- drifts index and benchmark weights between reviews;
- calculates price, gross-total, and net-total returns and levels;
- simulates the benchmark with its own review weights;
- calculates formal one-way turnover from pre-trade drifted weights to new
  targets;
- records daily opening and closing weights for analysis;
- stores a range-specific simulation artifact when a workspace and explicit
  data revision are supplied.

`WeightDrift.PRICE_RETURN` is the default. `WeightDrift.MARKET_CAP` is available
when reliable daily market capitalization is supplied. Corporate-action
adjustments, currency conversion, tax treatment, and dividend inputs must
already be represented correctly in the canonical daily market data.

The simulator is intended for index research. It does not model order routing,
bid-ask spreads, execution slippage, or a portfolio manager's trading book.

## Analytics

```python
from icapa.analytics import analyze_backtest

analytics = analyze_backtest(backtest_result, simulation)
```

The provider-neutral analytics layer includes:

- review weight validation;
- constituent count, maximum weight, top-10 weight, HHI, effective N, and
  active share;
- country and industry portfolio, benchmark, and active exposures;
- adjacent target-review weight change;
- formal simulation turnover as a separate measure;
- annualized performance, volatility, drawdown, tracking error, and information
  ratio from daily index and benchmark returns;
- optional Brinson-Fachler attribution from explicitly supplied, point-in-time
  aligned inputs.

Analytics never loads providers, changes target weights, writes a workspace, or
renders a report.

## Reporting

```python
from icapa.reporting import write_index_research_report

report_path = write_index_research_report(
    backtester.workspace_store,
    backtest_result,
    filename="index_research_report",
    simulation=simulation,
    analytics=analytics,
    data_sources=[
        {
            "capability": "universe",
            "provider_name": match.profile.universe_provider_name,
            "data_type": "canonical_universe",
            "fields": ["instrument_id", "benchmark_weight"],
        },
        {
            "capability": "daily_market_data",
            "provider_name": match.profile.market_data_provider_name,
            "data_type": "canonical_daily_market_data",
            "fields": ["price_return", "gross_dividend", "net_dividend"],
        },
    ],
)
```

Reports can be written only under the fingerprinted workspace's `reports/`
directory. The fixed template contains:

- Overview
- Review Schedule
- Latest Holdings
- All Review Weights
- Performance
- Exposures
- Turnover
- Attribution
- Methodology Parameters
- Data Sources
- Validation

Optional sections state `Not available` when their required inputs are absent.
The report adapter uses field allowlists, blocks formula injection and external
workbook links, and never serializes provider parameters, credentials, or
simulation-private metadata.

## Generic optimisation contracts

The public `portfolio_construction` package separates optimisation into:

```text
objective builder
    + linear/nonlinear constraint specifications
    + OptimizationProblem
    + PortfolioSolver
    -> verified OptimizationResult
```

The default solver is `ScipySLSQPSolver`. An external weight producer may inject
a `PortfolioSolver` or provide another solver without changing data loading.
Reusable builders cover weight bounds, group constraints, turnover, tracking
error, distance objectives, and covariance-based objectives. Every solver
result includes an objective value, iteration count, message, and maximum
verified constraint violation.

## Public extension boundary

`portfolio_construction` contains the generic optimisation contracts described
above plus empty extension placeholders. The private implementation folders
`portfolio_construction/methodologies/` and
`portfolio_construction/rules/engines/` intentionally contain only `.gitkeep`
in the public repository. Deployments connect their external weight producer
without changing the provider, cache, simulation, analytics, or report
contracts.
