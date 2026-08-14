# ICAPA Index Research Workflow

## Purpose and boundary

ICAPA is designed for researchers who construct, backtest, compare, and review
indices. A methodology produces target weights at effective dates. The
simulation layer then evolves those weights between effective dates using
realized market data.

ICAPA does not model orders, execution venues, bid-ask spreads, a live trading
book, or real-time portfolio-manager positions.

```text
ResearchSpec / IndexRecipe
    -> automatic identity and cache preflight
    -> canonical provider snapshots and stage artifacts
    -> effective-date target weights
    -> effective-date simulation intervals
       streamed and cached in calendar-month partitions
    -> analytics plugins and baseline/candidate comparison
    -> Excel + JSON + Parquet report bundle
```

`DataContext` remains the compatibility carrier used by existing construction
code. It is not intended to become a production database or data lake.

## Research specification

The high-level request is split into business definition and execution scope:

- `IndexDefinition` contains the index ID, methodology or recipe, declared
  rebalance frequency, base currency, and stable definition attributes.
- `Calendar` contains the exact `reference_date` and `effective_date` pairs.
- `ResearchSimulationSpec` contains realized market-data configuration, date
  range, drift behavior, and materialization choices.
- `AnalyticsSpec` selects a deterministic analytics profile and explicit return
  series.
- `CacheOptions` controls reusable artifacts without changing calculation
  identity.
- `ReportBundleSpec` optionally creates a deliverable from the completed run.
- `recipe_providers` explicitly binds each native recipe capability to one
  provider and its non-secret parameters.
- `random_seed` optionally overrides the automatically derived recipe seed.
- `label`, `tags`, and `ResearchStatus` support workspace review governance.

```python
from icapa import (
    AnalyticsSpec,
    CacheMode,
    CacheOptions,
    CacheStage,
    Calendar,
    IndexDefinition,
    PriceReturnDrift,
    RebalanceFrequency,
    RebalancePhase,
    ResearchSimulationSpec,
    ResearchSpec,
    ResearchWorkspace,
    SimulationMaterialization,
    SimulationParams,
    WeightSnapshotMode,
)

calendar = Calendar.from_dates(
    [
        {
            "reference_date": "2026-03-20",
            "effective_date": "2026-04-01",
        },
        {
            "reference_date": "2026-06-19",
            "effective_date": "2026-07-01",
        },
    ]
)

spec = ResearchSpec(
    definition=IndexDefinition(
        index_id="RESEARCH_INDEX",
        name="Research Index",
        base_currency="USD",
        methodology=recipe_or_methodology,
        rebalance_frequency=RebalanceFrequency.QUARTERLY,
    ),
    calendar=calendar,
    simulation=ResearchSimulationSpec(
        market_data_provider_name="market_data",
        provider_parameters={"dataset": "daily_returns"},
        start_date="2026-04-01",
        end_date="2026-12-31",
        params=SimulationParams(
            index_drift=PriceReturnDrift(),
            benchmark_drift=PriceReturnDrift(),
            rebalance_phase=RebalancePhase.OPEN,
            materialization=SimulationMaterialization(
                weight_snapshots=WeightSnapshotMode.NONE,
                include_asset_returns=False,
            ),
        ),
        segmented_cache=True,
        streaming=True,
    ),
    analytics=AnalyticsSpec.standard_research(),
    cache=CacheOptions(
        mode=CacheMode.OFF,
        stage_modes={
            CacheStage.SOURCE_DATA: CacheMode.REUSE,
            CacheStage.REVIEWS: CacheMode.REUSE,
            CacheStage.SIMULATION: CacheMode.REUSE,
            CacheStage.ANALYTICS: CacheMode.REUSE,
        },
    ),
    label="baseline",
    tags=("quarterly", "research"),
)

workspace = ResearchWorkspace.open("research_index_20260410")
baseline = workspace.run(spec)
```

The provider and methodology objects in this example may be deployment
supplied. The public core includes its canonical provider-neutral
methodologies, but contains no production credentials, SQL, or schema.

## Automatic identity and run manifests

Researchers do not enter a methodology version, source digest, provider adapter
version, data snapshot ID, calendar revision, dependency digest, runtime
version, or random seed merely to make a cache key.

ICAPA records three related identities:

| Identity | Scope | Main inputs |
| --- | --- | --- |
| `definition_fingerprint` | Stable index construction definition | Index definition, methodology/recipe/engine/solver/kernel identity, construction and review provider adapters, calendar provider/configuration, and construction runtime |
| `request_fingerprint` | Exact requested calculation | Review pairs, schedule-validation controls, explicit random-seed override, simulation range/provider/drift/materialization, analytics profile, and execution components |
| `result_fingerprint` | Actual completed result | Request fingerprint plus consumed input digests and immutable output artifact digests |

The definition fingerprint intentionally excludes the requested review and
simulation range. A shorter or longer request for the same construction
definition can therefore reuse overlapping reviews, source partitions, and
simulation coverage. Simulation-only providers, drift models, materialization,
and simulator identity belong to the request and immutable simulation-segment
keys, not the construction definition. The request fingerprint still
distinguishes the exact calculation. Governance-only labels, tags, and status
are stored in the manifest but deliberately excluded from calculation
fingerprints.

### Code and runtime identity

For importable Python components, the identity collector records:

- qualified component and distribution identity;
- installed distribution version when available;
- a cached digest of all installed Python files in that distribution;
- source-file and local import-closure content digests;
- Git commit metadata when the source is inside a repository;
- actual source content, including dirty local files;
- immutable component configuration;
- Python, ICAPA, NumPy, Pandas, SciPy, PyArrow, and selected solver versions;
- dependency-lock digest when a recognized lock/configuration file is present.

A cacheable recipe stage also includes its implementation and reachable helper
source, callable defaults, keyword defaults, closure or bound state,
configuration, declared inputs, input artifact digests, relevant provider
revisions, and declared review dimensions. Downstream recipe configuration is
not inserted into an upstream content-stage key, so unchanged data and feature
stages can be shared across parameter scenarios.

A random stage declares `uses_randomness`. By default the high-level runner
derives a deterministic seed from `definition_fingerprint` and records it.
`ResearchSpec.random_seed` provides an explicit non-negative override. The seed
enters only cache keys for stages that declare randomness; non-random upstream
stages remain reusable.

Dynamic notebook callables or opaque components that have no stable source
identity may run with cache mode `OFF`. ICAPA does not use a class name as a
false version. Any non-`OFF` calculation mode fails when executable identity
cannot be proved. This is distinct from a source-verifiable provider that lacks
a preflight snapshot token: that provider may be read and content-hashed before
downstream reuse.

### Data identity

Data identity is automatic:

- CSV and Excel adapters hash the source file bytes without storing the local
  path in the manifest.
- In-memory and pseudodata frames use a logical content digest covering values,
  schema, dtypes, columns, and index metadata.
- A provider may implement
  `describe_snapshot(capability, request)` to expose a lightweight,
  non-sensitive snapshot token before a data read.
- A provider may expose `research_data_identity` when its identity service is
  separate from data retrieval.
- Without a preflight snapshot token, ICAPA reads and canonicalizes the data,
  computes its content digest, and may persist the new object. That verified
  content revision can then reuse downstream simulation work, but the provider
  is read again on a later run to prove that its content is still unchanged.

Provider-backed recipe stages declare a `ProviderRequestSpec` for each
capability before review caching can be enabled. The declaration identifies
the exact keyword request as provider-binding parameters plus explicitly
selected review dimensions, such as `reference_date` and `effective_date`.
This prevents a generic synthetic request from being used as evidence for a
different custom-stage call. A dynamic provider request that cannot be
declared remains available with cache mode `OFF`; non-`OFF` review caching
fails clearly instead of trusting an approximate identity.

Provider snapshot payloads are reduced to digests. Credentials, passwords,
tokens, connection strings, hosts, SQL, query text, schemas, and user names are
redacted or omitted from manifests and reports.

Successful and failed named executions both receive a secret-safe
`run_manifest.json`. A failed manifest records only the exception type and a
safe message, not the provider exception body.

## Named workspace and immutable storage

The default root is:

```text
~/.icapa/workspaces
```

`ICAPA_WORKSPACE_ROOT` can change that deployment-wide root. A caller supplies a
validated workspace name, not an arbitrary output path.

The current physical layout is:

```text
<workspace root>/<workspace name>/
  catalog.sqlite
  objects/
    sha256/<prefix>/<content-digest>/<file-checksum>.parquet
    metadata/...
  bindings/<cache-stage>/<cache-key>/...
  runs/<definition-fingerprint>/
    executions/<execution-id>/run_manifest.json
    reviews/
    simulations/
    analytics/
    reports/<report-bundle-id>/
  state/
    invalidations/...
    research_status/...
```

High-level v2 bundles use the per-definition `runs/.../reports/` location.
The other per-definition stage directories are stable reference namespaces;
content-addressed objects and reusable bindings remain normalized at workspace
level so identical inputs can be shared across definitions without copying.
The v1 workspace-root `reports/` path remains available to the direct
reporting and workspace interfaces.

The SQLite catalog uses WAL transactions. Large tabular artifacts use
content-addressed, ZSTD-compressed Parquet with logical-content and file-byte
checksums. Manifests and compact metadata use checksummed JSON. Existing JSON
workspace artifacts remain readable; new v2 tabular writes use Parquet.

Objects are immutable. `REFRESH` creates or resolves a new immutable object and
updates only its binding; it does not overwrite an old object. A checksum
mismatch is corruption and is never accepted as a cache hit.

`ResearchWorkspace` provides:

- `list()`, `latest()`, `open_run()`, and `coverage()`;
- labels, tags, and draft/in-review/approved/rejected/superseded status;
- `verify()` for manifests and attached objects;
- `invalidate()` to make a run inactive without deleting it;
- `rebuild_catalog()` from checksummed manifests and sidecars;
- `prune(dry_run=True)` to inspect unreferenced objects before explicit removal.

There is no automatic expiry. Retention and pruning are explicit operational
choices. Researchers can remove local artifacts, but a missing or damaged
object will no longer be trusted.

## Cache modes and cache stages

```python
from icapa import CacheMode, CacheOptions, CacheStage

cache = CacheOptions(
    mode=CacheMode.OFF,
    stage_modes={
        CacheStage.SOURCE_DATA: CacheMode.REUSE,
        CacheStage.REVIEWS: CacheMode.REUSE,
        CacheStage.SIMULATION: CacheMode.REFRESH,
    },
)
```

| Mode | Read reusable artifacts | Calculate/provider read | Write reusable binding |
| --- | --- | --- | --- |
| `OFF` | No | Yes | No |
| `REUSE` | Yes, after identity and checksum validation | Only missing or unverifiable work | Yes |
| `REFRESH` | No | Yes | Yes, to immutable content |
| `READ_ONLY` | Required | No provider method or calculation; local artifacts are verified | No |

The high-level API defaults to `OFF`; it never silently forces a researcher
to use cached calculations. A named workspace still writes its execution
manifest and immutable result artifacts in `OFF` mode. Those result artifacts
are lineage evidence, not reusable-cache claims.

Current reusable stages are:

- `SOURCE_DATA`: canonical daily-market-data partitions plus verified monthly
  coverage descriptors. Keys exclude the index, methodology, and scenario so
  exact or containing partitions can be shared safely.
- `REVIEWS`: verified effective-date results and persistent native recipe-stage
  artifacts.
- `SIMULATION`: immutable closed effective-date segments plus calendar-month
  checkpoints for the open tail, assembled and sliced for each request.
- `ANALYTICS`: complete plugin results identified from the analytics
  specification, plugin-runner source, review contents, simulation contents,
  and optional research inputs.

`ResearchWorkspace.run()` builds the `AnalyticsCacheIdentity` automatically and
honors `OFF`, `REUSE`, `REFRESH`, and `READ_ONLY` for
`CacheStage.ANALYTICS`. Analytics tables use immutable Parquet; compact
result/specification/diagnostic metadata is committed last. The run manifest
records the selected mode, the automatic analytics input digest, and whether
the result came from calculation or the workspace.

For source data, `READ_ONLY` requires a locally recorded provider snapshot
identity so the correct partition key can be derived without calling the
provider. It verifies every required source artifact before accepting a
downstream simulation hit. With no snapshot protocol, `REUSE` calls the
provider, canonicalizes and hashes the response, and can reuse simulation work
identified by that content; it never assumes an older response is current.

## IndexRecipe: standard execution without a standard algorithm

`IndexRecipe` is an execution shell, not a fixed methodology. Its conceptual
pipeline is:

```text
Data requirements
    -> Transform
    -> Eligibility
    -> Selection
    -> Weighting or optimization
    -> Constraints
    -> Validation
    -> canonical index_weight
```

The recipe compiles to a directed acyclic graph. Each `IndexStage` declares:

- namespaced input and output artifacts;
- immutable canonical configuration;
- implementation version and source identity;
- current-review and previous-review requirements;
- provider capabilities;
- cache scope (`CONTENT`, `RECIPE`, `RUN`, or `DISABLED`);
- determinism, randomness, side-effect class, and parallel-safety properties;
- diagnostics and final artifact schema.

The graph derives dependencies from artifact flow in addition to explicit
ordering. It rejects duplicate producers, missing inputs, cycles, undeclared
outputs, and invalid final weights.

Researchers have five extension levels:

1. Compose reusable standard stages.
2. Implement an arbitrary custom Python `IndexStage`.
3. Expose a composite stage behind a stable public contract.
4. Use one monolithic custom stage that directly produces target weights.
5. Wrap an `execute(DataContext)` methodology with
   `IndexRecipe.from_methodology(methodology)`.

The only universal construction contract is a finite, non-negative
`index_weight` indexed by `instrument_id` and summing to one within tolerance.
Custom methodologies do not have to use a standard scorer, selector, or
optimizer.

### Previous-review state

A stage can explicitly request previous target weights, membership, ranks, or
custom state through namespaced artifacts. When a stateful sequence starts in
the middle, the caller must provide a valid previous state, replay earlier
reviews, or resolve a cached seed. Missing required state fails explicitly; the
requested start is never treated silently as the first historical review.

## Review schedule and rebalance frequency

Every review contains:

- `reference_date`: the point-in-time information cutoff used to construct
  target weights;
- `effective_date`: the date on which those target weights begin to apply.

Daily realized observations use `business_date`.

The explicit effective-date schedule is authoritative.
`RebalanceFrequency.WEEKLY`, `MONTHLY`, `QUARTERLY`, `SEMI_ANNUAL`, `ANNUAL`,
or `CUSTOM` is definition metadata and a defensive validation rule. The
current validator rejects multiple effective dates in the same declared
frequency period. It does not invent missing reviews or replace manually
adjusted holiday dates.

`Calendar.from_frequency(...)` is an explicit convenience for generating a
schedule. Production deployments should pass their own business-day calendar.
`Calendar.from_dates(...)` supports manual, CSV, and Excel schedules.

## Target weights and daily simulation

Construction and drift are deliberately separate:

```text
methodology or recipe
    -> target weights on effective_date

simulation
    -> daily opening weights
    -> realized return
    -> closing-weight drift
    -> next effective-date rebalance
```

The simulator treats each interval as:

```text
[effective_i, effective_i+1)
final interval = [effective_last, requested_end]
```

It can start inside an interval by locating the most recent effective target
and reconstructing or reusing its continuation state. It never treats the
requested start date as a new rebalance.

### Streaming and segmented reuse

`ResearchSimulationSpec.streaming` defaults to `True`. The simulator loads one
calendar-month market-data partition at a time, validates it, advances a
compact continuation checkpoint, and releases the partition before loading the
next one. Source partitions are independently content-addressed and can be
shared across parameter scenarios.

Closed holding periods are stored as immutable
`[effective_i, effective_i+1)` segments. The still-open final holding period is
checkpointed by calendar month. Segment identity covers target checksums,
market-data partition lineage, business-day identity, simulator and drift
implementation, runtime/dependency identity, and calculation parameters other
than `base_value`.

This supports:

- warm assembly from already verified effective-period segments;
- shorter-range reuse without a provider call; when the requested end falls
  inside a cached segment, only the exact ending checkpoint is replayed from
  the shared source Parquet rather than persisting daily constituent weights;
- extension by calculating only missing closed periods or open-tail months;
- assembly-time rebasing, so changing `base_value` does not recalculate daily
  return factors.

Existing whole-range result archives remain readable, but segmented research
writes do not depend on one monolithic cached range.

A checkpoint contains the ending business date, index and benchmark weights,
return-level state, and the prior observations required by a drift strategy.
Absolute index level is a presentation choice; cached return factors remain the
calculation identity.

### Drift strategies

Index and benchmark drift are configured independently:

```python
SimulationParams(
    index_drift=PriceReturnDrift(),
    benchmark_drift=RelativeCapitalizationDrift(),
    rebalance_phase=RebalancePhase.OPEN,
)
```

| Strategy | Calculation | Intended use |
| --- | --- | --- |
| `PriceReturnDrift` | `normalize(w_open * (1 + price_return))` | Any target-weight methodology |
| `CapitalizationDrift` | `normalize(current adjusted market_cap)` | A genuinely capitalization-weighted target |
| `RelativeCapitalizationDrift` | `normalize(w_open * cap_t / cap_t-1)` | Preserve a non-cap target while drifting by capitalization change |

`CapitalizationDrift` validates the effective-date target against normalized
capitalization and fails if it would erase a tilt. No drift strategy silently
falls back to another. `WeightDrift.PRICE_RETURN` remains available to direct
low-level simulations. Absolute `WeightDrift.MARKET_CAP` behavior is isolated
for persisted-result replay; high-level research selects an explicit strategy.

`DividendTreatment.STANDARD` and `DividendTreatment.ALTERNATIVE` are the
provider-neutral calculation variants. Dividend data provenance is configured
separately.

`RebalancePhase.OPEN` applies the target before that business date's return.
`RebalancePhase.CLOSE` applies it after the return. Non-business effective dates
follow the selected `RebalanceTiming`.

### Materialization

```python
SimulationMaterialization(
    weight_snapshots=WeightSnapshotMode.NONE,
    include_asset_returns=False,
)
```

| Mode | Materialized constituent weights |
| --- | --- |
| `NONE` | No public constituent-weight table; daily index series, rebalance events, and internal checkpoint remain |
| `REBALANCE` | Effective-date pre-rebalance, target, and end-of-day snapshots |
| `DAILY` | Full daily opening and closing holdings for detailed analysis |

`IndexSimulationResult.holdings` retains its stable schema. Explicit
rebalance lifecycle weights are available through
`rebalance_weight_snapshots`. The high-level research default is `NONE`, so
default persisted size does not grow with
`instrument_count * business_date_count`.

The simulator calculates price, gross-total, and net-total index and benchmark
returns, levels, active returns, and formal one-way rebalance turnover.
Corporate actions, currency conversion, tax treatment, and dividend inputs must
already be correct in the canonical daily market data.

### Defensive calendar checks

When a provider supplies explicit business days, ICAPA validates coverage.
Otherwise it checks actual date ordering, duplicates, effective-date mapping,
and missing observations for held instruments. These are conservative
defenses; they do not require a specific provider database schema.

## Optimization extension

The direct low-level API remains:

```text
objective callable
    + linear/nonlinear constraints
    + OptimizationProblem
    + PortfolioSolver
    -> OptimizationResult
```

`ScipySLSQPSolver` continues to support nonlinear and custom objectives. The
additive model layer provides:

- `WeightVariableSpec` for ordered instruments, warm start, investment level,
  and asymmetric per-instrument bounds;
- `PortfolioModelSpec` for compiling canonical constituent fields into
  instrument, country/industry/issuer group, numeric exposure, liquidity,
  turnover, and tracking-error constraints;
- `OptimizationModelSpec` for an inspectable objective and linear/nonlinear
  constraints;
- squared-distance, linear, and minimum-variance objective specifications;
- Entropy-Guided Multiplicative Update (EGMU) solvers for exact and elastic
  minimum-relative-entropy exposure targets, plus KL/Bregman--Dykstra
  projection across linear target, group, and instrument constraints;
- reusable group, turnover, tracking-error, and general constraint builders;
- `SolverRouter`, which checks declared capabilities and uses one explicitly
  selected backend without silent fallback;
- optional sparse `OSQPBackend` for compatible convex linear/quadratic models;
- phase-one linear feasibility analysis;
- requested, achieved, slack, binding, and violation diagnostics for every
  bound and constraint.

`GroupWeightConstraintSpec`, `FieldExposureConstraintSpec`, and
`LiquidityConstraintSpec` provide the field-level configuration for those
models. They compile into the general linear/nonlinear contracts without
changing a solver. Unsupported backend capabilities fail before solve; ICAPA
does not rewrite the model silently.

`EGMUNewtonSolver` solves exact exposure equalities in the
exposure-dimensional dual, while `EGMUElasticSolver` explicitly trades target
residual against relative entropy. `EGMUProjectionSolver` preserves the same
multiplicative KL geometry for equality, interval, and one-sided linear
constraints. `EGMUConstrainedElasticSolver` combines a soft exposure target
set with hard group and instrument constraints. These solvers require a
strictly positive prior on the active support; callers with zero benchmark
weights must restrict the problem to the positive support or choose an
explicit smoothing policy.

The public **Entropy Exposure** construction wraps those solvers as
`EntropyExposureEngine`, imported from
`icapa.portfolio_construction.engines`. Callers supply a `DataContext` whose
investable constituents already contain the benchmark weights and requested
exposure fields. Hard mode enforces exposure, group, capacity, and weight
constraints. Elastic mode softens only the exposure requests and continues to
verify the structural constraints as hard bounds.

The same public packages also expose `FactorTiltMethodology`,
`MinimumVarianceMethodology`, and `QuantileSelectionMethodology` with their
corresponding engines and configuration enums. Each methodology provides both
direct `execute(DataContext)` execution and a provider-aware `to_recipe()`
adapter.

Minimum-variance research uses `ReturnWindowSpec` with any public
`CovarianceEstimator`:

- `SampleCovarianceEstimator` provides pairwise or complete-case sample
  covariance with an explicit numerical ridge and PSD policy;
- `ShrinkageCovarianceEstimator` applies an explicit deterministic intensity
  toward either the sample diagonal or a scaled identity target;
- `FactorCovarianceEstimator` builds a deterministic statistical factor model
  from principal covariance components and diagonal specific risk.

All three return the same validated `CovarianceEstimate` contract. The
estimator type, immutable configuration, resolved point-in-time window, input
returns, covariance output, and diagnostics can therefore be identified and
cached by the same recipe stage. Shrinkage and factor estimators add no runtime
dependency beyond NumPy and Pandas.

Install the optional sparse backend with:

```bash
python -m pip install -e '.[qp]'
```

## Analytics plugins

The existing `AnalyticsEngine` and `AnalyticsResult` remain unchanged.
`AnalyticsPluginRunner` wraps that parity result and adds versioned,
read-only research modules.

`AnalyticsWorkspaceCache` is the reusable storage contract behind high-level
analytics execution. It can also wrap an explicit zero-argument analytics
calculation with `OFF`, `REUSE`, `REFRESH`, or `READ_ONLY` behavior. Its
outcome includes the cache source and immutable artifact references.

`AnalyticsSpec.standard_research()` includes:

- review validation and v1 summary statistics;
- entrants, exits, and membership stability;
- selection and exclusion reasons when supplied;
- weight-change contributors and turnover decomposition;
- requested versus achieved targets;
- constraint binding, slack, and violations;
- factor and signal exposure;
- liquidity and capacity coverage;
- calendar-period and rolling performance/risk;
- drawdown duration and recovery;
- data coverage, missingness, and point-in-time freshness;
- multi-period attribution when required inputs are available;
- methodology diagnostics.

Optional unavailable inputs produce structured skip diagnostics by default.
They do not cause the plugin runner to invent data. `ReturnSeries` is explicit;
the standard default is `NET_TOTAL`. Adding another return column never changes
the analysis basis silently.

Analytics consumes completed backtest and simulation results. It does not load
providers, change target weights, or render reports.

## Baseline and candidate comparison

```python
candidate = workspace.run(candidate_spec)
comparison = workspace.compare(
    baseline=baseline,
    candidates=[candidate],
)
```

Comparison checks lineage compatibility, aligns common effective and business
dates, and aligns the union of constituents with absent weights set to zero.
It reports:

- parameter and lineage differences;
- review coverage;
- constituent additions and removals;
- weight differences;
- exposure, performance, turnover, and validation differences.

Comparison uses completed runs and does not rerun construction or simulation.

## Reporting

The Excel v1 renderer, template, worksheets, fields, and integrity checks remain
available. The v2 writer creates one atomic directory:

```text
<bundle-id>/
  report.xlsx
  summary.json
  manifest.json
  checksums.json
  tables/*.parquet
```

The default v2 bundle keeps complete tabular outputs in Parquet and appends
research sheets to the v1 workbook, including Research Summary, Calendar
Periods, Rolling Metrics, Constituent Changes, Constraint Diagnostics, Data
Coverage, Run Manifest, and optional Comparison sheets.

Tables beyond Excel's row limit are split deterministically across worksheets.
Parquet remains the complete machine-readable source. The writer blocks formula
injection and external workbook links, filters sensitive manifest fields, and
never serializes credentials or connection details.

```python
comparison = workspace.compare(baseline, [candidate])
bundle = workspace.write_report(
    candidate,
    comparison=comparison,
)
```

## Stable lower-level contracts

The high-level research workflow composes, rather than replaces, these
lower-level contracts:

- direct `Backtester` and `WorkspaceStore`;
- direct `IndexSimulator`, drift configuration, daily holdings, and simulation
  result fields;
- existing JSON review and simulation cache reading;
- `AnalyticsEngine`, `AnalyticsResult`, and `analyze_backtest`;
- Excel v1 template and `write_index_research_report`;
- existing `execute(DataContext)` methodologies through direct use or
  `IndexRecipe.from_methodology`.

`ResearchWorkspace` composes these interfaces. A custom methodology may remain
opaque to the workflow as long as it emits the canonical target-weight
contract.

## Public extension boundary

The public package contains provider contracts, recipes, optimization
interfaces, all repository methodology and engine implementations, workspaces,
simulation, analytics, comparison, and reporting. Deployments may add external
custom methodologies, but the canonical implementations are included in the
public source and wheel distributions.

For data fields, date semantics, provider capabilities, and adapter design, see
[the data-loading guide](data_loading_guide.md). For reproducible performance
measurement, see [the scale-suite guide](performance_scale_suite.md).
