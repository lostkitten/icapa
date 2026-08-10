# ICAPA Scale and Performance Suite

## Purpose

The independent scale suite measures the public research architecture without
slowing the normal correctness tests. It uses deterministic generated data and
does not require a database, credentials, or private methodology.

The suite is opt-in. A normal `pytest -q` run collects scale tests but skips
them.

## Profiles

The quick profile validates the benchmark harness and the same execution paths
with small inputs:

```bash
pytest -q tests/scale --run-scale --scale-profile=quick -s
```

The full profile runs the approved research matrix:

- 5,000 and 10,000 instruments;
- 10-year and 20-year histories;
- monthly, quarterly, and semiannual review schedules;
- 20 construction parameter scenarios;
- cold, exact-warm, shorter-range, and extended-range simulation runs.

```bash
pytest -q tests/scale --run-scale --scale-profile=full \
  --scale-output=/tmp/icapa-scale.jsonl -s
```

Do not use process-level parallelism for the full profile unless the machine
has enough memory for multiple 10,000-instrument partitions. The streaming test
already measures the intended single-process production path.

## Benchmarks

### Streaming simulation matrix

This benchmark constructs deterministic effective-date target weights and
streams daily market data in calendar-month partitions. It verifies that:

- daily numeric outputs remain finite;
- default constituent holdings and asset-return tables are not materialized;
- the largest provider response is bounded by one calendar month;
- persisted bytes are measured relative to instrument-business-day count.

The full profile executes every combination of instrument count, history
length, and rebalance frequency.

### Immutable segmented cache reuse

This benchmark executes one monthly index definition four ways:

1. cold calculation;
2. identical warm request;
3. shorter request inside the cached range;
4. extension beyond the cached range.

It asserts that warm and shorter requests make no market-data provider call,
and that the extension loads only missing immutable effective-date segments or
calendar-month open-tail checkpoints. If a shorter end date falls inside a
cached segment, the daily series uses the cached prefix and the exact ending
constituent state is rebuilt from the single shared Parquet source partition.
Daily constituent weights are not persisted merely to support arbitrary
truncation. The existing whole-range prefix cache is an archive-reading path, not
the v2 persistence model.

### Parameter-scenario stage reuse

This benchmark runs 20 recipe variants over an unchanged 10,000-instrument
market-data partition and feature artifact. It verifies that:

- the shared provider partition is loaded once and then read from the
  scenario-independent source cache;
- the shared feature stage executes once;
- each unique parameterized weighting stage executes once;
- rerunning an existing scenario reuses both stages;
- unchanged upstream data/features are not duplicated per scenario.

## Recorded metrics

Every completed benchmark emits one line prefixed with
`ICAPA_SCALE_METRIC`. When `--scale-output` is supplied, the same canonical
record is appended as JSON Lines. Fields include:

- requested and actual scale;
- rebalance frequency and scenario count;
- wall-clock seconds;
- process peak RSS in bytes;
- total workspace disk delta in bytes, including shared source artifacts;
- provider call and row counts;
- largest source partition;
- cache-hit ratio or per-run provider-call counts;
- persisted bytes per instrument-business-day where applicable.

Peak RSS is the operating system's process high-water mark, so later test cases
may inherit a higher value from earlier cases. Compare runs on the same machine,
Python environment, test order, and process model.

## Reproducible comparison

Record the following beside the JSON Lines result:

- Git commit and dirty-worktree state;
- operating system, CPU, and memory;
- Python and dependency-lock versions;
- whether the optional OSQP dependency is installed;
- local or network filesystem;
- command line and selected scale profile.

Run cold and warm measurements on the same filesystem. Do not clear the named
test workspace between phases within one benchmark; the test uses a temporary,
isolated workspace and cleans it through pytest's temporary-directory lifecycle.

The scale suite records measurements and verifies architectural invariants. It
does not impose universal wall-time or memory thresholds because those values
depend on hardware and storage. Deployment-specific CI can parse the JSON Lines
output and apply regression budgets appropriate to its runner.
