# ICAPA

<img src="assets/icapa.png" alt="ICAPA" width="360">

ICAPA is provider-neutral infrastructure for index research, simulation,
portfolio analytics, and reporting. Database access is isolated behind explicit
provider interfaces, while target weights are supplied by an external
weight-production component:

```text
Provider
  -> canonical contract
  -> external weight producer
  -> cached target weights
  -> simulation
  -> analytics
  -> report
```

GIS and FactSet are SQL Server integration placeholders. The ICE Data Indices
Library and Snowflake are also unconfigured placeholders. CSV and Excel files
are supported for controlled ad-hoc inputs. The package contains no default
credentials, queries, schemas, production datasets, or implicit provider
fallback.

## Structure

| Package | Responsibility |
| --- | --- |
| `data_sources` | Provider interfaces, registry, canonical contracts, database placeholders, and file loading |
| `portfolio_construction` | Generic optimisation contracts, objectives, constraints, solver interfaces, and empty private-extension placeholders |
| `backtesting` | Review calendars, external weight-producer orchestration, cached target weights, and stateful daily index simulation |
| `workspace` | Fixed-root memory and disk artifacts with fingerprints, checksums, and cache policies |
| `helpers` | Explicit underlying-profile and identifier-mapping configuration |
| `analytics` | Review statistics, exposures, performance, drawdown, turnover, and optional attribution |
| `reporting` | Safe report payloads, a fixed Excel template, and workspace-confined rendering |
| `tools` | The lightweight `DataContext`, enums, and deterministic return utilities |

## Public extension boundary

The public repository does not distribute private portfolio-construction
implementations. The private extension folders under
`portfolio_construction/methodologies/` and
`portfolio_construction/rules/engines/` intentionally contain only `.gitkeep`.
Deployments provide an external weight producer that consumes canonical data
and writes `index_weight`. The public core then validates, caches, simulates,
analyses, and reports those weights.

Named workspaces are stored under `~/.icapa/workspaces` by default. The root can
be changed only with `ICAPA_WORKSPACE_ROOT`. Overlapping reviews are reused when
the same index definition is rerun for a shorter or longer date range; only
missing reviews are calculated.

The [data-loading guide](docs/data_loading_guide.md) defines canonical fields,
date semantics, provider capabilities, and integration parameters. The
[index research workflow](docs/index_research_workflow.md) explains workspace
retention, configurable underlying mappings, review-weight reuse, daily
simulation, analytics, optimisation extension points, and report generation.

## Installation

ICAPA requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Offline demonstrations and smoke tests

The public-core tests run offline with generated data:

```bash
pytest -q
```

Integration code should test its external weight producer separately, then run
the public pipeline against synthetic canonical inputs before connecting
controlled data.

## License

The public ICAPA core is available under the MIT License.
