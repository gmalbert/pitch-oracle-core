# 17 — Multi-Consumer / League-Neutral Patterns

Pitch Oracle is not one app; it is a **core package** plus a fleet of
**consumer repositories** (Belgium, Netherlands, Scotland, Turkey, EPL,
Portugal, …). The `penaltyblog` integration must not collapse into an
EPL-centric design. This roadmap spells out the patterns that keep
every consumer on the same statistical engine without copying code.

## 17.1 What the core owns

The core package (`pitch-oracle-core`) owns:

- `pitch_oracle_core/team_mappings.py` — canonical team identity,
  shared by every consumer.
- `pitch_oracle_core/markets.py` — the `FootballProbabilityGrid`
  wrapper; consumers get the helper functions for free.
- `pitch_oracle_core/models/registry.py` — the model registry and
  promotion gate; every consumer's Model Lab reads the same artifact.
- `pitch_oracle_core/ui/components/` and `ui/shells/` — the visual
  components; consumers get the same look without a CSS fork.
- `scripts/observability/` and the observability contracts in
  [15](15-production-observability.md).
- `scripts/eval/` and the metrics in [14](14-metrics-evaluation.md).
- The manifest contract (currently v3) — every consumer publishes
  artifacts under the same keys.

## 17.2 What consumers own

A consumer repository owns:

- `leagues/<key>.yaml` — the league configuration.
- `data_files/raw/`, `data_files/entities/` — local data.
- `data_files/artifacts/` — the published artifacts (must be
  manifest-compatible).
- A pinned `core_ref` in `requirements.txt` and the reusable workflow.
- The Streamlit entrypoint (`app.py` or `Home.py`).
- Optional: `scripts/ingest/` for providers not in the core.

A consumer **never** owns:

- A copy of any goal model.
- A copy of the `FootballProbabilityGrid` helper.
- A copy of the model registry.
- A copy of the Streamlit components.

If a consumer is tempted to do one of these, that is a signal that
the core has a missing module; add the module and remove the copy.

## 17.3 Capability flags

The manifest carries a `capabilities` block. Every consumer declares
the capabilities it has. Pages consult the flags; providers stay
optional.

```json
{
  "league": "eredivisie",
  "capabilities": {
    "odds":            true,
    "events":          true,
    "squad":           true,
    "referee":         true,
    "weather":         true,
    "manager_history": false,
    "fpl":             false
  }
}
```

A consumer without `odds` does not show the Market Lab tab; a consumer
without `events` does not show the Expected Threat tab; a consumer
without `fpl` does not show the FPL tab. The behaviour is the same
across consumers, including the wording of the "not available"
message.

## 17.4 Provider adapters

Every external data source is behind a `ProviderAdapter` interface:

```python
# pitch_oracle_core/providers/base.py
class ProviderAdapter(Protocol):
    name: str
    capability: str
    def fetch(self, *, since: datetime | None = None) -> pd.DataFrame: ...
    def is_available(self) -> bool: ...
```

`ClubElo`, `Understat`, `FootballData`, and `FBRef` from
`penaltyblog` are *one* implementation. The other implementations
might wrap StatsBomb open data, ESPN, or a private feed — they all
expose the same interface.

The core ships with the penaltyblog implementations. Consumers can
register their own (e.g. for Opta) without changing the core:

```python
# consumer: belgium-soccer/providers/opta.py
from pitch_oracle_core.providers.base import ProviderAdapter

class OptaAdapter(ProviderAdapter):
    name = "opta"
    capability = "events"
    def fetch(self, *, since=None): ...
    def is_available(self) -> bool:
        return os.environ.get("OPTA_KEY") is not None
```

The provider is then registered in the consumer's
`providers.yaml`:

```yaml
providers:
  - name: opta
    module: belgium_soccer.providers.opta.OptaAdapter
    capabilities: [events]
```

## 17.5 Cross-league comparison

A separate index app reads all consumers' published artifacts and
builds a cross-league view (F35 in the product-expansion catalog).
Each consumer publishes a `cross_league_metrics.parquet` artifact:

```python
# scripts/cross_league/build.py
import pandas as pd, json
from pathlib import Path

def build():
    out = {
        "league":          load_manifest()["league"],
        "avg_total_goals": avg_total_goals(),
        "home_advantage":  home_advantage(),
        "draw_rate":       draw_rate(),
        "competitive_balance": elo_dispersion(),
        "calibration_ece": rolling_ece(),
    }
    Path("data_files/artifacts/cross_league_metrics.json").write_text(
        json.dumps(out, indent=2)
    )
```

The index app reads the same artifact from every consumer repo and
renders the comparison. It is a separate repository; no consumer
repo references it.

## 17.6 Shared golden test data

Every consumer runs the same golden-master test fixtures (see
[18](18-test-data-reproducibility.md)). The fixtures are committed to
the core package under `tests/fixtures/golden/` and are versioned
together with the core release.

Consumers can opt out of a fixture (e.g. EPL skips the "no-promoted-
teams" fixture) but cannot modify the fixture itself. If a fixture is
wrong, the fix is in core and every consumer inherits the corrected
version.

## 17.7 League configuration

The league configuration is the only place a consumer expresses
something league-specific. The current `leagues/<key>.yaml` already
covers:

- Season boundaries (calendar-year vs split-year).
- Competition phases (regular / split / playoff).
- Provider capabilities.
- Team-name aliases.

Add to it:

- **Promotion / relegation** — the rule engine reads this section.
- **Title / Europe / relegation thresholds** — named outcome labels
  (F28).
- **Default scoreline draw rate prior** — used by the Bayesian
  hierarchical model when a league has too few fixtures.
- **Streamlit default theme override** — only if a consumer has a
  legitimate reason (e.g. Belgium red / black / yellow).

Everything else is core-owned.

## 17.8 The thin-consumer contract

A consumer PR can change exactly three things:

1. **The league configuration** (`leagues/<key>.yaml`).
2. **The data files** (`data_files/raw/`, `data_files/entities/`,
   `data_files/artifacts/`).
3. **The Streamlit entrypoint** (`app.py`, plus optional league-
   specific pages).

Anything else is a signal that the change belongs in the core. A
reviewer should be able to verify the thin-consumer contract by
running `diff` between two consumer repos and seeing only those three
categories.

## 17.9 Consumer verification

The reusable consumer workflow calls a `verify_consumer.py` script
at the end:

```python
# pitch_oracle_core/consumer/verify.py
def verify_consumer(root: Path):
    assert_manifest_v3(root / "data_files/cache_manifest.json")
    assert_no_legacy_artifacts(root)
    assert_pages_under_pitch_oracle_core(root)
    assert_team_mappings_complete(root)
    assert_capability_flags_consistent(root)
    assert_golden_fixtures_pass(root)
```

The script enforces the thin-consumer contract mechanically. A
violation blocks the PR.

## 17.10 Pitch Oracle integration

**Touches**

- New `pitch_oracle_core/providers/` directory.
- New `pitch_oracle_core/consumer/verify.py`.
- `leagues/<key>.yaml` extended with promotion / relegation /
  thresholds.
- New `scripts/cross_league/build.py` per consumer.
- `tests/fixtures/golden/` per consumer (subset of the core fixtures).

**Does not touch**

- Any of the goal models, ratings, betting utilities.

**Migration cost**

- One engineer, ~3 days including the verify script and the new
  `leagues/<key>.yaml` keys.
