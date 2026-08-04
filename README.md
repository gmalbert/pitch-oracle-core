# pitch-oracle-core

Shared, versioned league-agnostic package for Pitch Oracle consumers. The current
consumer contract is **1.3.12** and requires Python 3.12 or newer.

The package provides configuration contracts, phase-aware competition logic, optional
data-source handling, a deterministic shot-based xG proxy, and a provider-neutral odds
adapter. It also provides shared Poisson goal-market calculations for expected goals,
over/under totals, BTTS, and most-likely scorelines. League repositories should pin a semantic release tag and provide only their
configuration, data files, league-specific adapters, and Streamlit entrypoint.

## Consumer installation

Pin an immutable release tag and install the consumer dependencies:

```text
pitch-oracle-core[consumer] @ git+https://github.com/gmalbert/pitch-oracle-core.git@v1.3.12
```

Optional neural models additionally require the `neural` extra.

The package-owned Streamlit navigation and pages are league-neutral. Migrated
root modules remain compatibility entrypoints for the artifact pipeline, but they
resolve the active league through `PITCH_ORACLE_LEAGUE`; provider-specific work is
gated by `LeagueConfig.sources`.

```python
from pitch_oracle_core import get_league_config

config = get_league_config("eredivisie")
```

The supported runtime is Python 3.12 or newer. The `consumer` extra contains the
training, diagnostics, data-source, and app dependencies required by the standard
artifact workflow.

## Required artifact pipeline

Consumers must build artifacts in this order using the same pinned core version:

1. prepare historical point-in-time features;
2. run `python -m train_models`;
3. run `python -m precompute_database`;
4. generate `upcoming_predictions.csv` with `FeatureContract`,
   `build_upcoming_feature_matrix`, and `build_prediction_frame`;
5. run `python -m build_cache_manifest`;
6. run consumer tests.

The manifest and preprocessed artifact carry core, league, and feature-policy
versions. Runtime startup rejects old artifacts, mismatched model widths, missing
prediction assessment fields, caches built by another core version, and artifacts
created for a different league.

## Starting a league consumer

Generate a repository from the maintained template:

```bash
python scripts/bootstrap_consumer.py eredivisie ..
```

The generator selects a consumer-ready built-in league, specializes the config and
workflows, creates a country-based repository name such as `netherlands-soccer`,
and refuses to overwrite an existing path. A league is consumer-ready
only when both baseline historical and fixture identifiers are configured. Keep the core tag identical in both
requirements files and in the reusable workflow call. The template deliberately
contains no EPL code or generated artifacts.

The reusable workflow requires `league_key`, `core_ref`, data preparation,
training, and prediction commands. It installs the pinned core, runs `pip check`,
builds the complete cache, validates it strictly, runs consumer tests, and commits
the generated artifacts once.

See [the 1.3 consumer migration guide](docs/consumer-migration-1.3.md) before
upgrading an existing league repository.

For a new repository, follow the complete
[new consumer repository runbook](docs/new-consumer-repository.md). It includes
the recommended pilot, source scope, GitHub settings, first production build,
browser validation, acceptance gates, and upgrade policy.
