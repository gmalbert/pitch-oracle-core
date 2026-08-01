# pitch-oracle-core

Shared, versioned league-agnostic package for Pitch Oracle consumers.

The package provides configuration contracts, phase-aware competition logic, optional
data-source handling, a deterministic shot-based xG proxy, and a provider-neutral odds
adapter. It also provides shared Poisson goal-market calculations for expected goals,
over/under totals, BTTS, and most-likely scorelines. League repositories should pin a semantic release tag and provide only their
configuration, data files, league-specific adapters, and Streamlit entrypoint.

The migrated legacy-compatible modules at the repository root and under `models/` are
the extracted EPL implementation. New league repositories should call
`pitch_oracle_core.app_factory.run(config)` and progressively replace provider-specific
legacy scripts with adapters that accept `LeagueConfig`.

```python
from pitch_oracle_core import get_league_config

config = get_league_config("eredivisie")
```
