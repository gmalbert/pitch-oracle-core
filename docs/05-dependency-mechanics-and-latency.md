# How the Shared Package Actually Works (Mechanics + Latency)

This doc answers two follow-up questions: how `pitch-oracle-core` actually gets used
by each league repo (it's installed, not copied), and whether that approach costs
anything in speed.

## Installed, not copied

`pitch-oracle-core` is a normal Python package living in its own repo. Each league
repo pulls it in via `requirements.txt`, the same way it already pulls in `pandas`
or `xgboost`:

```
# eredivisie-predictions/requirements.txt
git+https://github.com/gmalbert/pitch-oracle-core.git@v1.0.0
streamlit
pandas
...
```

When Streamlit Community Cloud builds the app, it `pip install`s straight from that
GitHub repo at the pinned tag. The code isn't duplicated into
`eredivisie-predictions` — it's imported at runtime, same as any other library:

```python
# eredivisie-predictions/entrypoint.py
from pitch_oracle_core import train_models, evaluate_poisson, app_shell
from config import LEAGUE_CONFIG

app_shell.run(LEAGUE_CONFIG)
```

**Why this matters vs. copy/paste:** a bug fix or model improvement in
`pitch-oracle-core` gets fixed once, tagged as a new version (`v1.1.0`), and then
each league repo picks it up with a one-line `requirements.txt` bump — not a manual
re-port into 5 drifted copies of the same file.

**Versioning discipline required:** pin each league repo to a specific tag
(`@v1.0.0`), not `@main`, or a work-in-progress change to core could break all 5
apps the next time Streamlit Cloud rebuilds them. Bump major versions deliberately
when core's interface changes. `premier-league` itself becomes consumer #1 of the
package — if its existing test suite still passes once it's importing from core, the
extraction didn't break anything.

## Does this cost anything in speed?

No runtime/latency difference. Once installed, `pitch-oracle-core` behaves
identically to any other installed library — importing it and calling, e.g.,
`train_models.train_ensemble()` is a local function call, same as calling into
`pandas` or `xgboost`. There's no network round-trip during normal app operation;
the git dependency only matters at **install time**, not request time.

### Where it does show up: build/deploy time, not the live app

- `pip install git+https://github.com/...@v1.0.0` does a git clone of that tag
  rather than pulling a prebuilt wheel from PyPI — a few extra seconds, not
  minutes, for a repo this size.
- Streamlit Community Cloud spins down inactive apps and rebuilds the environment
  on wake-up. Every cold start re-clones and reinstalls `pitch-oracle-core` along
  with the rest of `requirements.txt`. This is additive to existing cold-start
  time — but `premier-league` already pulls in `torch` for the NN/LSTM models,
  which dwarfs the cost of one more git dependency. Not noticeable in practice.
- Each of the 5 league apps runs in its own isolated Streamlit Cloud environment,
  so there's no shared build cache across them — but that's true whether the code
  is duplicated or shared, so it isn't a cost specific to this approach.

### If it ever actually becomes a problem

- Pin to a commit SHA or lightweight tag rather than a branch so pip doesn't have
  to resolve refs — marginal, but free.
- If cold-start time across 5+ apps becomes genuinely annoying, the fix is a paid
  Streamlit Cloud tier or always-on hosting, not abandoning the shared-package
  structure.
- For zero git-clone overhead, `pitch-oracle-core` could be built into an actual
  wheel and hosted on a package index (even a simple GitHub Releases-based one) —
  real infrastructure for a marginal gain, not worth it until there's observed
  pain rather than hypothetical pain.

**Net tradeoff:** a few extra seconds per cold start, in exchange for never having
to manually re-port a bug fix across 5 repos.
