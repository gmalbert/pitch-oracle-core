# 18 — Test Data, Reproducibility & Golden Masters

The penaltyblog roadmaps each add new code paths. Without a
disciplined test-data strategy, the new tests become flaky,
non-deterministic, and hard to maintain. This roadmap defines the
golden-master pattern, the property-based test inventory, the
regression suite, and the reproducibility rules.

## 18.1 Golden-master fixtures

A golden master is a small, fixed dataset plus the expected output
of every code path that consumes it. For the penaltyblog integration
the master fixtures are:

```text
tests/fixtures/golden/
  epl_2018_2019.parquet        # 380 matches, full season
  epl_2018_2019_shots.parquet  # 12k events from open StatsBomb
  epl_2018_2019_xg.parquet     # per-shot xG, from the same archive
  epl_alias_mapping.json       # 22 team name variants
  model_registry.golden.json   # expected registry after a canonical fit
  walk_forward.golden.parquet  # expected walk-forward metrics
  reliability.golden.parquet   # expected reliability table
  severity.golden.json         # expected observability severity
```

The fixtures are committed to the core package and **pinned by SHA**.
Any change to a fixture is a deliberate, reviewable event.

## 18.2 Golden-master test pattern

```python
# tests/test_dixon_coles_golden.py
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from penaltyblog.models import DixonColesGoalModel

FIX = Path("tests/fixtures/golden")
TOLERANCE = 1e-6

def _hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()

def test_dixon_coles_fit_is_deterministic():
    df = pd.read_parquet(FIX / "epl_2018_2019.parquet")
    model = DixonColesGoalModel(
        df["goals_home"], df["goals_away"],
        df["team_home"],   df["team_away"],
    ).fit()
    params = model.get_params()
    assert _hash(params) == "f4a0d1...expected_sha..."

def test_dixon_coles_grid_is_deterministic():
    df = pd.read_parquet(FIX / "epl_2018_2019.parquet")
    model = DixonColesGoalModel(
        df["goals_home"], df["goals_away"],
        df["team_home"],   df["team_away"],
    ).fit()
    g = model.predict("Arsenal", "Chelsea", max_goals=10)
    assert abs(g.home_win - 0.470918) < TOLERANCE
    assert abs(g.draw     - 0.246327) < TOLERANCE
    assert abs(g.away_win - 0.282755) < TOLERANCE
```

The deterministic grid test is the **strongest** contract: if a
penaltyblog release drifts a single parameter, the test fails and the
PR is blocked.

## 18.3 Property-based tests

Beyond the golden masters, every code path has a property-based test
that runs `hypothesis` over a wide range of inputs:

```python
# tests/test_grid_properties.py
import numpy as np
from hypothesis import given, strategies as st
from penaltyblog.models.football_probability_grid import create_dixon_coles_grid

@given(
    lam_h=st.floats(min_value=0.1, max_value=5.0),
    lam_a=st.floats(min_value=0.1, max_value=5.0),
    rho=st.floats(min_value=-0.5, max_value=0.5),
    max_goals=st.integers(min_value=4, max_value=12),
)
def test_grid_probabilities_sum_to_one(lam_h, lam_a, rho, max_goals):
    g = create_dixon_coles_grid(lam_h, lam_a, rho=rho, max_goals=max_goals)
    assert abs(g.home_win + g.draw + g.away_win - 1.0) < 1e-9

@given(
    lam_h=st.floats(min_value=0.1, max_value=5.0),
    lam_a=st.floats(min_value=0.1, max_value=5.0),
    rho=st.floats(min_value=-0.5, max_value=0.5),
    max_goals=st.integers(min_value=4, max_value=12),
)
def test_grid_btts_plus_no_btts_equals_one(lam_h, lam_a, rho, max_goals):
    g = create_dixon_coles_grid(lam_h, lam_a, rho=rho, max_goals=max_goals)
    assert abs(g.btts_yes + g.btts_no - 1.0) < 1e-9

@given(
    lam_h=st.floats(min_value=0.1, max_value=5.0),
    lam_a=st.floats(min_value=0.1, max_value=5.0),
    rho=st.floats(min_value=-0.5, max_value=0.5),
    max_goals=st.integers(min_value=4, max_value=12),
)
def test_grid_totals_sum_to_one(lam_h, lam_a, rho, max_goals):
    g = create_dixon_coles_grid(lam_h, lam_a, rho=rho, max_goals=max_goals)
    u, _, o = g.totals(2.5)
    assert abs(u + o - 1.0) < 1e-9
```

The property-based inventory:

| Property | Why it matters |
|----------|----------------|
| 1X2 sums to 1 | A grid that doesn't sum to 1 is a serious bug |
| BTTS yes + no = 1 | Same |
| Totals (over + under) = 1 | Same |
| Marginal > 0 | Empty cells must be impossible |
| Marginals sum to 1 | Per-team goal distribution |
| Asian handicap win + push + lose = 1 | Same |
| `home_win` symmetric in `lam_h, lam_a` if HFA=0 | Sanity |
| Monotonic in `lam_h`: as `lam_h` increases, `home_win` increases | Sanity |
| `Elo.predict` is symmetric in swap | Elo invariant |
| Kelly stake is 0 when `true_prob == 1/odds` | Edge case |
| Kelly stake is 0 when `true_prob < 1/odds` | Negative edge case |

## 18.4 Regression tests

Three regression suites run nightly in the consumer workflow:

### Walk-forward regression

```python
# tests/test_walk_forward_regression.py
import pandas as pd
from pathlib import Path

def test_walk_forward_metrics_have_not_regressed():
    cur = pd.read_parquet("data_files/eval/walk_forward_dc.parquet")
    golden = pd.read_parquet(
        Path("tests/fixtures/golden/walk_forward.golden.parquet")
    )
    assert cur["brier"].mean() < golden["brier"].mean() * 1.05
    assert cur["log_loss"].mean() < golden["log_loss"].mean() * 1.05
    assert cur["rps"].mean() < golden["rps"].mean() * 1.05
```

A 5% headroom is allowed for legitimate noise; a 10% regression
fails the test.

### Calibration regression

```python
# tests/test_calibration_regression.py
import pandas as pd
from pathlib import Path
from penaltyblog.metrics import calibration_error

def test_ece_home_within_tolerance():
    cur = pd.read_parquet("data_files/eval/reliability.parquet")
    golden = pd.read_parquet(
        Path("tests/fixtures/golden/reliability.golden.parquet")
    )
    cur_ece, _ = calibration_error(
        cur["result_H"], cur["p_home"], n_bins=10
    )
    gold_ece, _ = calibration_error(
        golden["result_H"], golden["p_home"], n_bins=10
    )
    assert cur_ece < gold_ece + 0.005
```

### Observability contract

```python
# tests/test_observability_contract.py
import json
from pathlib import Path

def test_severity_artifact_shape():
    s = json.loads(Path("data_files/observability/severity.json").read_text())
    assert "overall" in s
    assert s["overall"] in {"ok", "warning", "critical"}
    assert "reports" in s
    expected = {"input_drift", "output_drift", "calibration_drift",
                "coverage_drift", "operational_drift"}
    assert expected <= set(s["reports"].keys())
```

## 18.5 The reproducibility rules

1. **Pinned `penaltyblog` version.** `requirements.txt` carries
   `penaltyblog==1.x.y`. A version bump is a deliberate, reviewable
   event.
2. **Pinned `numpy` and `scipy` versions.** The Cython build is
   sensitive to the NumPy ABI. `pip check` is a required CI step.
3. **Hashed training sets.** Every persisted model carries the SHA
   of the training parquet; the manifest carries the same hash.
4. **Seeded random state.** `np.random.seed(42)` at the top of every
   Bayesian / simulation script.
5. **No live network in tests.** All tests use local fixtures.
6. **No live filesystem outside `data_files/`.** All paths go
   through a single helper that resolves relative to the project
   root.

## 18.6 The "fresh-fixture" workflow

When the golden masters need to be updated — because a new
penaltyblog release is intentional, or because a new league is
added — the process is:

```bash
# 1. Run the full pipeline against the proposed release
pip install penaltyblog==NEW_VERSION
python -m pitch_oracle_core.pipelines.build_consumer --league belgium
# 2. Compute the new golden hashes
python scripts/tests/refresh_golden.py --output PR_BODY.md
# 3. Open a PR titled "chore: refresh golden fixtures for penaltyblog NEW_VERSION"
# 4. The PR must include a manual review of every diff
```

The `refresh_golden.py` script produces a markdown body that
includes:

- The new SHA-256 of every changed fixture.
- The 5 worst regressions and 5 best improvements.
- A "manual review required" checklist.

A reviewer who doesn't tick the checklist cannot merge.

## 18.7 Test running rules

- Unit tests are < 1 second each.
- Golden-master tests are < 5 seconds each.
- Property-based tests are < 30 seconds each.
- Walk-forward regressions are < 60 seconds each.
- The full suite is < 5 minutes on a modern laptop.

If a test exceeds its budget, it is split or moved to a slower tier
(`tests/slow/`). The default `pytest` run never enters `slow/`.

## 18.8 CI matrix

```yaml
# .github/workflows/test.yml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python-version: ["3.12", "3.13"]
```

The matrix is the same for the core and every consumer. A failure on
one OS or one Python version blocks the release.

## 18.9 Pitch Oracle integration

**Touches**

- New `tests/fixtures/golden/` directory.
- 21 new test files (per roadmap 13).
- 3 new regression suites (this roadmap).
- 10 new property-based tests (this roadmap).
- `scripts/tests/refresh_golden.py`.

**Does not touch**

- The modelling code, the UI, the manifest.

**Migration cost**

- One engineer, ~2 days for the fixtures + ~3 days for the
  regression and property-based suites.
