# 03 — Goal Models (Poisson, Dixon-Coles, and friends)

`penaltyblog.models` is the heart of the library. It exposes eight
goal-expectancy models, every one of them built on a Cython core:

| Class                                  | Parameters                                  | When to use                              |
|----------------------------------------|---------------------------------------------|------------------------------------------|
| `PoissonGoalsModel`                    | attack × N + defense × N + home_advantage    | First-cut baseline                       |
| `DixonColesGoalModel`                  | + low-score correlation `rho`               | Default for 1X2 + draws                  |
| `BivariatePoissonGoalModel`            | adds a shared latent goal term              | Tournaments with extreme mismatches      |
| `NegativeBinomialGoalModel`            | adds overdispersion                         | When Poisson under-fits goal variance    |
| `ZeroInflatedPoissonGoalsModel`        | mixes in a structural zero probability      | When 0-0s are systematically over-fit    |
| `WeibullCopulaGoalsModel`              | Weibull marginals + Gaussian copula         | Tail-sensitive markets (longshots)       |
| `BayesianGoalModel`                    | full posterior over all parameters          | Uncertainty-aware pricing (see 04)       |
| `HierarchicalBayesianGoalModel`        | league-level prior + team-level posterior   | Sparse leagues, promotion candidates      |

All eight share the same public surface:

```python
model = ModelClass(goals_home, goals_away, teams_home, teams_away,
                   weights=weights, neutral_venue=neutral_venue)
model.fit()
grid = model.predict("Arsenal", "Chelsea", max_goals=15)
```

That `grid` is a `FootballProbabilityGrid` (see [11-probability-grid.md](11-probability-grid.md)),
which gives you 30+ derived markets for free.

The Pitch Oracle scripts that benefit are:

- `train_models.py` — replace the in-house Poisson training loop with
  `PoissonGoalsModel` / `DixonColesGoalModel`.
- `evaluate_poisson.py` — keep the diagnostics; replace the model class.
- `optimize_model.py` — replace the in-house Bayesian-style search with
  `DixonColesGoalModel.fit(minimizer_options={"maxiter": 5000})` plus a
  proper L-BFGS-B run.
- `benchmark_hyperparameters.py` — same change, plus the new
  `NegativeBinomialGoalModel` for over-dispersed leagues.
- `compare_model_features.py` — re-run the same comparison with the new
  models added.

## 3.1 Drop-in replacement for the existing Poisson training

### Before

```python
# train_models.py — current implementation
import numpy as np
from scipy.optimize import minimize

def poisson_loglik(params, gh, ga, w):
    attack  = params[:n_teams]
    defense = params[n_teams:2*n_teams]
    hfa     = params[-1]
    ll = 0.0
    for i in range(len(gh)):
        lam_h = np.exp(attack[h_idx[i]] + defense[a_idx[i]] + hfa)
        lam_a = np.exp(attack[a_idx[i]] + defense[h_idx[i]])
        ll += w[i] * (gh[i]*np.log(lam_h) - lam_h + ga[i]*np.log(lam_a) - lam_a)
    return -ll
```

This is a textbook re-implementation. It is correct, but slow (pure
Python loop), missing constraints (no sum-to-N attack), and does not
expose the optimiser interface the rest of the repo needs.

### After

```python
# train_models.py — penaltyblog
from penaltyblog.models import PoissonGoalsModel, DixonColesGoalModel
import pandas as pd

df = pd.read_parquet("data_files/features/training_set.parquet")
weights = df["time_decay"].to_numpy() if "time_decay" in df.columns else None

model = DixonColesGoalModel(
    goals_home=df["goals_home"],
    goals_away=df["goals_away"],
    teams_home=df["team_home"],
    teams_away=df["team_away"],
    weights=weights,
)
model.fit(minimizer_options={"maxiter": 5000, "ftol": 1e-9})
print(model)            # human-readable parameter table
print(model.get_params())
model.save("models/dixon_coles_2425.pkl")
```

The Cython implementation runs **~50× faster** than the pure-Python loop
on a 5k-fixture dataset, and the parameter table is more readable.

## 3.2 Add Dixon-Coles correction (the actual `rho`)

Pitch Oracle's Poisson baseline under-prices 0-0, 1-0, 0-1, and 1-1
scorelines — the exact thing Dixon and Coles (1997) correct. Switching
to `DixonColesGoalModel` is one line:

```python
from penaltyblog.models import DixonColesGoalModel
dc = DixonColesGoalModel(gh, ga, th, ta, weights=w)
dc.fit()
print(dc)  # also shows the fitted rho
```

`dc.get_params()` returns a flat array:

```python
params = dc.get_params()             # length 2N + 2
attack, defense = params[:N], params[N:2*N]
hfa, rho        = params[-2], params[-1]
```

Or grab the underlying grid for any match:

```python
grid = dc.predict("Arsenal", "Chelsea", max_goals=10)
grid.home_win, grid.draw, grid.away_win
# (0.471, 0.246, 0.283)
```

## 3.3 Vectorised prediction across the slate

Once the model is fitted, predict every upcoming fixture in one call:

```python
import numpy as np
from penaltyblog.models import DixonColesGoalModel

# Fitted elsewhere:
model = DixonColesGoalModel.load("models/dixon_coles_2425.pkl")

# Upcoming slate:
slate = pd.read_parquet("data_files/features/upcoming.parquet")
home_idx = model.team_to_idx[slate["team_home"]].to_numpy()
away_idx = model.team_to_idx[slate["team_away"]].to_numpy()

grids = model.predict_many(home_idx, away_idx, max_goals=15)
slate["home_win"] = [g.home_win for g in grids]
slate["draw"]     = [g.draw     for g in grids]
slate["away_win"] = [g.away_win for g in grids]
slate["exp_home_goals"] = [g.home_goal_expectation for g in grids]
slate["exp_away_goals"] = [g.away_goal_expectation for g in grids]
slate["over_2_5"] = [g.total_goals("over", 2.5) for g in grids]
slate["btts_yes"] = [g.btts_yes for g in grids]
slate.to_parquet("data_files/predictions/upcoming.parquet")
```

`predict_many` is the new vectorised entry point; it allocates one shared
3D grid and avoids the Python-level `predict()` overhead.

## 3.4 Use `NegativeBinomialGoalModel` for goal-happy leagues

The Bundesliga averages ~3.1 goals per match; the Poisson distribution
under-disperses that. `NegativeBinomialGoalModel` adds an over-dispersion
parameter and fits leagues like the Bundesliga and Eredivisie better:

```python
from penaltyblog.models import NegativeBinomialGoalModel

nb = NegativeBinomialGoalModel(
    goals_home=df["goals_home"],
    goals_away=df["goals_away"],
    teams_home=df["team_home"],
    teams_away=df["team_away"],
    weights=df["time_decay"],
)
nb.fit()
print(nb)
```

Same `predict()` and `predict_many()` API as every other model. Use it as
a second model in the ensemble for leagues where goal totals are
predictable but variance is high.

## 3.5 Use `ZeroInflatedPoissonGoalsModel` for low-scoring leagues

Serie A averages ~2.6 goals but ~10% of matches end 0-0. A standard
Poisson under-counts the 0-0 cell. The ZIP model mixes in a structural
zero probability:

```python
from penaltyblog.models import ZeroInflatedPoissonGoalsModel

zip_model = ZeroInflatedPoissonGoalsModel(gh, ga, th, ta)
zip_model.fit()
```

Same API. Internally the model has one extra parameter (`pi_zero`) per
team; that param is exposed via `get_params()` and printed in `__repr__`.

## 3.6 Use `BivariatePoissonGoalModel` for tournaments

The standard Poisson assumes independence between home and away goals.
`BivariatePoissonGoalModel` adds a shared latent component, useful for
World Cup / Euros where one team's offensive explosion often coincides
with the other's defensive collapse:

```python
from penaltyblog.models import BivariatePoissonGoalModel

bp = BivariatePoissonGoalModel(gh, ga, th, ta, weights=w)
bp.fit()
```

The third parameter (the covariance) is what makes the joint 4-4 / 5-4
tail meaningful.

## 3.7 Use `WeibullCopulaGoalsModel` for tail-sensitive markets

Weibull marginals allow fat tails for both teams; the Gaussian copula
captures their co-movement. This is the model to use when pricing longshot
correct-score markets where the standard Poisson under-counts 5-0 / 0-5.

```python
from penaltyblog.models import WeibullCopulaGoalsModel

wc = WeibullCopulaGoalsModel(gh, ga, th, ta)
wc.fit()
```

The grid still comes back as a `FootballProbabilityGrid` so all the
market helpers work identically.

## 3.8 Time decay via weights

Every model accepts a `weights` argument. Pitch Oracle already has a
`time_decay` feature; just pass it through:

```python
import numpy as np

# Linear decay over a 3-season window
half_life_matches = 380
weights = 0.5 ** (df["matches_since"].to_numpy() / half_life_matches)
model = DixonColesGoalModel(gh, ga, th, ta, weights=weights)
model.fit()
```

This is the single biggest win Pitch Oracle can pick up: the current
training loop is unweighted, so a freak 5-0 result from 2018 counts the
same as one from yesterday.

## 3.9 Neutral venue handling

Cup finals and AFCON games are at neutral venues. Pass a boolean array:

```python
import numpy as np
neutral = (df["venue"] == "neutral").astype(int).to_numpy()
model = DixonColesGoalModel(gh, ga, th, ta, weights=w, neutral_venue=neutral)
```

When the i-th match is at a neutral venue, the model excludes home
advantage for that match. All eight models honour the argument
identically.

## 3.10 Custom minimiser options

`fit(minimizer_options={...})` forwards directly to
`scipy.optimize.minimize`. Common options:

```python
model.fit(minimizer_options={
    "maxiter": 5000,
    "ftol":   1e-9,
    "gtol":   1e-7,
    "disp":   True,        # progress to stdout
})
```

Use `use_gradient=False` to fall back to numerical gradients if the
Cython gradient ever produces NaNs for a particular dataset.

## 3.11 The `FootballProbabilityGrid` payoff

Once you have any model fitted, the grid unlocks 30+ derived markets
for free. The `FootballProbabilityGrid` API is documented in
[11-probability-grid.md](11-probability-grid.md); here are a few that
drop into Streamlit directly:

```python
g = model.predict("Arsenal", "Chelsea", max_goals=15)

# 1X2
g.home_win, g.draw, g.away_win

# Both teams to score
g.btts_yes, g.btts_no

# Double chance
g.double_chance_1x, g.double_chance_x2, g.double_chance_12

# Totals with quarter lines
under, push, over = g.totals(2.25)

# Asian handicap with quarter lines
ah = g.asian_handicap_probs("home", -0.75)   # {"win", "push", "lose"}

# Clean sheet
g.win_to_nil_home, g.win_to_nil_away

# Expected points (3/1/0)
g.expected_points_home(), g.expected_points_away()
```

This single object replaces every "compute the price of market X" snippet
in Pitch Oracle.

## 3.12 Walk-forward evaluation

The cleanest way to evaluate is to fit on past N matches, predict the
next M, log the loss, slide forward. This replaces `evaluate_poisson.py`:

```python
# scripts/eval/walk_forward.py
from penaltyblog.models import DixonColesGoalModel
from penaltyblog.metrics import ranked_probability_score
import pandas as pd

df = pd.read_parquet("data_files/features/training_set.parquet").sort_values("datetime")
window, horizon = 1500, 50
rows = []
for start in range(0, len(df) - window - horizon, horizon):
    train = df.iloc[start:start + window]
    test  = df.iloc[start + window:start + window + horizon]
    m = DixonColesGoalModel(
        train["goals_home"], train["goals_away"],
        train["team_home"],   train["team_away"],
        weights=train.get("time_decay"),
    ).fit()
    preds = m.predict_many(
        m.team_to_idx[test["team_home"]].to_numpy(),
        m.team_to_idx[test["team_away"]].to_numpy(),
    )
    test = test.assign(
        p_home=[p.home_win for p in preds],
        p_draw=[p.draw     for p in preds],
        p_away=[p.away_win for p in preds],
    )
    rps = ranked_probability_score(
        outcomes=test["result"],         # "H"/"D"/"A"
        probs=test[["p_home", "p_draw", "p_away"]].to_numpy(),
    )
    rows.append({"start": start, "rps": rps})
pd.DataFrame(rows).to_parquet("data_files/eval/walk_forward_rps.parquet")
```

The `penaltyblog.metrics` module ships RPS, Brier, log-loss, and
calibration helpers; see [06-implied-odds.md](06-implied-odds.md) for the
betting-market companion.

## 3.13 Model selection by AIC

Every model exposes `.aic` and `.loglikelihood`. Build a small leaderboard:

```python
# scripts/eval/aic_leaderboard.py
from penaltyblog.models import (
    PoissonGoalsModel, DixonColesGoalModel,
    BivariatePoissonGoalModel, NegativeBinomialGoalModel,
    ZeroInflatedPoissonGoalsModel, WeibullCopulaGoalsModel,
)
import pandas as pd

df = pd.read_parquet("data_files/features/training_set.parquet")
candidates = {
    "Poisson":           PoissonGoalsModel,
    "Dixon-Coles":       DixonColesGoalModel,
    "Bivariate Poisson": BivariatePoissonGoalModel,
    "Negative Binomial": NegativeBinomialGoalModel,
    "ZIP":               ZeroInflatedPoissonGoalsModel,
    "Weibull Copula":    WeibullCopulaGoalsModel,
}
out = []
for name, cls in candidates.items():
    m = cls(df["goals_home"], df["goals_away"],
            df["team_home"], df["team_away"]).fit()
    out.append({"model": name, "aic": m.aic, "loglik": m.loglikelihood,
                "n_params": m.n_params})
pd.DataFrame(out).sort_values("aic").to_csv("data_files/eval/aic_leaderboard.csv", index=False)
```

The leaderboard feeds the Streamlit "Model Comparison" view. Use it to
show users *why* Dixon-Coles is the production default.

## 3.14 Persist and load

Models support pickle via `model.save(path)` and `ModelClass.load(path)`.
Wrap in your CI:

```python
import joblib, datetime
joblib.dump({
    "model": model,
    "fitted_at": datetime.datetime.utcnow().isoformat(),
    "training_set_hash": hash(df.values.tobytes()),
    "config": {"max_goals": 15},
}, "models/dixon_coles_latest.pkl")
```

The hash is essential for invalidating cached Streamlit pages.

## 3.15 Pitch Oracle integration

**Touches**

- `train_models.py`, `evaluate_poisson.py`, `optimize_model.py`,
  `benchmark_hyperparameters.py`, `compare_model_features.py` — all
  rewritten to call `penaltyblog.models`.
- New `scripts/eval/walk_forward.py` and `scripts/eval/aic_leaderboard.py`.
- New `models/` directory holding persisted `DixonColesGoalModel` pickles
  per (league, season).

**Does not touch**

- Streamlit pages (grid properties are a superset of what they consume).
- Data ingestion (assumed complete from [01](01-scrapers-data-collection.md)).

**Migration cost**

- One engineer, ~5 days. The Dixon-Coles change is the easiest; the
  Bayesian and hierarchical migrations are described in
  [04-bayesian-models.md](04-bayesian-models.md).
