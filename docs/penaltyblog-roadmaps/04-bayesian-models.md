# 04 — Bayesian Models

`penaltyblog.bayes` provides MCMC goal models with full posterior
distributions. There are two classes:

- `BayesianGoalModel` — single Dixon-Coles likelihood, MCMC posterior over
  attack × N, defense × N, home_advantage, and rho.
- `HierarchicalBayesianGoalModel` — adds a league-level prior on the
  attack and defense variances, so small-sample teams borrow strength
  from the rest of the league.

Both are drop-in replacements for the MLE models in
[03-goal-models.md](03-goal-models.md): the same `predict()`,
`predict_many()`, `__repr__`, and `FootballProbabilityGrid` return type.
The bonus is the trace: every posterior sample is a *fully consistent*
parameter vector, so you can derive uncertainty intervals on any derived
quantity for free.

Pitch Oracle scripts that benefit:

- `model_optimization.py` — add a "show prediction uncertainty" toggle.
- `benchmark_hyperparameters.py` — use the posterior to evaluate the
  parameter stability over time.
- New `scripts/eval/bayesian_calibration.py` — produce calibration plots
  from posterior predictive probabilities.

## 4.1 The simplest Bayesian fit

```python
from penaltyblog.models.bayesian_goal_model import BayesianGoalModel
import pandas as pd

df = pd.read_parquet("data_files/features/training_set.parquet")

bg = BayesianGoalModel(
    goals_home=df["goals_home"],
    goals_away=df["goals_away"],
    teams_home=df["team_home"],
    teams_away=df["team_away"],
    weights=df.get("time_decay"),
)
bg.fit(n_samples=2000, burn=1000, n_chains=4, n_cores=4)
print(bg)
```

`fit()` runs four independent MCMC chains in parallel, discards the first
1000 samples per chain as burn-in, and stores the retained samples as a
`(8000, 2N + 2)` trace.

### Predict

```python
grid = bg.predict("Arsenal", "Chelsea", max_goals=15)
grid.home_win, grid.draw, grid.away_win
```

The grid is a **posterior predictive** grid: every probability is
already averaged over parameter uncertainty. Compare with the MLE model,
where `home_win` is a point estimate.

## 4.2 Inspect the trace

```python
diag = bg.get_diagnostics()
# R-hat and ESS for every parameter, indexed by name
print(diag.sort_values("r_hat", ascending=False).head())
```

If `r_hat > 1.01` for any parameter, increase `n_samples` or `n_chains`.
`ESS` (effective sample size) should be > 400 for downstream credibility
intervals.

### Trace plots

```python
from penaltyblog.viz import plot_trace, plot_autocorr, plot_posterior

plot_trace(bg.trace_dict["home_advantage"])
plot_autocorr(bg.trace_dict["home_advantage"])
plot_posterior(bg.trace_dict["rho"])
```

These are first-class visualisations in `penaltyblog.viz`; see
[10-visualizations.md](10-visualizations.md) for the full list.

## 4.3 Credible intervals on derived quantities

The Bayesian model's superpower is that you can ask "what is the 90%
credible interval for Arsenal's home win probability vs Chelsea?":

```python
import numpy as np

# Pull 8000 posterior predictive probabilities
n_samples = 8000
home_win_samples = []
draw_samples = []
away_win_samples = []
for h_idx in [bg.team_to_idx["Arsenal"]]:
    for a_idx in [bg.team_to_idx["Chelsea"]]:
        for s in range(n_samples):
            row = bg.trace[s]                       # attack, defense, hfa, rho
            # Build a temporary grid using the s-th posterior draw
            from penaltyblog.models.football_probability_grid import (
                create_dixon_coles_grid,
            )
            lam_h = np.exp(row[h_idx] - row[a_idx + bg.n_teams] + row[-2])
            lam_a = np.exp(row[a_idx] - row[h_idx + bg.n_teams])
            g = create_dixon_coles_grid(lam_h, lam_a, rho=row[-1])
            home_win_samples.append(g.home_win)
            draw_samples.append(g.draw)
            away_win_samples.append(g.away_win)

print("Arsenal home win 90% CI:", np.percentile(home_win_samples, [5, 95]))
print("Draw 90% CI:           ", np.percentile(draw_samples, [5, 95]))
```

That interval is exactly the kind of thing Pitch Oracle's
`model_optimization.py` script is currently trying (and failing) to
approximate with bootstrap resampling.

### Or use the posterior directly via `predict_many` with the full trace

```python
# Faster: the model already knows how to posterior-predict across the slate
grids = bg.predict_many(home_idx, away_idx, max_goals=15)
# Each grid is already the mean over the posterior; for *full distributions*
# over a single match, use the helper above.
```

## 4.4 Hierarchical model for sparse leagues

A league like the Scottish Premiership has only ~228 fixtures per
season. MLE estimates for promoted/relegated teams swing wildly.
`HierarchicalBayesianGoalModel` puts a league-level prior on the
attack/defense variances; small-sample teams are pulled toward the
league mean, large-sample teams keep their own estimates.

```python
from penaltyblog.models.hierarchical_bayesian_goal_model import (
    HierarchicalBayesianGoalModel,
)

hb = HierarchicalBayesianGoalModel(
    goals_home=df["goals_home"],
    goals_away=df["goals_away"],
    teams_home=df["team_home"],
    teams_away=df["team_away"],
    weights=df.get("time_decay"),
)
hb.fit(n_samples=4000, burn=2000, n_chains=4, n_cores=4)
```

The trace now contains the league-level hyperparameters too. Inspect
them:

```python
plot_posterior(hb.trace_dict["sigma_attack"])
plot_posterior(hb.trace_dict["sigma_defense"])
```

A wide posterior on `sigma_attack` means teams in this league vary a
lot; a narrow one means teams are interchangeable.

## 4.5 Compare Bayesian vs MLE for the same match

This is the single most useful deliverable for the Streamlit UI: a
"model disagreement" view.

```python
from penaltyblog.models import DixonColesGoalModel
from penaltyblog.models.bayesian_goal_model import BayesianGoalModel

dc = DixonColesGoalModel(gh, ga, th, ta).fit()
bg = BayesianGoalModel(gh, ga, th, ta).fit(n_samples=2000, burn=1000)

slate = pd.read_parquet("data_files/features/upcoming.parquet")
mle_grids    = dc.predict_many(mle_home_idx, mle_away_idx, max_goals=15)
bayes_grids  = bg.predict_many(bay_home_idx, bay_away_idx, max_goals=15)

slate["mle_home_win"]   = [g.home_win for g in mle_grids]
slate["bayes_home_win"] = [g.home_win for g in bayes_grids]
slate["disagreement"]   = slate["mle_home_win"] - slate["bayes_home_win"]
slate.to_parquet("data_files/predictions/disagreement.parquet")
```

A row with large `disagreement` is one where the MLE point estimate is
unlikely to be calibrated — exactly the matches where you want to
highlight uncertainty in the UI.

## 4.6 Posterior-predictive calibration

The right way to check if a Bayesian model is calibrated:

```python
# scripts/eval/bayesian_calibration.py
from penaltyblog.metrics import brier_score
from penaltyblog.models.bayesian_goal_model import BayesianGoalModel
import pandas as pd, numpy as np

df = pd.read_parquet("data_files/features/training_set.parquet")
df = df.sort_values("datetime").reset_index(drop=True)

# Walk-forward: train on first 1500, predict next 200
window, horizon = 1500, 200
records = []
for start in range(0, len(df) - window - horizon, horizon):
    train = df.iloc[start:start + window]
    test  = df.iloc[start + window:start + window + horizon]
    bg = BayesianGoalModel(
        train["goals_home"], train["goals_away"],
        train["team_home"],   train["team_away"],
    ).fit(n_samples=1000, burn=500, n_chains=2, n_cores=2)
    grids = bg.predict_many(
        bg.team_to_idx[test["team_home"]].to_numpy(),
        bg.team_to_idx[test["team_away"]].to_numpy(),
    )
    test = test.assign(
        p_home=[g.home_win for g in grids],
        p_draw=[g.draw     for g in grids],
        p_away=[g.away_win for g in grids],
    )
    records.append({
        "start": start,
        "brier": brier_score(outcome=test["result_H"], prob=test["p_home"]),
    })
pd.DataFrame(records).to_parquet("data_files/eval/bayesian_brier.parquet")
```

This is the gold-standard diagnostic. Compare to the equivalent MLE
loop; the Bayesian version should win on cold-start fixtures and never
lose materially elsewhere.

## 4.7 Use the posterior as a feature for the ML ensemble

Pitch Oracle already runs an XGBoost / Random Forest / GBT / Logistic
ensemble on top of the goal models. The Bayesian posterior gives you
**uncertainty features**:

```python
# scripts/ensemble/build_features.py
import numpy as np, pandas as pd
from penaltyblog.models.bayesian_goal_model import BayesianGoalModel

df = pd.read_parquet("data_files/features/training_set.parquet")
bg = BayesianGoalModel(df["goals_home"], df["goals_away"],
                      df["team_home"], df["team_away"]).fit(
                          n_samples=2000, burn=1000, n_chains=4
                      )

# Per-match: compute the spread of home_win over posterior samples
spread_rows = []
for s in range(0, len(bg.trace), 50):  # thin to 160 samples
    # Rebuild the grid with this draw, for every match in the slate
    ...
spread = np.array(spread_rows)
features["bayes_p_home_mean"] = spread.mean(axis=0)
features["bayes_p_home_std"]  = spread.std(axis=0)
features["bayes_p_home_ci_width"] = (
    np.percentile(spread, 95, axis=0) - np.percentile(spread, 5, axis=0)
)
features.to_parquet("data_files/features/bayes_features.parquet")
```

The ensemble then learns: when `bayes_p_home_ci_width` is large, the
XGBoost prediction should be shrunk toward 50/30/20.

## 4.8 Productionisation

Bayesian models are slower than MLE. A typical MLE fit on 1500 fixtures
takes ~50ms; a Bayesian fit with 2000 samples × 4 chains takes ~6
seconds on 4 cores. Two practical tips:

1. **Train once per day**, not once per page-load. The trace is
   serialised; Streamlit reads the trace, not the model.
2. **Persist the trace**, not the model. The trace is a
   `(n_samples, 2N+2)` array — pickling it is cheap.

```python
import joblib
joblib.dump({
    "trace": bg.trace,
    "teams": bg.teams,
    "team_to_idx": bg.team_to_idx,
    "fitted_at": pd.Timestamp.utcnow().isoformat(),
}, "models/bayes_2425_trace.pkl")
```

The Streamlit page then loads it via:

```python
import joblib
trace_data = joblib.load("models/bayes_2425_trace.pkl")
bg = BayesianGoalModel.__new__(BayesianGoalModel)
bg.trace = trace_data["trace"]
bg.teams = trace_data["teams"]
bg.team_to_idx = trace_data["team_to_idx"]
bg.n_teams = len(bg.teams)
bg.fitted = True
```

That detour avoids re-running MCMC in the UI.

## 4.9 Pitch Oracle integration

**Touches**

- `model_optimization.py` — add a "Bayesian with credible intervals"
  tab.
- `benchmark_hyperparameters.py` — add `sigma_attack`, `sigma_defense`
  diagnostics.
- New `scripts/eval/bayesian_calibration.py` (see 4.6).
- New `scripts/ensemble/build_features.py` to emit uncertainty features
  for the existing XGBoost / RF / GBT / LR ensemble.
- New `models/bayes_*_trace.pkl` artifacts.

**Does not touch**

- The ML ensemble code itself — it just gets new feature columns.
- Streamlit pages (data shape preserved).

**Migration cost**

- One engineer, ~4 days including the productionisation steps and the
  walk-forward evaluation harness.

## 4.10 Common pitfalls

1. **Don't trust `bg.predict()` before `bg.fitted`.** MCMC needs to
   converge first; check `bg.get_diagnostics()`.
2. **Don't compare R-hat across chains of different lengths.** Always
   run the same `n_samples` per chain.
3. **Don't use `n_cores=os.cpu_count()`.** Cython's OpenMP already
   parallelises inside each chain; oversubscribing slows things down.
   Use `n_cores = min(4, n_chains)`.
4. **Don't throw away the trace.** It is the entire value of the model.
