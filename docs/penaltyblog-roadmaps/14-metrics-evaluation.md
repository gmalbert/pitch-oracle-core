# 14 — Metrics, Evaluation & Diagnostics

`penaltyblog.metrics` is the only module in the library that does not
own a feature; it owns the **measurement instrument** that every other
roadmap depends on. It exposes:

- Proper scoring rules: `ranked_probability_score`, `brier_score`,
  `log_loss`, `pinball_loss` (interval), `crps`.
- Decompositions: `brier_decomposition` (reliability / resolution /
  uncertainty), `calibration_error` (ECE / MCE).
- Helpers: `reliability_table`, `concordance`, `calibration_curve`.
- Multi-class variants: `multiclass_brier`, `multiclass_log_loss`.

The Pitch Oracle surfaces that benefit are:

- `evaluate_poisson.py` — its current scoring is ad-hoc; the new
  library covers the proper scores that the product-expansion docs
  already demand.
- `validate_models.py` — currently only checks point-estimate
  accuracy; the new metrics add reliability, calibration, and
  ranked-probability checks.
- `precompute_model_diagnostics.py` — the artifact that feeds the
  Model Lab page; should now include reliability diagrams, cohort
  metrics, and a model-registry report.
- `track_predictions.py` — per-fixture history of the proper scores;
  the prediction-history page reads this.

## 14.1 The five scores Pitch Oracle must report

| Score | Use | Where it appears |
|-------|-----|------------------|
| `brier_score` | 3-outcome accuracy + calibration in one number | Model Lab header tile |
| `log_loss` | Punishes confident wrong calls; key for value betting | Model Lab header tile |
| `ranked_probability_score` | Multiclass proper score; better than Brier for 1X2 | Model Lab header tile |
| `brier_decomposition` | Splits Brier into reliability, resolution, uncertainty | Model Lab reliability tab |
| `calibration_error` | ECE / MCE in 10 decile bins | Model Lab reliability tab |

The library also exposes `concordance` (rank agreement between two
probability vectors) — useful for "did the new ensemble agree with the
old?" sanity checks during migration.

### Single-fixture scoring

```python
from penaltyblog.metrics import brier_score, log_loss, ranked_probability_score
import numpy as np

# outcome in one-hot form
outcome = np.array([1, 0, 0])           # home win
probs   = np.array([0.55, 0.25, 0.20])

brier_score(outcome, probs)              # 0.225
log_loss(outcome, probs)                # 0.597
ranked_probability_score(outcome, probs) # 0.139
```

The result of every match-day forecast is one row in an append-only
`forecast_ledger.parquet` that carries the proper scores alongside the
probabilities.

## 14.2 Walk-forward harness

The single most important evaluation script. It replaces the existing
"train on all, test on all" pattern in `evaluate_poisson.py`:

```python
# scripts/eval/walk_forward.py
from penaltyblog.models import DixonColesGoalModel
from penaltyblog.metrics import (
    brier_score, log_loss, ranked_probability_score,
    brier_decomposition,
)
import pandas as pd

df = pd.read_parquet("data_files/features/training_set.parquet") \
       .sort_values("datetime").reset_index(drop=True)

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
        p_home=[g.home_win for g in preds],
        p_draw=[g.draw     for g in preds],
        p_away=[g.away_win for g in preds],
    )

    out = test["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    P   = test[["p_home", "p_draw", "p_away"]].to_numpy()
    rows.append({
        "start": start,
        "brier": brier_score(out, P),
        "log_loss": log_loss(out, P),
        "rps":   ranked_probability_score(out, P),
        "decomp_reliability": brier_decomposition(out, P)["reliability"],
        "decomp_resolution":  brier_decomposition(out, P)["resolution"],
    })

pd.DataFrame(rows).to_parquet("data_files/eval/walk_forward_dc.parquet")
```

`window` and `horizon` are the only two knobs; everything else is
deterministic given a fixed training set hash.

### Paired comparisons

Every model in the library is a challenger until it wins a paired
comparison against the current champion:

```python
# scripts/eval/paired_compare.py
from penaltyblog.models import (
    DixonColesGoalModel, NegativeBinomialGoalModel,
)
from penaltyblog.metrics import brier_score
import pandas as pd

df = pd.read_parquet("data_files/features/training_set.parquet")
df = df.sort_values("datetime").reset_index(drop=True)

# Build the same walk-forward predictions for both models
def run(cls):
    rows = []
    for start in range(0, len(df) - 1500 - 50, 50):
        train = df.iloc[start:start + 1500]
        test  = df.iloc[start + 1500:start + 1550]
        m = cls(train["goals_home"], train["goals_away"],
                train["team_home"],   train["team_away"],
                weights=train.get("time_decay")).fit()
        preds = m.predict_many(
            m.team_to_idx[test["team_home"]].to_numpy(),
            m.team_to_idx[test["team_away"]].to_numpy(),
        )
        out = test["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        P   = pd.DataFrame({
            "h": [g.home_win for g in preds],
            "d": [g.draw     for g in preds],
            "a": [g.away_win for g in preds],
        }).to_numpy()
        rows.append(brier_score(out, P))
    return rows

a = run(DixonColesGoalModel)
b = run(NegativeBinomialGoalModel)

# Paired t-test on per-window Brier
from scipy.stats import ttest_rel
print(ttest_rel(a, b))
```

A challenger wins when (i) the paired test rejects "no difference" in
its favour AND (ii) it is no worse than the champion in any cohort
(see §14.7). Both conditions are recorded in `model_registry.json`.

## 14.3 Reliability diagram

A reliability diagram plots predicted probability (binned) against
empirical frequency. The library computes the table; Pitch Oracle
renders the chart:

```python
# scripts/eval/reliability.py
from penaltyblog.metrics import reliability_table
import pandas as pd
import plotly.graph_objects as go

preds = pd.read_parquet("data_files/eval/all_predictions.parquet")
table = reliability_table(
    y_true=preds["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy(),
    y_score=preds["p_home"].to_numpy(),
    n_bins=10,
)
# columns: bin_midpoint, n, fraction_positive, predicted, gap
fig = go.Figure()
fig.add_trace(go.Scatter(x=table["bin_midpoint"], y=table["fraction_positive"],
                         mode="markers+lines", name="actual"))
fig.add_trace(go.Scatter(x=table["bin_midpoint"], y=table["predicted"],
                         mode="lines", name="predicted", line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="grey"), showlegend=False))
st.plotly_chart(fig, use_container_width=True)
```

A reliability diagram that hugs the diagonal means the model is
calibrated. A systematic bow means the model is over- or
under-confident in a specific band. The diagram is **per market** (1X2
home / 1X2 draw / 1X2 away / BTTS yes / over 2.5) and per cohort
(promoted / early season / derby / short rest).

## 14.4 Calibration error

`calibration_error` returns the Expected Calibration Error (ECE) and
Maximum Calibration Error (MCE) in a single call:

```python
from penaltyblog.metrics import calibration_error

ece, mce = calibration_error(
    y_true=preds["result_H"],
    y_score=preds["p_home"],
    n_bins=10,
    norm="l1",   # or "l2"
)
```

Pin a release-gate threshold in the manifest:

```python
# data_files/eval/gates.json
{
    "champion_model": "dixon_coles_v3",
    "brier":         0.205,
    "log_loss":      0.612,
    "rps":           0.180,
    "ece_home":      0.012,
    "ece_draw":      0.018,
    "ece_away":      0.014,
    "min_n":         500,
}
```

`precompute_model_diagnostics.py` fails if the new walk-forward
numbers regress on any of these by more than `2 * std(rolling-12)`.

## 14.5 Decomposition

Brier has a closed-form decomposition into reliability, resolution,
and uncertainty:

```python
from penaltyblog.metrics import brier_decomposition

out = brier_decomposition(y_true, P)
# {"reliability": ..., "resolution": ..., "uncertainty": ...,
#  "total": ..., "fraction_calibrated": ...}
```

- `reliability` (lower is better) — how far off the calibration curve
  is.
- `resolution` (higher is better) — how much the model's predictions
  vary across bins.
- `uncertainty` (data-only) — the variance of the true outcome.

A well-calibrated, sharp model has *low* reliability and *high*
resolution. The Model Lab renders the three numbers as a stacked
horizontal bar to make the trade-off obvious.

## 14.6 Pinball / interval

Bayesian posteriors give a credible interval on every probability.
The pinball loss measures the calibration of those intervals:

```python
from penaltyblog.metrics import pinball_loss

# For the lower bound of a 50% interval
pinball_loss(y_true, lower_bound, alpha=0.25)
# For the upper bound
pinball_loss(y_true, upper_bound, alpha=0.75)
```

A walk-forward pinball-loss curve, plotted for α ∈ {0.05, 0.25, 0.5,
0.75, 0.95}, becomes the **uncertainty fan diagnostic** in the UI (F07
in the product-expansion catalog).

## 14.7 Cohort metrics

The library is score-agnostic; the cohort logic lives in Pitch Oracle:

```python
# scripts/eval/cohorts.py
from penaltyblog.metrics import brier_score, log_loss
import pandas as pd

preds = pd.read_parquet("data_files/eval/all_predictions.parquet")
preds["cohort"] = (
    preds["is_promoted"].astype(str) + "_" +
    preds["rest_days"].clip(upper=14).astype(str) + "_" +
    preds["is_derby"].astype(str)
)

rows = []
for cohort, group in preds.groupby("cohort"):
    if len(group) < 30:                       # min sample guard
        continue
    out = group["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    P   = group[["p_home", "p_draw", "p_away"]].to_numpy()
    rows.append({
        "cohort": cohort,
        "n":      len(group),
        "brier":  brier_score(out, P),
        "log_loss": log_loss(out, P),
    })
pd.DataFrame(rows).sort_values("brier").to_csv(
    "data_files/eval/cohort_metrics.csv", index=False
)
```

The CSV is the cohort table in the Model Lab "Cohorts" tab. Min-sample
guard prevents noisy single-fixture cohorts from being compared
across models.

## 14.8 Append-only forecast ledger

Every prediction issued by the artifact pipeline is logged once,
immutably, with its eventual outcome and proper score:

```python
# scripts/eval/append_ledger.py
from penaltyblog.metrics import brier_score, log_loss, ranked_probability_score
import pandas as pd, json
from pathlib import Path
from datetime import datetime, timezone

LEDGER = Path("data_files/eval/forecast_ledger.parquet")
PRED   = Path("data_files/predictions/upcoming.parquet")
RESULT = Path("data_files/results/latest.json")

preds = pd.read_parquet(PRED)
results = json.loads(RESULT.read_text())
preds = preds.merge(results, on="match_id", how="inner")
preds = preds[preds["result"].notna()]

def one_hot(r): return {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[r]
out = np.array([one_hot(r) for r in preds["result"]])
P   = preds[["p_home", "p_draw", "p_away"]].to_numpy()

ledger = pd.DataFrame({
    "issued_at_utc": preds["issued_at_utc"],
    "match_id":      preds["match_id"],
    "model_id":      preds["model_id"],
    "p_home":        preds["p_home"],
    "p_draw":        preds["p_draw"],
    "p_away":        preds["p_away"],
    "result":        preds["result"],
    "brier":         [brier_score(o, p) for o, p in zip(out, P)],
    "log_loss":      [log_loss(o, p)    for o, p in zip(out, P)],
    "rps":           [ranked_probability_score(o, p) for o, p in zip(out, P)],
})

if LEDGER.exists():
    old = pd.read_parquet(LEDGER)
    ledger = pd.concat([old, ledger]).drop_duplicates(
        subset=["issued_at_utc", "match_id", "model_id"]
    )
ledger.to_parquet(LEDGER)
```

The ledger is the source for the prediction-history page (F38 in the
product-expansion catalog) and the long-window reliability reports.

## 14.9 Model registry

A single `model_registry.json` records every model that has ever been
fitted, with its evaluation summary and promotion reason:

```python
# scripts/eval/registry.py
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone

def registry_path(): return Path("data_files/eval/model_registry.json")

def register(model_id, *, version, parent, training_set_hash,
             metrics, promoted_reason):
    path = registry_path()
    registry = json.loads(path.read_text()) if path.exists() else {"models": []}
    entry = {
        "model_id":         model_id,
        "version":          version,
        "parent":           parent,
        "training_set_hash": training_set_hash,
        "metrics":          metrics,
        "promoted_reason":  promoted_reason,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    registry["models"].append(entry)
    path.write_text(json.dumps(registry, indent=2))
```

The Model Lab renders the registry as a sortable table. Promotion
reason is a free-form string but the schema requires it to include the
paired test statistic and the cohort pass/fail.

## 14.10 Diagnostics artifact

A single artifact that the Streamlit app reads:

```python
# scripts/eval/build_diagnostics.py
import json
from pathlib import Path
import pandas as pd

out = {
    "registry":      json.loads(Path("data_files/eval/model_registry.json").read_text()),
    "walk_forward":  pd.read_parquet("data_files/eval/walk_forward_dc.parquet").to_dict(orient="records"),
    "reliability":   pd.read_parquet("data_files/eval/reliability.parquet").to_dict(orient="records"),
    "cohorts":       pd.read_csv("data_files/eval/cohort_metrics.csv").to_dict(orient="records"),
    "gates":         json.loads(Path("data_files/eval/gates.json").read_text()),
}
Path("data_files/artifacts/diagnostics.json").write_text(json.dumps(out, indent=2))
```

The artifact is registered in manifest v3 under
`artifacts/diagnostics.json`; cache-contract tests assert that
every Streamlit page reading it can find it.

## 14.11 Pitch Oracle integration

**Touches**

- `evaluate_poisson.py` — replace ad-hoc scoring with
  `pb.metrics.*`; the script now also produces walk-forward parquet.
- `validate_models.py` — read `model_registry.json` and the gates
  file; fail if any gate regresses.
- `precompute_model_diagnostics.py` — emit the
  `artifacts/diagnostics.json` artifact.
- `track_predictions.py` — append to `forecast_ledger.parquet`.
- New `scripts/eval/walk_forward.py`, `paired_compare.py`,
  `reliability.py`, `cohorts.py`, `registry.py`, `build_diagnostics.py`.
- Streamlit Model Lab page reads `diagnostics.json` (no live compute).

**Does not touch**

- The goal models (metrics are independent).
- The Streamlit scrapers / ingestion pages.

**Migration cost**

- One engineer, ~3 days including the diagnostics artifact wiring.

## 14.12 Anti-patterns

1. **Don't report accuracy.** It hides calibration problems and is
   useless for value betting.
2. **Don't compute proper scores on in-sample predictions.** Every
   score is computed on out-of-time rows; the walk-forward harness
   is the only sanctioned source.
3. **Don't bin predicted probabilities inconsistently across
   diagrams.** Always use 10 equally-populated bins; the
   `reliability_table` function enforces this.
4. **Don't promote on rolling-12 Brier alone.** A model can win on
   one cohort and lose on another; the gate requires no regression
   in any cohort with `n >= 30`.
5. **Don't use the same binning for tail probabilities.** A 0–0 cell
   with predicted prob 0.02 needs a different reliability table
   than the home-win column.
