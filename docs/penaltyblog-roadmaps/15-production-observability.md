# 15 — Production Observability, Drift & Alerts

Roadmap 14 gave us a *measurement instrument*; this roadmap gives us
the *closed loop* — drift detection, calibration monitoring, and
alerts. Every artefact lives in `data_files/observability/` and is
registered in manifest v3.

The five sub-systems:

1. **Input drift** — feature distribution shifts (PSI, KS).
2. **Output drift** — probability / odds / market distribution shifts.
3. **Calibration drift** — ECE / Brier / RPS over a rolling window.
4. **Coverage drift** — entity resolution, missing fixtures, stale
   snapshots.
5. **Operational drift** — latency, cache hit rate, scraper failure
   rate.

Each one produces a JSON severity report that the consumer workflow
fails on if any severity is `critical`.

## 15.1 Input drift — PSI / KS

Population Stability Index (PSI) on every feature in
`data_files/features/training_set.parquet`. Reference window is the
last full season; comparison window is the most recent 30 days.

```python
# scripts/observability/input_drift.py
import numpy as np, pandas as pd
from scipy.stats import ks_2samp

REF_PATH = "data_files/features/training_set.parquet"
CUR_PATH = "data_files/features/incoming.parquet"
OUT_PATH = "data_files/observability/input_drift.json"

ref = pd.read_parquet(REF_PATH)
cur = pd.read_parquet(CUR_PATH)

NUMERIC = [c for c in ref.columns if pd.api.types.is_numeric_dtype(ref[c])]
rows = []
for col in NUMERIC:
    r, c = ref[col].dropna().to_numpy(), cur[col].dropna().to_numpy()
    if len(r) < 100 or len(c) < 30:
        continue
    psi = _psi(r, c, n_bins=10)
    ks_stat, ks_p = ks_2samp(r, c)
    rows.append({
        "feature": col,
        "psi": float(psi),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "severity": _severity(psi, ks_p),
    })

def _psi(reference, current, n_bins=10):
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, edges)[0] / len(reference)
    cur_counts = np.histogram(current,    edges)[0] / len(current)
    ref_counts = np.clip(ref_counts, 1e-6, None)
    cur_counts = np.clip(cur_counts, 1e-6, None)
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))

def _severity(psi, ks_p):
    if psi > 0.25 or ks_p < 1e-4: return "critical"
    if psi > 0.10 or ks_p < 1e-2: return "warning"
    return "ok"

pd.DataFrame(rows).to_json(OUT_PATH, orient="records")
```

Convention: `psi < 0.10` is no change, `0.10 ≤ psi < 0.25` is a
warning, `psi ≥ 0.25` is critical.

## 15.2 Output drift

The probability distribution of every market (1X2, totals, BTTS)
should not suddenly shift. Compare the most recent 7 days of
`forecast_ledger.parquet` against the rolling 90-day baseline:

```python
# scripts/observability/output_drift.py
from penaltyblog.metrics import brier_score
import pandas as pd
import numpy as np
import json

ledger = pd.read_parquet("data_files/eval/forecast_ledger.parquet")
ledger = ledger.sort_values("issued_at_utc")

base = ledger[ledger["issued_at_utc"] <  ledger["issued_at_utc"].max() - pd.Timedelta(days=7)]
cur  = ledger[ledger["issued_at_utc"] >= ledger["issued_at_utc"].max() - pd.Timedelta(days=7)]

# For each market, compare the predicted distribution with a Wasserstein
from scipy.stats import wasserstein_distance
markets = ["p_home", "p_draw", "p_away", "p_btts", "p_over_2_5"]
rows = []
for col in markets:
    d = wasserstein_distance(base[col].to_numpy(), cur[col].to_numpy())
    rows.append({"market": col, "wasserstein": float(d),
                 "severity": "critical" if d > 0.05 else
                             "warning"  if d > 0.02 else "ok"})

json.dump(rows, open("data_files/observability/output_drift.json", "w"))
```

A Wasserstein distance above `0.05` on `p_home` typically means a
promoted team has swung the entire distribution; a check that the
underlying entity-coverage report is clean prevents a false alarm.

## 15.3 Calibration drift

Calibration is monitored on the rolling 28-day window, with a
publication gate at 7 days. A "calibration breach" is `ECE > 2 * rolling
median(ECE)` for two consecutive weeks:

```python
# scripts/observability/calibration_drift.py
from penaltyblog.metrics import calibration_error
import pandas as pd
import json

ledger = pd.read_parquet("data_files/eval/forecast_ledger.parquet")
ledger["week"] = ledger["issued_at_utc"].dt.to_period("W").astype(str)

rows = []
for week, group in ledger.groupby("week"):
    if len(group) < 50:
        continue
    ece_h, _ = calibration_error(group["result_H"], group["p_home"], n_bins=10)
    ece_d, _ = calibration_error(group["result_D"], group["p_draw"], n_bins=10)
    ece_a, _ = calibration_error(group["result_A"], group["p_away"], n_bins=10)
    rows.append({
        "week": week,
        "ece_home": ece_h, "ece_draw": ece_d, "ece_away": ece_a,
        "n": len(group),
    })

weekly = pd.DataFrame(rows).sort_values("week")
weekly["rolling_median_ece_home"] = weekly["ece_home"].rolling(8, min_periods=4).median()
weekly["calibration_breach"] = weekly["ece_home"] > 2 * weekly["rolling_median_ece_home"]
weekly.to_json("data_files/observability/calibration_drift.json", orient="records")
```

A breach triggers a workflow that:

1. Re-runs the walk-forward harness.
2. Compares the latest 28-day Brier / RPS to the rolling baseline.
3. If both are worse, demotes the current champion in
   `model_registry.json` and promotes the previous one.

## 15.4 Coverage drift

Entity coverage is the single most important trust signal. It is
monitored against the same manifest contract that F46 / F47 / F48
in the product-expansion catalog define:

```python
# scripts/observability/coverage_drift.py
import pandas as pd, json

fixtures = pd.read_parquet("data_files/fixtures/latest.parquet")
entities = pd.read_parquet("data_files/entities/teams.parquet")
mappings = pd.read_csv("data_files/entities/team_mappings.csv")

unresolved = fixtures[~fixtures["team_home_id"].isin(entities["team_id"])
                    & ~fixtures["team_away_id"].isin(entities["team_id"])]

out = {
    "n_fixtures":         len(fixtures),
    "n_unresolved":       len(unresolved),
    "fraction_resolved":  1.0 - len(unresolved) / max(len(fixtures), 1),
    "stale_aliases":      mappings[mappings["last_verified_utc"] <
                                   (pd.Timestamp.utcnow() - pd.Timedelta(days=30)).isoformat()],
    "severity":           "critical" if len(unresolved) > 0 else
                          "warning"  if (mappings["last_verified_utc"] <
                                         (pd.Timestamp.utcnow() - pd.Timedelta(days=30)).isoformat()).any() else
                          "ok",
}
json.dump(out, open("data_files/observability/coverage_drift.json", "w"))
```

A `fraction_resolved < 1.0` is `critical`; the workflow blocks.

## 15.5 Operational drift

The most boring category, but the most likely to bite in CI:

- Scraper latency per provider (mean / p95 over the last 7 runs).
- Cache hit rate (fraction of pages that hit the manifest cache vs
  recompute).
- Streamlit cold-start time (from a synthetic probe page).
- Number of artifact files referenced by the manifest but missing
  from disk.

```python
# scripts/observability/operational_drift.py
import json, time, glob, os
from pathlib import Path
import pandas as pd

manifest = json.loads(Path("data_files/cache_manifest.json").read_text())
artifacts = manifest["artifacts"]
missing = [a["path"] for a in artifacts if not Path(a["path"]).exists()]

# Cache hit rate from a single probe run
from urllib.request import urlopen
t0 = time.time()
urlopen("http://localhost:8501/_stcore/health").read()  # local probe
cold_start_ms = (time.time() - t0) * 1000

out = {
    "n_artifacts_declared":  len(artifacts),
    "n_artifacts_missing":   len(missing),
    "missing":               missing,
    "cold_start_ms":         cold_start_ms,
    "severity":              "critical" if missing else
                             "warning"  if cold_start_ms > 5000 else "ok",
}
json.dump(out, open("data_files/observability/operational_drift.json", "w"))
```

## 15.6 Aggregated severity report

A single artifact summarises all five categories:

```python
# scripts/observability/aggregate.py
import json, glob
from pathlib import Path

reports = {}
for p in glob.glob("data_files/observability/*.json"):
    reports[Path(p).stem] = json.loads(Path(p).read_text())

def worst(severities):
    order = {"ok": 0, "warning": 1, "critical": 2}
    return max(severities, key=lambda s: order.get(s, 0))

overall = worst([v.get("severity", "ok") for v in reports.values()])
out = {"overall": overall, "reports": reports,
       "generated_at_utc": pd.Timestamp.utcnow().isoformat()}
Path("data_files/observability/severity.json").write_text(json.dumps(out, indent=2))
```

The consumer workflow calls `aggregate.py` last. If
`severity.overall == "critical"`, the workflow fails. If it is
`warning`, the workflow succeeds but the next PR is blocked from
merging until the warning is triaged.

## 15.7 Alert routing

Alerts are written to two sinks:

1. **GitHub Actions summary** — a markdown table with the worst 10
   features by PSI and the three worst calibration breaches. The
   consumer workflow surfaces this on the run page.
2. **Email / Slack** (optional) — the `observability/` artifact is
   uploaded as a workflow artifact; downstream consumers can wire it
   into whatever notification system they use.

Pitch Oracle does not own the email/Slack plumbing; it only guarantees
the artifact is shaped consistently across consumers.

## 15.8 Reproducibility & regression suites

Three regression suites run nightly:

| Suite | What it catches |
|-------|-----------------|
| `tests/test_drift_regression.py` | Golden input/output distributions; a failed PSI guard means a feature column was dropped or recomputed. |
| `tests/test_calibration_regression.py` | Golden reliability table; a failed expectation means a model was promoted that should not have been. |
| `tests/test_observability_contract.py` | The `severity.json` artifact has the right schema; every report is keyed by the right name; the overall severity is the worst. |

The drift regression tests are the **only** place where a fixed
input set is allowed to drift; they are pinned by a SHA.

## 15.9 Productionisation rules

1. **Never auto-roll-back a model.** Surface a warning, but a human
   reads the calibration breach and decides.
2. **Never run observability in the Streamlit app.** It runs in the
   artifact workflow; the app reads the JSON, never recomputes.
3. **Never alert on a single-week breach.** Two consecutive weeks
   is the threshold.
4. **Never ignore an unresolved team.** A `critical` coverage report
   blocks publication; a warning does not.

## 15.10 Pitch Oracle integration

**Touches**

- New `scripts/observability/` directory with one script per
  sub-system plus `aggregate.py`.
- `precompute_model_diagnostics.py` — extended to invoke the
  observability scripts.
- `.github/workflows/consumer.yml` — calls the aggregate step
  last; fails on `critical`.
- Streamlit Data Control Room — new tile reading
  `severity.json`.
- New `tests/test_drift_regression.py`,
  `tests/test_calibration_regression.py`,
  `tests/test_observability_contract.py`.

**Does not touch**

- The goal models, the ratings, the betting utilities.

**Migration cost**

- One engineer, ~4 days including the three regression suites and the
  workflow wiring.
