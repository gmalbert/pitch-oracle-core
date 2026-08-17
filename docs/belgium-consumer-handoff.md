# Belgian Pro League Consumer — Handoff & Learnings

Status: **launched** · Core **v1.3.27** · PR #11 merged

This document captures what it took to bring the `belgium-soccer` consumer
from a bare scaffold to a green, locally-runnable app, the root causes we
hit, and the remaining follow-ups. It complements
`docs/new-consumer-repository.md` and `docs/model-improvement-handoff.md`.

---

## Outcome

- `gmalbert/belgium-soccer` is a complete Pro League consumer: bootstrap
  scaffold, `config.py`, artifact pipeline, committed prediction cache.
- The scheduled **Belgian Pro League artifact pipeline** is green and
  committed a 63-fixture prediction cache.
- The Streamlit app runs locally on Windows and renders all pages
  (Overview, Predictions, Standings, Team Deep Dive, Statistics, Model Lab,
  Raw Data) with real data.
- Root-cause fixes shipped in core **v1.3.27** (PR #11):
  - walk-forward **Poisson production candidate** (unblocks the release gate)
  - phase-aware league support (Belgium/Scotland split formats)
  - shared theme chooser + branding fixes
  - Odds-API.io credential plumbing
  - turkey consumer prep (verified `tur.1` slug + aliases)

---

## Key learnings

### 1. The release gate failed because the no-odds model can't beat baseline for Belgium

The first pipeline run failed at the **"Audit chronology and no-odds model
candidate"** step with exit code 2. The audit's release gate requires the
production candidate to beat the rolling class-prior baseline on **both**
log loss and Brier score. Belgium's no-odds logistic scored
`log_loss 1.120 vs baseline 1.081` — worse on log loss — so the gate
correctly refused to ship.

Experiments that did **not** help:
- Tuning logistic `C` from 0.05 → 1.0 (best was still 1.100 log loss)
- Dropping >99%-empty optional-source columns (17 dropped; still failed)
- Dropping the partial 2022 season or playoff matches

The features were clean (0.2% missing) and chronologically valid — the
logistic features just lack signal for Belgium's data. **This is not a
consumer-repo bug; it's a core modeling capability limit.**

### 2. The walk-forward Poisson model is the fix

The point-in-time Poisson goals model (team attack/defense rates from
completed history only) beats the baseline for Belgium:
`log_loss 1.044 < 1.081`, `brier 0.628 < 0.654`. It carries signal the
logistic features don't, because goal rates aggregate more robustly than
the engineered form features on a small league.

Implementation (core v1.3.27):
- `pitch_oracle_core/model_audit.py` — added `poisson` candidate to
  `evaluate_feature_ablation` (skips gracefully when goal columns absent).
- `pitch_oracle_core/audit_cli.py` — gate passes when the **best** of
  `no_odds` / `poisson` beats baseline; records `production_candidate`.
- `models/poisson_evaluation.py` — new `predict_upcoming_outcomes()`
  returns 1X2 probabilities + expected goals for fixtures.
- `pitch_oracle_core/predictions.py` — `production_probabilities()`
  dispatch: `no_odds` loads `ensemble_model.pkl`, `poisson` computes
  walk-forward; wires expected goals into the goal-market UI.
- `train_models.py` / `cache.py` / consumer `precompute_predictions.py` /
  `verify_consumer.py` — all candidate-aware; metadata must match the
  audit's `production_candidate`.

Validation: Belgium passes (`production_candidate: poisson`), Eredivisie
still passes (Poisson actually scores better there too: 0.997 vs 1.031
log loss). Full core suite: **111 tests pass**.

### 3. Windows line endings break the cache integrity gate

After the pipeline went green, running the app locally failed with:

```
RuntimeError: Cache artifact 'model_metadata' failed integrity validation
```

Root cause: the workflow commits JSON with LF (Linux), but a Windows
checkout converts them to CRLF. `cache_manifest.json` records byte-exact
SHA-256 hashes, so CRLF files fail validation. The `.gitattributes` only
forced `*.csv eol=lf`, not `*.json`.

Fix (committed to belgium-soccer and the consumer template):

```
*.csv text eol=lf
*.json text eol=lf
```

Also re-checked-out the affected JSONs locally. **Any future consumer
shipped with the template now inherits this fix.** Note: `.pkl` artifacts
are binary and unaffected.

### 4. Workflow bot commits can race your local pushes

The scheduled artifact pipeline auto-commits cache refreshes to `main`.
A local push can be rejected with "fetch first" when a workflow commit
landed in between. Fix: `git fetch && git rebase origin/main && git push`.

### 5. gh CLI is the auth path on this machine

Plain HTTPS git operations fail without credentials; the `gh` CLI token
(scope `repo`, `workflow`) via Git Credential Manager is what makes
`git push`/`gh` work. Use `gh run watch <id>` and `gh api
repos/<owner>/<repo>/actions/jobs/<job>/logs` to inspect pipeline logs.

---

## Current repo state

**belgium-soccer** (`main`, clean, in sync with origin):
- `05840fd` fix: keep generated JSON cache artifacts byte-stable on Windows
- `596797d` / `91ff6ab` chore: refresh precomputed prediction cache
- `551ce6d` fix: adopt core v1.3.27 with Poisson production model
- `4fec603` feat: bootstrap Belgian Pro League consumer
- `0f11a6e` Initial commit

**pitch-oracle-core**: PR #11 merged to `main` (`b523f34`), tag **v1.3.27**
pushed. Local branch `agent/turnkey-consumer-guide` clean; two pre-existing
untracked `validation-*/` dirs left untouched.

---

## Next steps

1. **Bump `netherlands-soccer` to core v1.3.27.** It still pins v1.3.26 and
   works, but the Poisson candidate scores better on its data
   (0.997 vs 1.031 log loss). Follow the core-upgrade section of
   `docs/new-consumer-repository.md` (requirements, workflow `uses`,
   `core_ref`, rebuild, re-run acceptance gate).

2. **Watch the next scheduled Belgium pipeline.** Confirm the cache refresh
   stays green and the committed manifest still matches on a fresh clone.
   If the seed (pre-season) 2026 data is thin, expect the gate to remain
   on `poisson` — that is correct behavior.

3. **Optional-source promotion (one at a time).** Per the runbook, add
   ClubElo/weather/referee/injury only after league-specific coverage and
   failure-mode tests exist. Belgium currently runs baseline sources only;
   its `LeagueConfig` has no stadium coordinates or weather timezone yet.

4. **Consider documenting the Poisson-candidate gate** in
   `docs/new-consumer-repository.md` (the runbook still describes the gate
   as no-odds-only). The gate semantics changed in v1.3.27.

5. **Clean up the two untracked `validation-*/` dirs** in
   `pitch-oracle-core` when no longer needed (pre-existing, not part of
   this work).

6. **If another Windows developer clones a consumer:** the `.gitattributes`
   fix handles JSON; if a future artifact type is added to the manifest,
   extend `eol=lf` to it and re-run the full acceptance gate.

---

## Local run commands

```powershell
cd C:\Users\gmalb\Downloads\belgium-soccer
git pull --ff-only origin main
venv\Scripts\streamlit run predictions.py      # http://localhost:8501
venv\Scripts\python -m pytest -q               # contract + artifact tests
venv\Scripts\python scripts\verify_consumer.py # strict artifact + model gate
```

Do **not** re-run `scripts\bootstrap_local.py` unless a full rebuild from
live data is intended (re-downloads history, retrains, takes a while).
The committed cache is already valid.
