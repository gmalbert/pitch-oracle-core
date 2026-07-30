# Architecture Decision: How to Build the Other 5 Leagues

## The three options

### Option A — Copy/paste `premier-league` into 5 new repos
Fork the repo 5 times (`scottish-premiership`, `eredivisie-predictions`, etc.), rename
files, swap the football-data.co.uk `Div` code, redeploy.

**Why this is the wrong default here, specifically:** `premier-league` isn't a script,
it's a small platform — ensemble + NN + LSTM training, Poisson diagnostics, referee
scraping, injury scraping, weather, PDF reports, prediction tracking, precomputed DB,
CI. Every one of `analyze_referee_impact.py`, `train_models.py`,
`precompute_database.py`, `evaluate_poisson.py`, `generate_pdf_report.py`,
`track_predictions.py`, and the test suite is **league-agnostic logic that happens to
have EPL assumptions hardcoded in it** (team names, `Div=E0`, ESPN slug `eng.1`,
stadium coordinates, referee-source URLs). Copy/pasting means:

- A bug fix or model improvement (e.g. your next Poisson tuning pass) has to be
  manually ported 6 times (5 new + the original) or it silently drifts
- 6x the Streamlit Community Cloud deployments, 6x the Cloudflare DNS/tunnel
  configs, 6x the GitHub Actions minutes
- Team-name mapping, stadium geocoding, and referee/injury scraper targets have to be
  rebuilt from scratch per league even though the *code shape* is identical

### Option B — Shared core package + thin per-league repos (recommended)
Extract the reusable ~70% of `premier-league` into a new repo,
`pitch-oracle-core`, installed by each league repo via
`pip install git+https://github.com/gmalbert/pitch-oracle-core.git@vX.Y.Z`. Each
league then gets its own small repo containing only what's actually different.

**What moves into `pitch-oracle-core`:**
- `train_models.py`, `model_optimization.py`, `optimize_model.py`,
  `benchmark_hyperparameters.py`, `compare_model_features.py`,
  `feature_importance_analysis.py` — the ensemble/NN/LSTM training pipeline,
  parameterized by a `league_code` and a data directory
- `evaluate_poisson.py` + `test_poisson_evaluation.py` — pure math, zero
  league-specific assumptions
- `precompute_database.py` + `test_precomputed_data.py`
- `generate_pdf_report.py`, `track_predictions.py`
- `fetch_clubelo.py`, `fetch_understat_xg.py` (call site becomes conditional — see
  the data source matrix), `fetch_api_football.py`, `bzzoiro_football_api.py`
- `combine_raw_data.py`, `prepare_model_data.py`, `team_name_mapping.py` (becomes a
  per-league lookup table passed in, not a hardcoded dict)
- The Streamlit UI shell itself (tabs, chart components, `footer.py`) as a
  reusable app factory, so each league's `*.py` entrypoint is ~30 lines that
  imports the shell and passes config

**What stays in each thin league repo:**
- `config.py` — league code, football-data.co.uk `Div`, ESPN slug, ClubElo country
  code, ordinal/playoff structure flags (see `04-per-league-notes.md`)
- `data_files/` — that league's historical CSVs and any manually-curated
  stadium/team lookup tables
- `scrape_referees.py` / `scrape_injuries.py` **only if** a working source exists for
  that league (several don't — see the data matrix)
- `entrypoint.py` — the Streamlit page that wires config into the shared shell
- Its own `README.md`, `requirements.txt` (pinned to a `pitch-oracle-core` version),
  and Streamlit Cloud deployment

**Tradeoffs:** more upfront work (you have to actually do the extraction before
league #1 ships) and you now have real package-versioning discipline to maintain
(tag releases, don't break the interface out from under 5 consumers). This is the
same tradeoff you already accepted with granitestateappeals.com vs.
strictscrutiny.com being separate deployments rather than duplicated codebases —
just applied one layer deeper.

### Option C — Single multi-league monorepo, one Streamlit app, league selector
Everything in one repo, one deployment, a dropdown that switches league context.
Lowest total infrastructure (one Streamlit Cloud app, one domain, one Cloudflare
config), and it's the fastest path to "all 5 leagues live."

**Why not this one:** it breaks the pattern you've already established across your
portfolio (granitestateappeals.com and strictscrutiny.com are deliberately separate
apps/domains, not a jurisdiction dropdown on one app), and Streamlit Community
Cloud's free-tier memory ceiling (1 GB) gets tight fast once you're holding
ensemble+NN+LSTM models and precomputed data for 6 leagues in one process, even with
lazy loading. It's also a worse SEO/branding outcome if you want each league to be a
discoverable, linkable product the way pitch-oracle.com is for EPL.

## Recommendation

**Option B.** It costs you one extraction sprint up front (roughly the "Phase 0" in
the implementation plan) but every league after the first thin repo takes a fraction
of the time `premier-league` originally took, and fixes/improvements propagate
everywhere by bumping a version pin. Given you're planning 4 more leagues (Turkey
being the 5th if you count it, though Süper Lig was in your list — 4 *new* repos:
Scotland, Netherlands, Portugal, Belgium, Turkey = 5), this pays for itself by league
#2.

One addition worth scoping into Phase 0: the Python package
[`soccerdata`](https://soccerdata.readthedocs.io/) already ships pre-built,
maintained connectors for ClubElo, FBref, football-data.co.uk ("MatchHistory"),
ESPN, WhoScored, and Sofascore, keyed by a `{country}-{league}` identifier
(confirmed working example: `NED-Eredivisie` → `ClubElo: NED_1`,
`MatchHistory: N1`, `ESPN: ned.1`). Swapping your custom `fetch_clubelo.py` /
parts of `fetch_upcoming_fixtures.py` for `soccerdata` calls inside
`pitch-oracle-core` would remove a real chunk of the per-league scraper
maintenance burden — worth a half-day spike before you commit to hand-rolling
5 more sets of source-specific fetchers.
