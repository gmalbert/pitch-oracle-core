# PitchAPI production runbook

## Release scope and promotion state

Version 1.5.0 supplies optional historical analytics, observation-aware feature
families, independent model bundles, forecast revisions, and shared Streamlit
pages for Belgium, Eredivisie, EPL, La Liga, Ligue 1, Scotland, and Turkey.
Portugal is outside this rollout. The primary match sources remain in place.

No PitchAPI family is promoted in this release's generated consumer caches.
Real-provider mapping audits remain below the 99.5% gate, and the October 2,
2026 backfill captures were collected after historical kickoffs. The strict
walk-forward reports therefore retain the baseline. They do not establish a
historical predictive improvement. Retrospective reports cannot promote models.

## Daily and hourly operations

Install the pinned consumer requirements on Python 3.12 or later. Set
`PITCH_ORACLE_LEAGUE` to the core league key and provide `PITCH_API_KEY` as a
server-side environment variable or protected Actions secret. The variable is
never written into raw data, browser payloads, model metadata, or log messages.

After refreshing the consumer's primary historical data and schedule:

```shell
python -m pitch_oracle_core.pitchapi.pipeline daily --league epl
python -m pitch_oracle_core.pitchapi.pipeline train --league epl
python -m pitch_oracle_core.evaluation.pitchapi_ablation --league epl
python -m pitch_oracle_core.pitchapi.pipeline hourly --league epl
```

La Liga and Ligue 1 retain their legacy preparation and training commands. Pass
`--historical-file pitchapi_historical_features.csv` to these four shared
commands so the strict mart does not replace their legacy model-ready data.

The shared daily workflow audits the primary chronology, trains the baseline,
generates diagnostics and prediction caches, and validates the cache manifest.
Hourly jobs refresh the primary fixture schedule, poll lineups within 48 hours,
capture successful forecasts within seven days, publish the optional index, and
rebuild the appropriate consumer manifest. They do not refit models. Both jobs
use the same per-repository concurrency group to avoid competing cache commits.

When there are no eligible fixtures in the seven-day forecast window, the
hourly report records zero fixtures and creates no issue timestamp. The live
October 2 check encountered this condition in all seven consumers. Synthetic
tests cover the issue, confirmation, correction, reschedule and closing stages.

## Backfills and corrections

```shell
python -m pitch_oracle_core.pitchapi.pipeline daily --league epl --seasons 2025/2026 --request-budget 100
python -m pitch_oracle_core.pitchapi.pipeline daily --league epl --all-seasons --request-budget 100
```

Every provider fixture must first reconcile against a primary historical or
upcoming identity. All-season mode visits the catalogue from newest to oldest.
The budget limits new analytics endpoint calls; HTTP retries remain separately
bounded. Repeating a partial run reuses saved responses and advances the
backfill. Do not invent historical capture times or widen kickoff tolerances to
make an audit pass. Future fixtures outside the primary schedule horizon are
explicitly reported outside its comparison window; unresolved fixtures inside
that window still fail the mapping gate.

Completed responses are checked daily for seven days after the completion
estimate of kickoff plus 120 minutes, then retained. Use `--refresh` for an
explicit correction after that period. Ordinary endpoint absence has a daily
negative cache during the correction window. Authentication, rate-limit or
systemic transport failures stop further endpoint requests and leave existing
analytics intact. Baseline preparation and serving continue.

## Data and model contracts

`data_files/pitchapi_cache` contains immutable response observations and
content-addressed payload blobs. Actual observation times follow successful
responses. Unchanged hourly lineup captures retain separate observations while
sharing the payload blob. Provider schedules, capability failures and mapping
audits are also recorded.

Normalized revisions live in `pitchapi_shots.parquet`,
`pitchapi_advanced_team.parquet`, `pitchapi_player_match.parquet`,
`pitchapi_lineup_snapshots.parquet`, and the momentum/network/heatmap artifacts.
`pitchapi_response_revisions.parquet` records complete successful responses,
including empty corrections. Removed team/player rows become null deletion
observations, so an earlier cutoff remains reproducible. A complete empty shot
response is known zero; missing analytics remain unknown.

The strict historical mart and live builder share fixture identities, rolling
windows and observation eligibility. Post-match observations must belong to an
earlier source fixture and be known at the target cutoff. Lineup status is
selected independently for each team. Formation slots never invent player
roles. Unknown roles and uncertain keeper attribution stay unknown.

`models/pitchapi_baseline.pkl` always contains an independently fitted model.
Promoted bundles have their own exact feature order, fitted imputation and
calibration, league ID, training availability, validation evidence and contract
fingerprint. A missing bundle, stale capability audit, failed mapping gate,
missing per-side state or stale lineup selects the baseline with a recorded
reason. Feature policy version 4 refuses incompatible old artifacts.

`pitchapi_forecast_revisions.parquet` preserves actual successful initial,
24-hour, lineup, hourly and schedule-update issues. Either team's confirmed XI
change creates a lineup revision. Closing references the last successful actual
pre-kickoff issue; no forecast or timestamp is invented at kickoff. Cancellation
and changed kickoff versions exclude obsolete closes.

The optional index `pitchapi_artifacts.json` hashes each published analytics
file. Missing or corrupt optional files degrade only dependent views. Required
prediction artifacts remain strict. Legacy La Liga and Ligue 1 pages may overlay
`pitchapi_upcoming_predictions.csv` only after a family is configured with its
evidence ID, with an intact hash and an issue no more than two hours old. Their
historical prediction logs are retained. The default release uses their existing
baseline forecasts.

Generated `data_files`, `models` and `precomputed` paths use Git `-text`
attributes. Their integrity records hash the exact stored bytes; checkout must
preserve those bytes on Linux and Windows. After changing these attributes on an
existing consumer, rebuild the optional index and required manifest, then stage
the artifact paths with `git add --renormalize`. Verify their hashes against the
staged files before publishing the coherent bundle.

## Promotion and rollback

Evaluate A0, individual A1–A8 families and an explicitly selected A9 combination
on paired chronological folds. Promotion requires at least 0.5% relative
log-loss or 1% Brier improvement, uncertainty supporting that improvement,
calibration and subgroup stability, sufficient coverage, at least 200 paired
fixtures and ten independent week blocks, and no chronology violations.
Missing predicted lineups do not prevent league operation.

Only enable the combination supported by the strict report's evidence ID in
`data_files/pitchapi_feature_config.json`, then train both independent bundles
and regenerate the consumer cache and manifest together. Analytics collection
alone never enables a family. Model metadata and Feature validation show the
active families and evidence.

For rollback, set the league's enabled family list to empty, retain its raw
observations, regenerate predictions/index/manifest, and run consumer validation.
Alternatively restore the prior compatible core pin and its complete artifact
set. Do not pair an old feature contract with a newly trained model. A missing
credential is an optional-provider state, not a reason to delete primary caches.

## Verification record

The final local full core suite passed 421 tests, including the schedule and
outage regressions. All seven consumer
test suites passed. Real payload pilots passed 63 Playwright section/mobile
checks; all seven actual consumer apps passed 84 production, section, mobile,
team and validation checks. Version 1.5.0 built and imported from an isolated
installed wheel. All 440 executable Python files compiled.

One pre-existing file, `ligue-1/docs/streamlit_ligue_odds_app.py`, contains
Markdown prose and fails Python compilation. It is documentation rather than
an imported application module and was left unchanged. Six shared-core CI jobs
passed on Linux, Windows and macOS with Python 3.12/3.13; all seven consumer PR
CI checks passed after reconciling main. Review and CI links are in the
[rollout review record](pitchapi-rollout-prs.md). GitHub secret setup remains
pending explicit user approval.
