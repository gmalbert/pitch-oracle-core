# New consumer repository runbook

This is the required path for creating a thin league repository backed by
`pitch-oracle-core`. The maintained template, not `premier-league`, is the source
for new repositories. Premier League remains the production reference for the
resulting artifact layout and application behavior.

## Recommended first consumer: Eredivisie

Use repository name `eredivisie` and league key `eredivisie`.

It is the lowest-risk pilot among the five planned leagues because it has a normal
18-team double round-robin regular season, registered football-data (`N1`), ESPN
(`ned.1`), and ClubElo (`NED_1`) identifiers, and no split-table transformation.
The separate European-qualification playoff is outside the initial prediction
scope. Shipping the regular season first exercises league isolation without also
testing a new competition format.

The official league has published its 2026/27 schedule and notes that later-round
kickoff times can still move around European commitments. ESPN currently exposes
the league under `ned.1`, which is why the baseline refreshes fixtures every day:

- [official 2026/27 Eredivisie schedule](https://eredivisie.nl/nieuws/definitief-programma-2026-27/)
- [ESPN Eredivisie fixtures](https://www.espn.com/soccer/fixtures/_/league/ned.1)

Launch v1 with the mandatory baseline sources only:

- football-data.co.uk historical results and odds;
- ESPN upcoming fixtures;
- the core shot-based xG proxy;
- shared chronological training, feature contract, cache manifest, and UI.

Do not make ClubElo, weather, referee, injury, Understat, API-Football, or live odds
a launch dependency. Add each only after its league-specific identifiers, team
aliases, coverage, and failure behavior pass a source spike. Understat does not
cover the Eredivisie and must remain disabled.

## 1. Generate the repository

From a clean, current checkout of `pitch-oracle-core` (the generated runtime still
pins an immutable release):

```bash
git pull --ff-only origin main
python scripts/bootstrap_consumer.py eredivisie ../eredivisie
cd ../eredivisie
git init
git add .
git commit -m "feat: bootstrap Eredivisie consumer"
gh repo create gmalbert/eredivisie --public --source=. --remote=origin --push
```

The bootstrap command refuses to overwrite an existing path. It creates:

- the league configuration and Streamlit entrypoint;
- immutable runtime and CI dependency pins;
- pull-request CI and the scheduled artifact workflow;
- prediction generation and strict artifact verification;
- tests and tracked artifact-directory placeholders.

Do not copy generated EPL data, models, caches, aliases, branding, or secrets.

## 2. Configure GitHub once

In **Settings → Actions → General → Workflow permissions**, select
**Read and write permissions** so the scheduled artifact job can commit one
coherent cache set. Keep branch protection enabled for human changes; allow the
repository `GITHUB_TOKEN` to push the scheduled artifact commit according to the
repository's protection policy.

No secrets are required for the baseline. Add provider secrets only when that
provider is deliberately enabled in the league configuration and workflow.

## 3. Prove the baseline locally

Use Python 3.12 or newer:

```bash
python -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements-ci.txt
./venv/bin/python -m compileall -q .
./venv/bin/python -m pytest -q
```

On Windows PowerShell, replace `./venv/bin/python` with
`./venv/Scripts/python.exe`.

The initial tests validate configuration and pin synchronization. Artifact tests
skip until the first coherent artifact set exists.

## 4. Run the first artifact build

Push the scaffold, open **Actions → Eredivisie artifact pipeline**, and run it
manually. The reusable workflow performs, in order:

1. dependency installation and `pip check`;
2. historical download and upcoming-fixture fetch;
3. point-in-time feature preparation;
4. chronological model training;
5. diagnostics and upcoming prediction generation;
6. strict cache-manifest creation;
7. consumer tests;
8. one atomic artifact commit.

Never commit a partial manual rebuild. `data_files/`, `models/`, and
`precomputed/` must advance together under the same core version.

## 5. Acceptance gate

The consumer is ready to expose when all of these are true:

- CI is green on the initial commit;
- the manual artifact workflow is green on real upstream data;
- `precomputed/cache_manifest.json` records `core_version: 1.3.1` and
  `league: eredivisie`;
- `python scripts/verify_consumer.py` succeeds;
- the app opens all seven navigation pages without a Streamlit exception;
- Playwright click-navigation reports no browser console errors;
- team names in history and fixtures normalize to the same values;
- prediction probabilities are present, bounded, and sum to one;
- chronological accuracy and log loss are plausible and recorded, without being
  compared to the old EPL random-split metrics.

## 6. Optional-source promotion rule

Promote one optional source at a time on a pull request. For each source, document
coverage, add aliases/configuration, add a failure-mode test, rebuild everything,
and compare chronological holdout metrics. Keep the source disabled if it adds no
measurable value or cannot fail safely.

For the Eredivisie, investigate optional sources in this order:

1. verified team aliases and ClubElo mappings;
2. stadium coordinates and weather;
3. API-Football supplemental data, if a paid key is justified;
4. referee and injury coverage;
5. live odds.

## 7. Core upgrades

Upgrade only to an immutable core release. Change the tag in
`requirements.txt`, `requirements-ci.txt`, the caller workflow's `uses` line,
and `core_ref` together. Then rebuild all generated artifacts and rerun the full
acceptance gate. Never point a production consumer at a branch or `main`.

## Done definition for every later consumer

A new repository is turnkey only when it can be recreated from its committed
configuration, fetch live inputs without copied EPL data, generate a strict
league-matched manifest, pass compile/tests/parity checks, and render cleanly in a
browser. League-specific competition rules must be represented in `LeagueConfig`
before that league is allowed into production.
