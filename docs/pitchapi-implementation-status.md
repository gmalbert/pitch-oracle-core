# PitchAPI implementation and verification record

This record tracks the full expansion defined in the three PitchAPI planning
documents. Unchecked work remains part of the requested implementation.

## Agreed release behavior

- Scope: historical modeling, lineup-aware forecasts, and analytics pages in
  Eredivisie, EPL, La Liga, Ligue 1, Scotland, Belgium, and Turkey.
- Poll lineups hourly. Either team confirming or materially correcting an XI
  creates an immutable forecast revision. Closing uses the final successful
  hourly pre-kickoff revision and displays its actual timestamp.
- Retain a separately trained baseline model for automatic, recorded fallback.
- Recheck completed payloads daily for seven days; preserve changed versions;
  support explicit refresh afterward.
- Promote features independently by league. Initial promotion requires 0.5%
  relative log-loss or 1% Brier improvement, statistical evidence, stable
  calibration, coverage, subgroup checks, and no chronology violations.
- Missing predicted lineups do not prevent a league from operating.
- Historical backfills are retrospective experiments unless actual archived
  observation timestamps prove availability at the forecast cutoff. Never
  fabricate capture timestamps. New snapshot collection supports strict replay.

## Required delivery and evidence

- [x] Package client, bounded retries, typed HTTP errors, SDK-independent interface.
- [x] Immutable versioned raw cache, seven-day correction policy, append-only lineups.
- [x] Canonical fixture reconciliation, aliases, ambiguity and reversed-team audit.
- [x] Versioned matches, shots, shot summaries, advanced-team and player artifacts.
- [x] Momentum, passing networks, player/team heatmaps and coordinate semantics.
- [x] Observation-aware team ledger, rolling/EWM shot and advanced features.
- [x] Player strength, role scaling, minute/recency weighting, transfers and priors.
- [x] Goalkeeper exposure, non-own goals, shrinkage and sample coverage.
- [x] Per-team lineup selection, continuity, availability, bench and keeper context.
- [x] Initial/24-hour/lineup/closing forecast lifecycle with explicit as-of replay.
- [x] Independent baseline model, model contract fingerprints, deterministic fallback.
- [x] Style interactions and walk-forward feature-family ablations.
- [x] Bootstrap uncertainty, calibration, subgroup and missingness reports.
- [x] Capability-specific health, schedules discrepancy and latency audits.
- [x] Reusable CI secret, historical refresh, hourly snapshot jobs, optional failure.
- [x] Manifest metadata, lazy filtered artifact reads, compatibility validation.
- [x] Match Center pre/post-match analytics, Team Center trends, diagnostics.
- [x] All seven consumer configurations and pipeline integration.
- [x] Updated assessment, data contracts, production runbook and rollback procedure.
- [x] Meaningful synthetic chronology, failure and replay regression tests.
- [x] Full Python tests and py_compile across core and affected consumers.
- [x] Playwright runtime validation of all seven apps and optional-data states.
- [x] Real-provider pilot mapping/coverage and walk-forward evidence where credentials permit.
- [x] Narrow commits with detailed Markdown descriptions; branches pushed and draft PRs attached.
- [x] All seven consumer post-push CI checks passed after reconciling main.
- [ ] Protected GitHub Actions PitchAPI secret configured in each consumer (approval requested).

## Current evidence

Initial inspection: no local PitchAPI or lineup observation archive. Ligue 1
contains 920 odds rows at six capture times (July 7–12, 2026). This is not an
archive sufficient to replay all planned forecast stages across all leagues.
The existing `.venv312` has Python 3.12.14 and Streamlit 1.61.1. The declared
test and pipeline dependencies were restored and checked before validation.


## October 2 local rollout evidence

- Implemented package pipelines, complete response corrections, bounded and
  resumable backfills, independent fallback, all eight feature families,
  observation-aware replay and analytics across seven consumers.
- Refreshed primary histories and schedules. The five shared consumers passed
  their baseline chronology/model release gates. La Liga and Ligue 1 retain
  history from 2015 and use a separate strict PitchAPI mart.
- All seven consumer test suites passed. The final core full suite passed 421 tests,
  including the schedule and outage regressions.
- Seven real-payload pilots passed 63 Playwright checks. The seven actual apps
  passed 84 checks across production pages, match sections, mobile, team
  analytics and feature validation. Replaced the standalone apps' blocked iframe
  navigation with native browser timezone context.
- Built version 1.5.0 and imported its pipelines from an isolated wheel install.
  Compiled 440 executable Python sources. A pre-existing Ligue 1 `.py`
  documentation file contains Markdown and is not executable; it was preserved.
- Live current-season captures and strict ablation reports are published in the
  local consumer data directories. Mapping gates remain below threshold and
  historical captures do not prove past availability. No enhanced model is
  promoted. Actual hourly refreshes had zero fixtures in the next-seven-day
  window and therefore fabricated no issue records.

Operational commands, data contracts, release limits and rollback instructions
are in [the production runbook](pitchapi-production-runbook.md). The original
planning documents remain reference material; implementation choices follow the
user’s accepted decisions recorded above.

All eight draft reviews and automated validation links are in the
[rollout review record](pitchapi-rollout-prs.md). Consumer merge conflicts were
resolved against current main; upstream source fixes were retained. Exact staged
artifact hashes were verified after disabling checkout newline normalization for
generated data/model/cache paths. Protected secret configuration is the remaining
activation step awaiting user approval.
