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

- [ ] Package client, bounded retries, typed HTTP errors, SDK-independent interface.
- [ ] Immutable versioned raw cache, seven-day correction policy, append-only lineups.
- [ ] Canonical fixture reconciliation, aliases, ambiguity and reversed-team audit.
- [ ] Versioned matches, shots, shot summaries, advanced-team and player artifacts.
- [ ] Momentum, passing networks, player/team heatmaps and coordinate semantics.
- [ ] Observation-aware team ledger, rolling/EWM shot and advanced features.
- [ ] Player strength, role scaling, minute/recency weighting, transfers and priors.
- [ ] Goalkeeper exposure, non-own goals, shrinkage and sample coverage.
- [ ] Per-team lineup selection, continuity, availability, bench and keeper context.
- [ ] Initial/24-hour/lineup/closing forecast lifecycle with explicit as-of replay.
- [ ] Independent baseline model, model contract fingerprints, deterministic fallback.
- [ ] Style interactions and walk-forward feature-family ablations.
- [ ] Bootstrap uncertainty, calibration, subgroup and missingness reports.
- [ ] Capability-specific health, schedules discrepancy and latency audits.
- [ ] Reusable CI secret, historical refresh, hourly snapshot jobs, optional failure.
- [ ] Manifest metadata, lazy filtered artifact reads, compatibility validation.
- [ ] Match Center pre/post-match analytics, Team Center trends, diagnostics.
- [ ] All seven consumer configurations and pipeline integration.
- [ ] Updated assessment, data contracts, production runbook and rollback procedure.
- [ ] Meaningful synthetic chronology, failure and replay regression tests.
- [ ] Full Python tests and py_compile across core and affected consumers.
- [ ] Playwright runtime validation of all seven apps and optional-data states.
- [ ] Real-provider pilot mapping/coverage and walk-forward evidence where credentials permit.
- [ ] Narrow commits with detailed Markdown descriptions; reviewed pushes/PR checks.

## Current evidence

Initial inspection: no local PitchAPI or lineup observation archive. Ligue 1
contains 920 odds rows at six capture times (July 7–12, 2026). This is not an
archive sufficient to replay all planned forecast stages across all leagues.
The existing `.venv312` has Python 3.12.14 and Streamlit 1.61.1; declared test
and pipeline dependencies are being restored before baseline validation.
