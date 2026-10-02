# PitchAPI rollout reviews

The full implementation is published as eight draft pull requests. The seven
consumers pin shared core 1.5.0 at
`e31c5b9703ef05bad486fc348894fedd64655033`. Later core documentation and template
changes do not change that validated runtime revision. Merge the shared core
before the dependent consumer rollout.

| Repository | Draft review | Automated validation |
| --- | --- | --- |
| Shared core | [PR #30](https://github.com/gmalbert/pitch-oracle-core/pull/30) | [Six Python 3.12/3.13 jobs](https://github.com/gmalbert/pitch-oracle-core/actions/runs/37032581483) passed on Linux, Windows and macOS |
| Belgium | [PR #13](https://github.com/gmalbert/belgium-soccer/pull/13) | [Consumer CI](https://github.com/gmalbert/belgium-soccer/actions/runs/37035652291) passed |
| La Liga | [PR #5](https://github.com/gmalbert/la-liga/pull/5) | [Consumer CI](https://github.com/gmalbert/la-liga/actions/runs/37035666837) passed |
| Ligue 1 | [PR #4](https://github.com/gmalbert/ligue-1/pull/4) | [Consumer CI](https://github.com/gmalbert/ligue-1/actions/runs/37035676752) passed |
| Eredivisie | [PR #18](https://github.com/gmalbert/netherlands-soccer/pull/18) | [Consumer CI](https://github.com/gmalbert/netherlands-soccer/actions/runs/37035700655) passed |
| EPL | [PR #70](https://github.com/gmalbert/premier-league/pull/70) | [Consumer CI](https://github.com/gmalbert/premier-league/actions/runs/37035725614) passed |
| Scotland | [PR #18](https://github.com/gmalbert/scotland-premiership/pull/18) | [Consumer CI](https://github.com/gmalbert/scotland-premiership/actions/runs/37035744560) passed |
| Turkey | [PR #13](https://github.com/gmalbert/turkey-soccer/pull/13) | [Consumer CI](https://github.com/gmalbert/turkey-soccer/actions/runs/37035766400) passed |

All consumer branches include current `main`, preserving upstream source fixes
and resolving older nightly artifact conflicts to the newly validated coherent
bundle. Generated artifact paths preserve exact bytes across Git checkouts;
all required and optional descriptors were checked against staged Git bytes.

Local evidence includes 421 core tests, all seven consumer suites, 84 checks
across actual apps, 63 live-payload browser checks, 440 executable Python source
compilations, a version 1.5.0 wheel build/import and dependency compatibility.
The merged Turkey suite now passes ten tests, including its upstream ESPN
date-range fallback regression. Existing unrelated local edits are excluded.

No enhanced family is promoted. Live mapping coverage is below the strict gate,
and current backfills do not prove historical pre-kickoff availability. The
independently fitted baseline and existing production selections remain the
serving fallback. Live hourly pilots had no eligible fixtures and created no
artificial issues. Future real captures support strict replay and promotion.

`PITCH_API_KEY` is not yet configured in the seven consumer GitHub repositories.
Automatic approval review rejected transferring the local credential because
explicit authorization for those destinations was missing. The user approval
question remains pending. Provider collection activates after that approval and
secret configuration; baseline workflows continue without the optional key.

Operational commands, release gates and rollback are in the
[production runbook](pitchapi-production-runbook.md). Original planning files
remain reference material; implementation follows the accepted release choices
in the [implementation record](pitchapi-implementation-status.md).
