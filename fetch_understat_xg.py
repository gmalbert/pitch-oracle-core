"""
Fetch expected-goals (xG) match data from Understat.

Powered by ``penaltyblog.scrapers.Understat`` — fixes the four known bugs in
the legacy implementation (missing headers, undocumented cookies, dropped
forecast columns, no team-name normalisation).

Outputs (additive contract — new forecast columns added, existing unchanged):
  data_files/understat_xg.csv — match-level HomeXG, AwayXG, and forecasts per season

Run: python fetch_understat_xg.py
"""

import time
import pandas as pd
from os import path
from collections.abc import Mapping

from penaltyblog.scrapers import Understat

from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.leagues import get_league_config
from pitch_oracle_core.team_mappings import (
    UNDERSTAT_TEAM_MAP,
    penaltyblog_competition,
)

DATA_DIR = 'data_files/'


def pb_fixtures_to_legacy_rows(
    fixtures: pd.DataFrame,
    year: int,
    name_map: Mapping[str, str],
) -> list[dict]:
    """Project a ``penaltyblog`` Understat fixtures frame onto the legacy CSV rows.

    ``fixtures`` is the output of ``Understat.get_fixtures()`` with columns
    ``team_home``, ``team_away``, ``xg_home``, ``xg_away``, ``forecast_w``,
    ``forecast_d``, ``forecast_l``, ``datetime``, etc.  Team names are the raw
    Understat display names (no ``team_mappings`` was passed to the scraper).
    """
    rows: list[dict] = []
    for _, row in fixtures.iterrows():
        home_raw = str(row["team_home"])
        away_raw = str(row["team_away"])
        rows.append({
            "MatchDate": pd.to_datetime(row["datetime"], errors="coerce"),
            "HomeTeam_Understat": home_raw,
            "AwayTeam_Understat": away_raw,
            "HomeTeam": name_map.get(home_raw, home_raw),
            "AwayTeam": name_map.get(away_raw, away_raw),
            "HomeXG_Understat": round(float(row["xg_home"]), 4),
            "AwayXG_Understat": round(float(row["xg_away"]), 4),
            "Forecast_HomeWin": round(float(row["forecast_w"]), 4),
            "Forecast_Draw": round(float(row["forecast_d"]), 4),
            "Forecast_AwayWin": round(float(row["forecast_l"]), 4),
            "Season": year,
        })
    return rows


def main(league: LeagueConfig | str = 'epl'):
    config = get_league_config(league) if isinstance(league, str) else league
    if not config.sources.understat:
        raise RuntimeError(f'Understat is unavailable for {config.display_name}')

    competition = penaltyblog_competition(config, "understat")
    name_map = dict(UNDERSTAT_TEAM_MAP, **config.team_aliases)
    all_matches: list[dict] = []
    # EPL xG data available from 2014/15 season (year=2014)
    seasons = range(2014, 2026)

    for year in seasons:
        season_label = f"{year}-{year + 1}"
        print(f'Fetching xG for {season_label}...')
        try:
            scraper = Understat(competition, season_label)
            fixtures = scraper.get_fixtures()
            rows = pb_fixtures_to_legacy_rows(fixtures, year, name_map)
            print(f'  {len(rows)} completed matches')
            all_matches.extend(rows)
        except Exception as e:
            print(f'  Warning: could not fetch season {season_label}: {e}')
        time.sleep(1.5)   # polite rate limit

    if not all_matches:
        print('No xG data fetched.')
        return

    df = pd.DataFrame(all_matches)
    df = df.dropna(subset=['MatchDate'])
    df['MatchDate'] = df['MatchDate'].dt.normalize()   # date only, no time
    df = df.sort_values('MatchDate').reset_index(drop=True)

    out = path.join(DATA_DIR, 'understat_xg.csv')
    df.to_csv(out, sep='\t', index=False)

    print(f'\nSaved {len(df)} match xG records to {out}')
    print(f'Seasons covered: {df["Season"].min()}\u2013{df["Season"].max()}')
    print(f'Columns: {list(df.columns)}')


if __name__ == '__main__':
    main()
