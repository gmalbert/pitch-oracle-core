"""
Fetch ClubElo ratings for the configured league's teams.

Powered by ``penaltyblog.scrapers.ClubElo`` — team names are normalised through
the shared ``pitch_oracle_core.team_mappings`` registry, so no per-script
reconciliation step is needed.

Outputs (unchanged on-disk contract):
  data_files/clubelo_ratings.csv  — full Elo history per team (tab-separated)
  data_files/clubelo_fixtures.csv — upcoming matches with H/D/A probabilities

Run: python fetch_clubelo.py
"""

import requests
import pandas as pd
import io
import time
from os import path

from penaltyblog.scrapers import ClubElo

from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.leagues import get_league_config
from pitch_oracle_core.team_mappings import (
    CLUBELO_TEAM_MAP,
    mappings_for_league,
)

DATA_DIR = 'data_files/'

# penaltyblog's ClubElo frames are snake-cased; restore the legacy contract.
_LEGACY_COLUMN_NAMES = {
    "rank": "Rank", "team": "Club", "country": "Country",
    "level": "Level", "elo": "Elo", "from": "From", "to": "To",
}


def team_map_for(league: LeagueConfig | str) -> dict[str, str]:
    """Return a league-specific ClubElo mapping supplied by the consumer."""
    config = get_league_config(league) if isinstance(league, str) else league
    return dict(config.team_aliases)


def legacy_ratings_frame(frame: pd.DataFrame, hist_name: str) -> pd.DataFrame:
    """Project a ``ClubElo.get_elo_by_team`` frame onto the legacy CSV schema."""
    restored = frame.reset_index().rename(columns=_LEGACY_COLUMN_NAMES)
    restored["HistTeam"] = hist_name
    return restored


def fetch_team_elo(clubelo_name, scraper: ClubElo | None = None):
    """Fetch full Elo history for one club via penaltyblog's ClubElo scraper."""
    scraper = scraper or ClubElo()
    try:
        df = scraper.get_elo_by_team(clubelo_name)
        if df.empty or "elo" not in df.columns:
            return None
        return df
    except Exception as e:
        print(f'  Warning: failed to fetch {clubelo_name}: {e}')
        return None


def fetch_upcoming_fixtures():
    """
    Fetch upcoming match probabilities from api.clubelo.com/Fixtures.
    Returns a DataFrame with columns: Date, HomeTeam, AwayTeam,
    ClubElo_HomeWinProb, ClubElo_DrawProb, ClubElo_AwayWinProb.
    Probabilities are summed over goal-difference outcomes (>0 home, =0 draw, <0 away).

    Note: penaltyblog's ClubElo scraper does not expose the fixtures endpoint,
    so this pull stays local.
    """
    url = 'http://api.clubelo.com/Fixtures'
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        print(f'  Warning: failed to fetch ClubElo fixtures: {e}')
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    # ClubElo fixture CSV columns:
    # Date, HomeTeam, AwayTeam, ...prob columns for each GD from <=-6 to >=6
    # Identify goal-difference probability columns (numeric or named like "-5", "0", "5" etc.)
    gd_cols = [c for c in raw.columns if c not in ('Date', 'HomeTeam', 'AwayTeam')]

    rows = []
    for _, row in raw.iterrows():
        home_prob = 0.0
        draw_prob = 0.0
        away_prob = 0.0
        for col in gd_cols:
            try:
                gd = int(col)
                p = float(row[col]) if pd.notna(row[col]) else 0.0
                if gd > 0:
                    home_prob += p
                elif gd == 0:
                    draw_prob += p
                else:
                    away_prob += p
            except (ValueError, TypeError):
                continue

        rows.append({
            'Date': row.get('Date', ''),
            'HomeTeam': row.get('HomeTeam', ''),
            'AwayTeam': row.get('AwayTeam', ''),
            'ClubElo_HomeWinProb': round(home_prob, 4),
            'ClubElo_DrawProb': round(draw_prob, 4),
            'ClubElo_AwayWinProb': round(away_prob, 4),
        })

    return pd.DataFrame(rows)


def main(league: LeagueConfig | str = 'epl'):
    config = get_league_config(league) if isinstance(league, str) else league
    team_map = team_map_for(config) or CLUBELO_TEAM_MAP
    scraper = ClubElo(team_mappings=mappings_for_league(config))
    all_frames = []
    success = 0
    fail = 0

    for hist_name, elo_slug in team_map.items():
        print(f'Fetching Elo for {hist_name} ({elo_slug})...')
        df = fetch_team_elo(elo_slug, scraper=scraper)
        if df is not None:
            all_frames.append(legacy_ratings_frame(df, hist_name))
            print(f'  {len(df)} records')
            success += 1
        else:
            print(f'  No data')
            fail += 1
        time.sleep(0.5)   # polite rate limit

    print(f'\n{success} teams fetched, {fail} not found.')

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined['From'] = pd.to_datetime(combined['From'], errors='coerce')
        combined['To'] = pd.to_datetime(combined['To'], errors='coerce')
        combined['Elo'] = pd.to_numeric(combined['Elo'], errors='coerce')
        out = path.join(DATA_DIR, 'clubelo_ratings.csv')
        combined.to_csv(out, sep='\t', index=False)
        print(f'Saved {len(combined)} Elo records to {out}')
    else:
        print('No Elo data saved.')

    # Fetch upcoming fixture probabilities
    print('\nFetching ClubElo upcoming fixture probabilities...')
    fixtures_df = fetch_upcoming_fixtures()
    if not fixtures_df.empty:
        out_fix = path.join(DATA_DIR, 'clubelo_fixtures.csv')
        fixtures_df.to_csv(out_fix, sep='\t', index=False)
        print(f'Saved {len(fixtures_df)} upcoming fixtures to {out_fix}')
    else:
        print('No upcoming fixture data fetched.')


if __name__ == '__main__':
    main()
