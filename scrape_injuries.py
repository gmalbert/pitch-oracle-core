import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import os
from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.leagues import get_league_config

def scrape_premier_injuries(league: LeagueConfig | str = 'epl'):
    """
    Fetch injury data from API-Football or return mock data for testing
    Returns DataFrame with injury information
    """
    # You'll need to get a free API key from https://www.api-football.com/
    config = get_league_config(league) if isinstance(league, str) else league
    if not config.sources.injuries:
        return pd.DataFrame()
    API_KEY = os.getenv("SPORTS_API_KEY", "")

    if not API_KEY:
        print("No API key provided - using mock injury data for testing")
        # Return some mock injury data for testing
        mock_injuries = [
            {'Player': 'Harry Kane', 'Team': 'tottenham', 'Injury': 'Hamstring', 'ExpectedReturn': '2024-02-01', 'ScrapedDate': datetime.now().strftime('%Y-%m-%d')},
            {'Player': 'Mohamed Salah', 'Team': 'liverpool', 'Injury': 'Ankle', 'ExpectedReturn': '2024-01-25', 'ScrapedDate': datetime.now().strftime('%Y-%m-%d')},
            {'Player': 'Kevin De Bruyne', 'Team': 'man_city', 'Injury': 'Knee', 'ExpectedReturn': '2024-03-01', 'ScrapedDate': datetime.now().strftime('%Y-%m-%d')},
            {'Player': 'Bruno Fernandes', 'Team': 'man_united', 'Injury': 'Thigh', 'ExpectedReturn': '2024-01-30', 'ScrapedDate': datetime.now().strftime('%Y-%m-%d')},
        ]
        return pd.DataFrame(mock_injuries)

    url = "https://v3.football.api-sports.io/injuries"

    headers = {
        'x-apisports-key': API_KEY
    }

    params = {
        'league': config.sources.api_football_league_id,
        'season': 2023  # Current season - adjust as needed
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        if data.get('response'):
            injuries = []

            for injury in data['response']:
                try:
                    player_info = injury.get('player', {})
                    team_info = injury.get('team', {})

                    injuries.append({
                        'Player': player_info.get('name', ''),
                        'Team': team_info.get('name', ''),
                        'Injury': injury.get('type', ''),
                        'ExpectedReturn': injury.get('return_date', ''),
                        'ScrapedDate': datetime.now().strftime('%Y-%m-%d')
                    })

                except Exception as e:
                    print(f"Error parsing injury record: {e}")
                    continue

            df = pd.DataFrame(injuries)

            if not df.empty:
                # Clean team names to match our data
                team_name_mapping = {
                    'Manchester United': 'man_united',
                    'Manchester City': 'man_city',
                    'Chelsea': 'chelsea',
                    'Arsenal': 'arsenal',
                    'Liverpool': 'liverpool',
                    'Tottenham': 'tottenham',
                    'Newcastle': 'newcastle',
                    'West Ham': 'west_ham',
                    'Aston Villa': 'aston_villa',
                    'Brighton': 'brighton',
                    'Crystal Palace': 'crystal_palace',
                    'Fulham': 'fulham',
                    'Wolves': 'wolves',
                    'Southampton': 'southampton',
                    'Brentford': 'brentford',
                    'Leeds': 'leeds',
                    'Everton': 'everton',
                    'Nottingham Forest': 'nottingham',
                    'Bournemouth': 'bournemouth',
                    'Burnley': 'burnley',
                    'Sheffield United': 'sheffield_united',
                    'Luton': 'luton',
                    'Ipswich': 'ipswich'
                }

                df['Team'] = df['Team'].map(team_name_mapping).fillna(df['Team'])

            print(f"Successfully fetched {len(df)} injury records from API-Football")
            return df
        else:
            print("No injury data found in API response")
            return pd.DataFrame()

    except requests.RequestException as e:
        print(f"Error fetching injury data: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error parsing injury data: {e}")
        return pd.DataFrame()

def count_team_injuries(injury_df):
    """
    Count injuries per team
    Returns dict with team injury counts
    """
    if injury_df.empty:
        return {}

    injury_counts = injury_df.groupby('Team').size().to_dict()
    return injury_counts

def create_injury_features(historical_df, injury_df):
    """
    Create injury features for match data
    """
    if injury_df.empty:
        print("No injury data available, adding zero injury features")
        historical_df['HomeInjuryCount'] = 0
        historical_df['AwayInjuryCount'] = 0
        historical_df['InjuryAdvantage'] = 0
        return historical_df

    injury_counts = count_team_injuries(injury_df)

    # Add injury counts to matches
    historical_df['HomeInjuryCount'] = historical_df['HomeTeam'].map(injury_counts).fillna(0).astype(int)
    historical_df['AwayInjuryCount'] = historical_df['AwayTeam'].map(injury_counts).fillna(0).astype(int)
    historical_df['InjuryAdvantage'] = historical_df['AwayInjuryCount'] - historical_df['HomeInjuryCount']

    print(f"Added injury features: {len(historical_df)} matches processed")
    return historical_df

if __name__ == "__main__":
    # Test the scraper
    injuries = scrape_premier_injuries()
    if not injuries.empty:
        print(injuries.head())
        print(f"\nInjury counts by team:")
        counts = count_team_injuries(injuries)
        for team, count in sorted(counts.items()):
            print(f"{team}: {count}")
    else:
        print("No injury data scraped")
