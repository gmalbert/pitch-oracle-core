import requests
import pandas as pd

API_KEY = '123'

def fetch_team_injuries(team_name):
    """
    Fetch injury list for a team
    Free API: https://www.thesportsdb.com/api.php
    """
    
    try:
        # Search for team
        team_url = f'https://www.thesportsdb.com/api/v1/json/{API_KEY}/searchteams.php?t={team_name}'
        team_response = requests.get(team_url)
        team_data = team_response.json()
        
        if team_data['teams']:
            team_id = team_data['teams'][0]['idTeam']
            
            # Get injuries for the team
            injuries_url = f'https://www.thesportsdb.com/api/v1/json/{API_KEY}/injuries.php?id={team_id}'
            injuries_response = requests.get(injuries_url)
            
            return injuries_response.json()
    except (requests.exceptions.RequestException, ValueError):
        # Handle API errors or invalid JSON
        return None
    
    return None

def count_key_injuries(team_name):
    """
    Count key injuries for a team (simplified - counts all injuries)
    """
    injuries_data = fetch_team_injuries(team_name)

    if injuries_data and 'injuries' in injuries_data:
        # Count injuries that are not empty
        return len([injury for injury in injuries_data['injuries'] if injury.get('strInjury')])
    return 0

def create_injury_impact_feature(home_team, away_team):
    """
    Create feature representing injury impact
    """
    # Simplified - in practice, weight by player importance
    home_injuries = count_key_injuries(home_team)
    away_injuries = count_key_injuries(away_team)

    return {
        'HomeInjuryCount': home_injuries,
        'AwayInjuryCount': away_injuries,
        'InjuryAdvantage': away_injuries - home_injuries
    }

def fetch_all_team_injuries(match_df):
    """
    Fetch injury data for all teams in the match dataframe
    Returns a dict of team -> injury count
    """
    teams = set(match_df['HomeTeam'].unique()) | set(match_df['AwayTeam'].unique())
    injury_data = {}

    for team in teams:
        count = count_key_injuries(team)
        injury_data[team] = count

    return injury_data

# Example usage
if __name__ == "__main__":
    # Test with a team
    team = "Arsenal FC"
    injuries = fetch_team_injuries(team)
    print(f"Injuries for {team}: {injuries}")

    count = count_key_injuries(team)
    print(f"Key injury count for {team}: {count}")

    # Test impact feature
    impact = create_injury_impact_feature("Arsenal FC", "Chelsea FC")
    print(f"Injury impact: {impact}")