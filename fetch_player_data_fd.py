import requests
import pandas as pd

API_KEY = 'your_api_key'  # Get from https://www.football-data.org/

def fetch_team_players(team_name):
    """
    Fetch player list for a team using Football-Data.org API
    Note: Injuries not directly available in free API, this gets basic player info
    """
    
    headers = {'X-Auth-Token': API_KEY}
    
    try:
        # First, get PL competition teams
        teams_url = 'https://api.football-data.org/v4/competitions/PL/teams'
        teams_response = requests.get(teams_url, headers=headers)
        teams_data = teams_response.json()
        
        # Find team by name
        team_id = None
        for team in teams_data.get('teams', []):
            if team_name.lower() in team['name'].lower() or team_name.lower() in team['shortName'].lower():
                team_id = team['id']
                break
        
        if team_id:
            # Get team details including squad
            team_url = f'https://api.football-data.org/v4/teams/{team_id}'
            team_response = requests.get(team_url, headers=headers)
            team_data = team_response.json()
            
            return team_data
        
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return None
    
    return None

def count_team_players(team_name):
    """
    Count players in team squad (proxy for team strength)
    """
    team_data = fetch_team_players(team_name)
    
    if team_data and 'squad' in team_data:
        return len(team_data['squad'])
    return 0

def create_player_impact_feature(home_team, away_team):
    """
    Create feature based on squad size (simplified proxy for team strength)
    """
    home_players = count_team_players(home_team)
    away_players = count_team_players(away_team)
    
    return {
        'HomeSquadSize': home_players,
        'AwaySquadSize': away_players,
        'SquadSizeAdvantage': home_players - away_players
    }

def fetch_all_team_player_data(match_df):
    """
    Fetch player data for all teams in the match dataframe
    Returns a dict of team -> squad size
    """
    teams = set(match_df['HomeTeam'].unique()) | set(match_df['AwayTeam'].unique())
    player_data = {}

    for team in teams:
        size = count_team_players(team)
        player_data[team] = size

    return player_data

# Example usage
if __name__ == "__main__":
    # Test with a team
    team = "Arsenal"
    players = fetch_team_players(team)
    print(f"Players for {team}: {players}")

    count = count_team_players(team)
    print(f"Squad size for {team}: {count}")

    # Test impact feature
    impact = create_player_impact_feature("Arsenal", "Chelsea")
    print(f"Player impact: {impact}")