import pandas as pd
from os import path

DATA_DIR = 'data_files/'

def get_team_form(team_name, num_matches=5):
    """Get recent form for a specific team"""
    df = pd.read_csv(path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv'), sep='\t')
    
    # Get matches where team played
    team_matches = df[
        (df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)
    ].sort_values('MatchDate', ascending=False).head(num_matches)
    
    form = []
    for _, match in team_matches.iterrows():
        if match['HomeTeam'] == team_name:
            result = match['FullTimeResult']
            if result == 'H':
                form.append('W')
            elif result == 'D':
                form.append('D')
            else:
                form.append('L')
        else:
            result = match['FullTimeResult']
            if result == 'A':
                form.append('W')
            elif result == 'D':
                form.append('D')
            else:
                form.append('L')
    
    return ''.join(form)

def get_team_form_stats(team_name, num_matches=5):
    """Get detailed form statistics for a team"""
    df = pd.read_csv(path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv'), sep='\t')
    
    # Get matches where team played
    team_matches = df[
        (df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)
    ].sort_values('MatchDate', ascending=False).head(num_matches)
    
    if len(team_matches) == 0:
        return {'wins': 0, 'draws': 0, 'losses': 0, 'points': 0, 'form_string': ''}
    
    wins = 0
    draws = 0
    losses = 0
    form = []
    
    for _, match in team_matches.iterrows():
        if match['HomeTeam'] == team_name:
            result = match['FullTimeResult']
            if result == 'H':
                wins += 1
                form.append('W')
            elif result == 'D':
                draws += 1
                form.append('D')
            else:
                losses += 1
                form.append('L')
        else:
            result = match['FullTimeResult']
            if result == 'A':
                wins += 1
                form.append('W')
            elif result == 'D':
                draws += 1
                form.append('D')
            else:
                losses += 1
                form.append('L')
    
    points = wins * 3 + draws
    
    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'points': points,
        'form_string': ''.join(form),
        'matches_played': len(team_matches)
    }