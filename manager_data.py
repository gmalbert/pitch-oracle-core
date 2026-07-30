# manager_data.py
"""
Manager and tactical data for Premier League teams.
Contains historical manager performance statistics and tactical preferences.
"""

MANAGER_RECORDS = {
    'Pep Guardiola': {
        'WinRate': 0.73,
        'GoalsPerGame': 2.4,
        'PreferredFormation': '4-3-3',
        'TacticalFlexibility': 0.8,
        'DefensiveSolidity': 0.85,
        'AttackingThreat': 0.90
    },
    'Jurgen Klopp': {
        'WinRate': 0.65,
        'GoalsPerGame': 2.2,
        'PreferredFormation': '4-3-3',
        'TacticalFlexibility': 0.7,
        'DefensiveSolidity': 0.75,
        'AttackingThreat': 0.85
    },
    'Mikel Arteta': {
        'WinRate': 0.58,
        'GoalsPerGame': 1.8,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.75,
        'DefensiveSolidity': 0.80,
        'AttackingThreat': 0.70
    },
    'Erik ten Hag': {
        'WinRate': 0.52,
        'GoalsPerGame': 1.6,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.65,
        'DefensiveSolidity': 0.70,
        'AttackingThreat': 0.75
    },
    'Thomas Tuchel': {
        'WinRate': 0.62,
        'GoalsPerGame': 1.9,
        'PreferredFormation': '3-4-2-1',
        'TacticalFlexibility': 0.80,
        'DefensiveSolidity': 0.78,
        'AttackingThreat': 0.82
    },
    'Antonio Conte': {
        'WinRate': 0.55,
        'GoalsPerGame': 1.4,
        'PreferredFormation': '3-5-2',
        'TacticalFlexibility': 0.60,
        'DefensiveSolidity': 0.88,
        'AttackingThreat': 0.65
    },
    'Unai Emery': {
        'WinRate': 0.48,
        'GoalsPerGame': 1.5,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.70,
        'DefensiveSolidity': 0.72,
        'AttackingThreat': 0.68
    },
    'Roberto De Zerbi': {
        'WinRate': 0.50,
        'GoalsPerGame': 1.7,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.75,
        'DefensiveSolidity': 0.65,
        'AttackingThreat': 0.78
    },
    'Sean Dyche': {
        'WinRate': 0.42,
        'GoalsPerGame': 1.2,
        'PreferredFormation': '4-4-2',
        'TacticalFlexibility': 0.55,
        'DefensiveSolidity': 0.82,
        'AttackingThreat': 0.58
    },
    'Andoni Iraola': {
        'WinRate': 0.45,
        'GoalsPerGame': 1.3,
        'PreferredFormation': '4-3-3',
        'TacticalFlexibility': 0.68,
        'DefensiveSolidity': 0.75,
        'AttackingThreat': 0.62
    },
    'Marco Silva': {
        'WinRate': 0.47,
        'GoalsPerGame': 1.4,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.72,
        'DefensiveSolidity': 0.70,
        'AttackingThreat': 0.65
    },
    'Oliver Glasner': {
        'WinRate': 0.49,
        'GoalsPerGame': 1.6,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.73,
        'DefensiveSolidity': 0.68,
        'AttackingThreat': 0.72
    },
    'Vincent Kompany': {
        'WinRate': 0.53,
        'GoalsPerGame': 1.8,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.67,
        'DefensiveSolidity': 0.76,
        'AttackingThreat': 0.74
    },
    'David Moyes': {
        'WinRate': 0.44,
        'GoalsPerGame': 1.3,
        'PreferredFormation': '4-4-2',
        'TacticalFlexibility': 0.58,
        'DefensiveSolidity': 0.79,
        'AttackingThreat': 0.61
    },
    'Gary O\'Neil': {
        'WinRate': 0.38,
        'GoalsPerGame': 1.1,
        'PreferredFormation': '4-3-3',
        'TacticalFlexibility': 0.62,
        'DefensiveSolidity': 0.71,
        'AttackingThreat': 0.55
    },
    'Scott Parker': {
        'WinRate': 0.35,
        'GoalsPerGame': 1.0,
        'PreferredFormation': '4-3-3',
        'TacticalFlexibility': 0.60,
        'DefensiveSolidity': 0.68,
        'AttackingThreat': 0.52
    },
    'Frank Lampard': {
        'WinRate': 0.41,
        'GoalsPerGame': 1.2,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.65,
        'DefensiveSolidity': 0.69,
        'AttackingThreat': 0.58
    },
    'Steven Gerrard': {
        'WinRate': 0.43,
        'GoalsPerGame': 1.4,
        'PreferredFormation': '4-3-3',
        'TacticalFlexibility': 0.66,
        'DefensiveSolidity': 0.73,
        'AttackingThreat': 0.64
    },
    'Steve Cooper': {
        'WinRate': 0.42,
        'GoalsPerGame': 1.3,
        'PreferredFormation': '4-3-3',
        'TacticalFlexibility': 0.68,
        'DefensiveSolidity': 0.74,
        'AttackingThreat': 0.60
    },
    'Ange Postecoglou': {
        'WinRate': 0.55,
        'GoalsPerGame': 1.7,
        'PreferredFormation': '4-3-3',
        'TacticalFlexibility': 0.78,
        'DefensiveSolidity': 0.72,
        'AttackingThreat': 0.76
    },
    'Eddie Howe': {
        'WinRate': 0.48,
        'GoalsPerGame': 1.5,
        'PreferredFormation': '4-3-3',
        'TacticalFlexibility': 0.70,
        'DefensiveSolidity': 0.78,
        'AttackingThreat': 0.65
    },
    'Rob Edwards': {
        'WinRate': 0.40,
        'GoalsPerGame': 1.2,
        'PreferredFormation': '4-3-3',
        'TacticalFlexibility': 0.64,
        'DefensiveSolidity': 0.71,
        'AttackingThreat': 0.58
    },
    'Ole Gunnar Solskjaer': {
        'WinRate': 0.48,
        'GoalsPerGame': 1.6,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.65,
        'DefensiveSolidity': 0.68,
        'AttackingThreat': 0.72
    },
    'Mauricio Pochettino': {
        'WinRate': 0.52,
        'GoalsPerGame': 1.7,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.75,
        'DefensiveSolidity': 0.74,
        'AttackingThreat': 0.78
    },
    'Jose Mourinho': {
        'WinRate': 0.55,
        'GoalsPerGame': 1.5,
        'PreferredFormation': '4-3-3',
        'TacticalFlexibility': 0.70,
        'DefensiveSolidity': 0.82,
        'AttackingThreat': 0.68
    },
    'Graham Potter': {
        'WinRate': 0.47,
        'GoalsPerGame': 1.4,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.72,
        'DefensiveSolidity': 0.73,
        'AttackingThreat': 0.65
    },
    'Roy Hodgson': {
        'WinRate': 0.42,
        'GoalsPerGame': 1.2,
        'PreferredFormation': '4-4-2',
        'TacticalFlexibility': 0.60,
        'DefensiveSolidity': 0.78,
        'AttackingThreat': 0.58
    },
    'Patrick Vieira': {
        'WinRate': 0.45,
        'GoalsPerGame': 1.3,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.68,
        'DefensiveSolidity': 0.71,
        'AttackingThreat': 0.62
    },
    'Carlo Ancelotti': {
        'WinRate': 0.58,
        'GoalsPerGame': 1.8,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.78,
        'DefensiveSolidity': 0.80,
        'AttackingThreat': 0.82
    },
    'Rafael Benitez': {
        'WinRate': 0.50,
        'GoalsPerGame': 1.4,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.72,
        'DefensiveSolidity': 0.76,
        'AttackingThreat': 0.68
    },
    'Bruno Lage': {
        'WinRate': 0.44,
        'GoalsPerGame': 1.3,
        'PreferredFormation': '4-3-3',
        'TacticalFlexibility': 0.66,
        'DefensiveSolidity': 0.70,
        'AttackingThreat': 0.61
    },
    'Julen Lopetegui': {
        'WinRate': 0.46,
        'GoalsPerGame': 1.4,
        'PreferredFormation': '4-3-3',
        'TacticalFlexibility': 0.74,
        'DefensiveSolidity': 0.72,
        'AttackingThreat': 0.67
    },
    'Ralph Hasenhuttl': {
        'WinRate': 0.43,
        'GoalsPerGame': 1.3,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.65,
        'DefensiveSolidity': 0.75,
        'AttackingThreat': 0.60
    },
    'Chris Hughton': {
        'WinRate': 0.38,
        'GoalsPerGame': 1.1,
        'PreferredFormation': '4-4-2',
        'TacticalFlexibility': 0.58,
        'DefensiveSolidity': 0.76,
        'AttackingThreat': 0.55
    },
    'Brendan Rodgers': {
        'WinRate': 0.52,
        'GoalsPerGame': 1.6,
        'PreferredFormation': '4-3-3',
        'TacticalFlexibility': 0.76,
        'DefensiveSolidity': 0.74,
        'AttackingThreat': 0.78
    },
    'Marcelo Bielsa': {
        'WinRate': 0.49,
        'GoalsPerGame': 1.5,
        'PreferredFormation': '3-3-1-3',
        'TacticalFlexibility': 0.82,
        'DefensiveSolidity': 0.70,
        'AttackingThreat': 0.75
    },
    'Daniel Farke': {
        'WinRate': 0.44,
        'GoalsPerGame': 1.4,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.70,
        'DefensiveSolidity': 0.68,
        'AttackingThreat': 0.64
    },
    'Kieran McKenna': {
        'WinRate': 0.55,
        'GoalsPerGame': 1.7,
        'PreferredFormation': '4-2-3-1',
        'TacticalFlexibility': 0.73,
        'DefensiveSolidity': 0.78,
        'AttackingThreat': 0.72
    }
}

# Current team-manager mapping (as of January 2026)
TEAM_MANAGERS = {
    'Manchester City': 'Pep Guardiola',
    'Liverpool': 'Jurgen Klopp',
    'Arsenal': 'Mikel Arteta',
    'Manchester United': 'Erik ten Hag',
    'Chelsea': 'Enzo Maresca',
    'Tottenham': 'Ange Postecoglou',
    'Newcastle': 'Eddie Howe',
    'Aston Villa': 'Unai Emery',
    'Brighton': 'Roberto De Zerbi',
    'Fulham': 'Marco Silva',
    'Crystal Palace': 'Oliver Glasner',
    'Brentford': 'Thomas Frank',
    'Everton': 'Sean Dyche',
    'Wolverhampton Wanderers': 'Gary O\'Neil',
    'Bournemouth': 'Andoni Iraola',
    'Southampton': 'Russell Martin',
    'Nottingham Forest': 'Steve Cooper',
    'Luton': 'Rob Edwards',
    'Burnley': 'Vincent Kompany',
    'Sheffield United': 'Chris Wilder'
}

# Historical manager mappings by season and date ranges
# Format: {team: [(start_date, end_date, manager), ...]}
HISTORICAL_MANAGERS = {
    # Manchester City
    'Manchester City': [
        ('2021-08-01', '2026-12-31', 'Pep Guardiola')
    ],
    
    # Liverpool
    'Liverpool': [
        ('2021-08-01', '2026-12-31', 'Jurgen Klopp')
    ],
    
    # Arsenal
    'Arsenal': [
        ('2021-08-01', '2026-12-31', 'Mikel Arteta')
    ],
    
    # Manchester United
    'Manchester United': [
        ('2021-08-01', '2021-11-20', 'Ole Gunnar Solskjaer'),
        ('2021-11-21', '2022-10-20', 'Michael Carrick'),  # Interim
        ('2022-10-21', '2024-10-20', 'Erik ten Hag'),
        ('2024-10-21', '2026-12-31', 'Erik ten Hag')
    ],
    
    # Chelsea
    'Chelsea': [
        ('2021-08-01', '2021-09-06', 'Frank Lampard'),
        ('2021-09-07', '2022-01-07', 'Thomas Tuchel'),
        ('2022-01-08', '2023-03-20', 'Thomas Tuchel'),
        ('2023-03-21', '2023-05-28', 'Frank Lampard'),  # Interim
        ('2023-05-29', '2023-12-19', 'Mauricio Pochettino'),
        ('2023-12-20', '2024-05-21', 'Mauricio Pochettino'),
        ('2024-05-22', '2026-12-31', 'Enzo Maresca')
    ],
    
    # Tottenham
    'Tottenham': [
        ('2021-08-01', '2021-11-19', 'Jose Mourinho'),
        ('2021-11-20', '2022-04-20', 'Antonio Conte'),
        ('2022-04-21', '2023-03-24', 'Antonio Conte'),
        ('2023-03-25', '2023-05-28', 'Cristian Stellini'),  # Interim
        ('2023-05-29', '2024-10-20', 'Ange Postecoglou'),
        ('2024-10-21', '2026-12-31', 'Ange Postecoglou')
    ],
    
    # Newcastle
    'Newcastle': [
        ('2021-08-01', '2026-12-31', 'Eddie Howe')
    ],
    
    # Aston Villa
    'Aston Villa': [
        ('2021-08-01', '2022-10-07', 'Steven Gerrard'),
        ('2022-10-08', '2023-01-16', 'Steven Gerrard'),
        ('2023-01-17', '2026-12-31', 'Unai Emery')
    ],
    
    # Brighton
    'Brighton': [
        ('2021-08-01', '2022-05-30', 'Graham Potter'),
        ('2022-05-31', '2022-08-08', 'Graham Potter'),
        ('2022-08-09', '2024-05-20', 'Roberto De Zerbi'),
        ('2024-05-21', '2026-12-31', 'Roberto De Zerbi')
    ],
    
    # Fulham
    'Fulham': [
        ('2021-08-01', '2021-11-29', 'Marco Silva'),
        ('2021-11-30', '2026-12-31', 'Marco Silva')
    ],
    
    # Crystal Palace
    'Crystal Palace': [
        ('2021-08-01', '2021-07-18', 'Roy Hodgson'),
        ('2021-07-19', '2023-02-17', 'Patrick Vieira'),
        ('2023-02-18', '2023-05-28', 'Roy Hodgson'),  # Interim
        ('2023-05-29', '2026-12-31', 'Oliver Glasner')
    ],
    
    # Brentford
    'Brentford': [
        ('2021-08-01', '2026-12-31', 'Thomas Frank')
    ],
    
    # Everton
    'Everton': [
        ('2021-08-01', '2021-12-16', 'Carlo Ancelotti'),
        ('2021-12-17', '2022-01-30', 'Rafael Benitez'),  # Interim
        ('2022-01-31', '2022-10-23', 'Frank Lampard'),
        ('2022-10-24', '2026-12-31', 'Sean Dyche')
    ],
    
    # Wolves
    'Wolverhampton Wanderers': [
        ('2021-08-01', '2022-05-20', 'Bruno Lage'),
        ('2022-05-21', '2022-10-27', 'Bruno Lage'),
        ('2022-10-28', '2023-08-20', 'Julen Lopetegui'),
        ('2023-08-21', '2026-12-31', 'Gary O\'Neil')
    ],
    
    # Bournemouth
    'Bournemouth': [
        ('2021-08-01', '2022-06-30', 'Scott Parker'),
        ('2022-07-01', '2026-12-31', 'Andoni Iraola')
    ],
    
    # Southampton
    'Southampton': [
        ('2021-08-01', '2021-11-07', 'Ralph Hasenhuttl'),
        ('2021-11-08', '2022-02-12', 'Ralph Hasenhuttl'),
        ('2022-02-13', '2023-02-12', 'Nathan Jones'),  # Interim
        ('2023-02-13', '2026-12-31', 'Russell Martin')
    ],
    
    # Nottingham Forest
    'Nottingham Forest': [
        ('2021-08-01', '2022-06-28', 'Chris Hughton'),  # Interim
        ('2022-06-29', '2026-12-31', 'Steve Cooper')
    ],
    
    # Luton
    'Luton': [
        ('2023-08-01', '2026-12-31', 'Rob Edwards')
    ],
    
    # Burnley
    'Burnley': [
        ('2023-08-01', '2026-12-31', 'Vincent Kompany')
    ],
    
    # Sheffield United
    'Sheffield United': [
        ('2023-08-01', '2026-12-31', 'Chris Wilder')
    ],
    
    # West Ham
    'West Ham': [
        ('2021-08-01', '2022-12-21', 'David Moyes'),
        ('2022-12-22', '2026-12-31', 'David Moyes')
    ],
    
    # Leicester
    'Leicester': [
        ('2021-08-01', '2023-02-06', 'Brendan Rodgers'),
        ('2023-02-07', '2023-05-28', 'Dean Smith'),  # Interim
        ('2023-05-29', '2025-02-17', 'Enzo Maresca'),
        ('2025-02-18', '2026-12-31', 'Enzo Maresca')
    ],
    
    # Leeds
    'Leeds': [
        ('2021-08-01', '2022-02-27', 'Marcelo Bielsa'),
        ('2022-02-28', '2022-05-13', 'Jesse Marsch'),
        ('2022-05-14', '2023-02-06', 'Jesse Marsch'),
        ('2023-02-07', '2023-05-28', 'Sam Allardyce'),  # Interim
        ('2023-05-29', '2024-12-20', 'Daniel Farke'),
        ('2024-12-21', '2026-12-31', 'Daniel Farke')
    ],
    
    # Norwich (relegated after 2021-22)
    'Norwich': [
        ('2021-08-01', '2022-01-06', 'Daniel Farke'),
        ('2022-01-07', '2022-05-22', 'Dean Smith')
    ],
    
    # Watford (relegated after 2021-22)
    'Watford': [
        ('2021-08-01', '2022-01-19', 'Hayden Mullins'),  # Interim
        ('2022-01-20', '2022-05-22', 'Roy Hodgson')
    ],
    
    # Sunderland (relegated after 2021-22)
    'Sunderland': [
        ('2021-08-01', '2021-10-13', 'Lee Johnson'),
        ('2021-10-14', '2022-05-22', 'Alex Neil')
    ],
    
    # Ipswich (promoted for 2024-25)
    'Ipswich': [
        ('2024-08-01', '2026-12-31', 'Kieran McKenna')
    ]
}

def get_manager_stats(manager_name):
    """
    Get comprehensive statistics for a manager.

    Args:
        manager_name (str): Name of the manager

    Returns:
        dict: Manager statistics or default values if not found
    """
    return MANAGER_RECORDS.get(manager_name, {
        'WinRate': 0.45,  # League average
        'GoalsPerGame': 1.4,  # League average
        'PreferredFormation': '4-3-3',  # Most common
        'TacticalFlexibility': 0.65,  # Average flexibility
        'DefensiveSolidity': 0.70,  # Average defensive rating
        'AttackingThreat': 0.65  # Average attacking rating
    })

def get_current_manager(team_name, match_date=None):
    """
    Get the manager for a team at a specific date.
    If no date provided, returns current manager.
    
    Args:
        team_name (str): Name of the team
        match_date (str or Timestamp): Date in 'YYYY-MM-DD' format, or None for current
    
    Returns:
        str: Manager name or None if not found
    """
    if match_date is None:
        return TEAM_MANAGERS.get(team_name)
    
    # Convert Timestamp to string if needed
    if hasattr(match_date, 'strftime'):
        match_date_str = match_date.strftime('%Y-%m-%d')
    else:
        match_date_str = str(match_date)
    
    # Look up historical manager
    if team_name in HISTORICAL_MANAGERS:
        for start_date, end_date, manager in HISTORICAL_MANAGERS[team_name]:
            if start_date <= match_date_str <= end_date:
                return manager
    
    # Fallback to current manager if no historical data
    return TEAM_MANAGERS.get(team_name)

def calculate_manager_advantage(home_manager, away_manager):
    """
    Calculate the managerial advantage/disadvantage.

    Args:
        home_manager (str): Home team manager
        away_manager (str): Away team manager

    Returns:
        dict: Managerial advantage metrics
    """
    home_stats = get_manager_stats(home_manager)
    away_stats = get_manager_stats(away_manager)

    return {
        'ManagerWinRateDiff': home_stats['WinRate'] - away_stats['WinRate'],
        'ManagerGoalsPerGameDiff': home_stats['GoalsPerGame'] - away_stats['GoalsPerGame'],
        'ManagerDefensiveAdvantage': home_stats['DefensiveSolidity'] - away_stats['DefensiveSolidity'],
        'ManagerAttackingAdvantage': home_stats['AttackingThreat'] - away_stats['AttackingThreat'],
        'ManagerTacticalFlexibilityDiff': home_stats['TacticalFlexibility'] - away_stats['TacticalFlexibility']
    }