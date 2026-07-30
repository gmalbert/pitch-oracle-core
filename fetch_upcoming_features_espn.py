import requests
from bs4 import BeautifulSoup
import pandas as pd
from os import path
from datetime import datetime, timedelta

DATA_DIR = 'data_files/'

# Get upcoming dates (next 7 days)
today = datetime.now()
upcoming_dates = [(today + timedelta(days=i)).strftime('%Y%m%d') for i in range(7)]

# Headers to mimic a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

all_fixtures = []

for date_str in upcoming_dates[:1]:  # Test with just first date
    url = f'https://www.espn.com/soccer/schedule/_/date/{date_str}'
    print(f"\nFetching fixtures for {date_str}...")
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Debug: Find all team links
    team_links = soup.find_all('a', href=lambda href: href and '/soccer/team/' in href)
    print(f"Found {len(team_links)} team links")
    for link in team_links[:10]:
        print(f"  Team: {link.text.strip()}")
    
    # Find all abbreviation elements (teams often use abbr tags)
    abbrs = soup.find_all('abbr')
    print(f"\nFound {len(abbrs)} abbreviation elements")
    for abbr in abbrs[:10]:
        print(f"  Abbr: {abbr.text.strip()} - Title: {abbr.get('title', 'N/A')}")
    
    # Process pairs of team links (home and away)
    i = 0
    while i < len(team_links) - 1:
        home_team = team_links[i].text.strip()
        away_team = team_links[i + 1].text.strip()
        
        # Filter for Premier League teams only
        # You can expand this list as needed
        pl_teams = [
            'Arsenal', 'Aston Villa', 'Brighton', 'Burnley', 'Chelsea', 
            'Crystal Palace', 'Everton', 'Fulham', 'Liverpool', 'Luton Town',
            'Manchester City', 'Manchester United', 'Newcastle United', 
            'Nottingham Forest', 'Sheffield United', 'Tottenham Hotspur', 
            'West Ham United', 'Wolverhampton Wanderers', 'Bournemouth', 'Brentford'
        ]
        
        if home_team in pl_teams and away_team in pl_teams:
            all_fixtures.append({
                'Date': date_str,
                'HomeTeam': home_team,
                'AwayTeam': away_team
            })
            i += 2  # Move to next pair
        else:
            i += 1  # Try next team

# Remove duplicates
seen = set()
unique_fixtures = []
for f in all_fixtures:
    key = (f['Date'], f['HomeTeam'], f['AwayTeam'])
    if key not in seen:
        seen.add(key)
        unique_fixtures.append(f)

df = pd.DataFrame(unique_fixtures)

# Save to CSV
df.to_csv(path.join(DATA_DIR, 'upcoming_fixtures.csv'), index=False)

print(f"\nFetched {len(df)} upcoming Premier League fixtures and saved to upcoming_fixtures.csv")
print(df)