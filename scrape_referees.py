import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import re
import json
from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.leagues import get_league_config


def scrape_referees(league: LeagueConfig | str = 'epl'):
    """League-neutral referee entrypoint; unavailable sources return an empty frame."""
    config = get_league_config(league) if isinstance(league, str) else league
    if not config.sources.referee:
        return pd.DataFrame()
    # The migrated EPL scraper remains the adapter implementation until a
    # league-specific source is configured.
    if config.key == 'epl':
        return scrape_premier_league_referees()
    return pd.DataFrame()

def scrape_premier_league_referees():
    """
    Attempt to scrape referee assignments from Premier League sources
    Returns DataFrame with match referee information
    """

    referees_data = []

    # Method 1: Try Playmaker Stats (most promising source found)
    print("Attempting to scrape referee data from Playmaker Stats...")
    playmaker_data = _scrape_playmaker_referees()
    if playmaker_data:
        referees_data.extend(playmaker_data)
        print(f"✓ Found {len(playmaker_data)} referee assignments from Playmaker")
        # If we found data from Playmaker, we can return early
        df = pd.DataFrame(referees_data)
        df['ScrapedDate'] = datetime.now().strftime('%Y-%m-%d')
        print(f"Successfully scraped {len(df)} referee assignments from Playmaker")
        return df
    else:
        print("✗ No referee data found on Playmaker")

    # Method 2: Try Premier League official website
    print("Attempting to scrape referee data from Premier League website...")

    try:
        # Check fixtures page for any embedded referee data
        fixtures_url = 'https://www.premierleague.com/fixtures'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(fixtures_url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for JSON data in scripts that might contain referee info
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                # Look for JSON containing match data
                if 'fixtures' in script.string.lower() or 'matches' in script.string.lower():
                    try:
                        # Try to parse as JSON
                        data = json.loads(script.string)
                        # Look for referee data in the JSON structure
                        _extract_referees_from_json(data, referees_data)
                    except json.JSONDecodeError:
                        continue

        # Method 2: Try individual match pages
        print("Checking individual match pages for referee data...")
        match_links = soup.find_all('a', href=re.compile(r'/matches/|/fixtures/'))
        for link in match_links[:5]:  # Check first 5 matches
            match_url = 'https://www.premierleague.com' + link.get('href')
            try:
                match_response = requests.get(match_url, headers=headers, timeout=10)
                match_soup = BeautifulSoup(match_response.text, 'html.parser')

                # Look for referee information on match page
                referee_info = _extract_referee_from_match_page(match_soup)
                if referee_info:
                    referees_data.append(referee_info)

                time.sleep(1)  # Be respectful to the server

            except Exception as e:
                print(f"Error checking match {match_url}: {e}")
                continue

    except Exception as e:
        print(f"Error accessing Premier League fixtures: {e}")

    # Method 3: Try alternative sources (if primary method fails)
    if len(referees_data) == 0:
        print("No referee data found on Premier League website. Trying alternative sources...")

        # Try football-data.co.uk API or other sources
        referees_data = _try_alternative_sources()

    # Method 4: Try PGMOL website (Professional Game Match Officials Limited)
    if len(referees_data) == 0:
        print("Trying PGMOL website...")
        try:
            pgmol_url = 'https://www.pgmol.co.uk'
            response = requests.get(pgmol_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Look for match official assignments
            official_assignments = soup.find_all(text=re.compile(r'referee|official', re.IGNORECASE))
            if official_assignments:
                print(f"Found {len(official_assignments)} potential referee mentions on PGMOL site")
                # This would need more specific parsing based on PGMOL site structure

        except Exception as e:
            print(f"Error accessing PGMOL website: {e}")

    # Create DataFrame
    if referees_data:
        df = pd.DataFrame(referees_data)
        df['ScrapedDate'] = datetime.now().strftime('%Y-%m-%d')
        print(f"Successfully scraped {len(df)} referee assignments")
        return df
    else:
        print("No referee data could be scraped from available sources")
        return pd.DataFrame(columns=['Date', 'HomeTeam', 'AwayTeam', 'Referee', 'AssistantReferee1', 'AssistantReferee2', 'FourthOfficial', 'VAR', 'AVAR', 'ScrapedDate'])

def _extract_referees_from_json(data, referees_data):
    """
    Extract referee information from JSON data structure
    """
    try:
        # Navigate through potential JSON structures
        if isinstance(data, dict):
            # Look for matches/fixtures in various possible locations
            matches = data.get('matches', data.get('fixtures', data.get('events', [])))

            if isinstance(matches, list):
                for match in matches:
                    referee = _extract_single_match_referee(match)
                    if referee:
                        referees_data.append(referee)

            # Also check nested structures
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    _extract_referees_from_json(value, referees_data)

    except Exception as e:
        pass  # Silently continue if JSON structure is unexpected

def _extract_single_match_referee(match):
    """
    Extract referee info from a single match object
    """
    try:
        if isinstance(match, dict):
            # Look for referee data in various possible fields
            referee_info = {
                'Date': match.get('date', match.get('kickoff', '')),
                'HomeTeam': match.get('homeTeam', match.get('home', '')),
                'AwayTeam': match.get('awayTeam', match.get('away', '')),
                'Referee': match.get('referee', match.get('official', '')),
                'AssistantReferee1': match.get('assistantReferee1', ''),
                'AssistantReferee2': match.get('assistantReferee2', ''),
                'FourthOfficial': match.get('fourthOfficial', ''),
                'VAR': match.get('var', match.get('videoAssistantReferee', '')),
                'AVAR': match.get('avar', match.get('assistantVideoAssistantReferee', ''))
            }

            # Clean up team names
            if referee_info['HomeTeam'] and referee_info['AwayTeam'] and referee_info['Referee']:
                return referee_info

    except Exception as e:
        pass

    return None

def _extract_referee_from_match_page(soup):
    """
    Extract referee information from a match page HTML
    """
    try:
        # Look for referee information in various HTML elements
        referee_selectors = [
            '.referee',
            '.match-official',
            '.officials',
            '[data-referee]',
            '.referee-name'
        ]

        for selector in referee_selectors:
            elements = soup.select(selector)
            if elements:
                referee_text = elements[0].get_text().strip()
                if referee_text:
                    # Try to extract match details from the page
                    teams = _extract_teams_from_page(soup)
                    date = _extract_date_from_page(soup)

                    if teams and date:
                        return {
                            'Date': date,
                            'HomeTeam': teams[0],
                            'AwayTeam': teams[1],
                            'Referee': referee_text,
                            'AssistantReferee1': '',
                            'AssistantReferee2': '',
                            'FourthOfficial': '',
                            'VAR': '',
                            'AVAR': ''
                        }

        # Look for text patterns
        page_text = soup.get_text()
        referee_patterns = [
            r'Referee:\s*([^\n\r]+)',
            r'Match Official:\s*([^\n\r]+)',
            r'([A-Z][a-z]+ [A-Z][a-z]+).*?(?:referee|official)'
        ]

        for pattern in referee_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                teams = _extract_teams_from_page(soup)
                date = _extract_date_from_page(soup)

                if teams and date:
                    return {
                        'Date': date,
                        'HomeTeam': teams[0],
                        'AwayTeam': teams[1],
                        'Referee': matches[0].strip(),
                        'AssistantReferee1': '',
                        'AssistantReferee2': '',
                        'FourthOfficial': '',
                        'VAR': '',
                        'AVAR': ''
                    }

    except Exception as e:
        pass

    return None

def _extract_teams_from_page(soup):
    """
    Extract home and away team names from match page
    """
    try:
        # Look for team names in various selectors
        team_selectors = [
            '.team-name',
            '.club-name',
            '[data-team]',
            '.home-team',
            '.away-team'
        ]

        teams = []
        for selector in team_selectors:
            elements = soup.select(selector)
            for element in elements:
                team_name = element.get_text().strip()
                if team_name and team_name not in teams:
                    teams.append(team_name)
                    if len(teams) == 2:
                        return teams

        # Try extracting from title or headings
        title = soup.find('title')
        if title:
            title_text = title.get_text()
            # Look for pattern like "Team A vs Team B"
            vs_pattern = r'(.+?)\s+vs\s+(.+)'
            match = re.search(vs_pattern, title_text, re.IGNORECASE)
            if match:
                return [match.group(1).strip(), match.group(2).strip()]

    except Exception as e:
        pass

    return []

def _extract_date_from_page(soup):
    """
    Extract match date from page
    """
    try:
        # Look for date in various formats
        date_selectors = [
            '.match-date',
            '.kickoff-time',
            '[data-date]',
            '.date'
        ]

        for selector in date_selectors:
            elements = soup.select(selector)
            if elements:
                date_text = elements[0].get_text().strip()
                if date_text:
                    return date_text

        # Try to find date in meta tags or structured data
        meta_date = soup.find('meta', {'property': 'article:published_time'})
        if meta_date:
            return meta_date.get('content', '').split('T')[0]

    except Exception as e:
        pass

    return ''

def _scrape_playmaker_referees():
    """
    Scrape referee assignments from Playmaker Stats
    Returns list of referee assignment dictionaries
    """
    referees_data = []
    
    try:
        # First, find the latest referee announcement URL
        news_url = 'https://www.playmakerstats.com/news/'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        news_response = requests.get(news_url, headers=headers, timeout=15)
        news_soup = BeautifulSoup(news_response.text, 'html.parser')
        
        # Look for the latest Premier League referee announcement
        referee_news_url = None
        news_items = news_soup.find_all(['article', 'div'], class_=re.compile(r'news|article|item'))
        
        for item in news_items:
            title_elem = item.find(['h1', 'h2', 'h3', 'a'])
            if title_elem:
                title = title_elem.get_text().strip()
                link = item.find('a')
                if link and link.get('href'):
                    if 'referee' in title.lower() and 'premier' in title.lower():
                        referee_news_url = 'https://www.playmakerstats.com' + link['href']
                        break
        
        if not referee_news_url:
            # Fallback to known URL pattern (this would need to be updated regularly)
            referee_news_url = 'https://www.playmakerstats.com/news/premier-league-referees-announced-matchweek-22/1025696'
        
        print(f"Found referee announcement URL: {referee_news_url}")
        
        # Now scrape the referee assignments
        ref_response = requests.get(referee_news_url, headers=headers, timeout=15)
        ref_soup = BeautifulSoup(ref_response.text, 'html.parser')
        
        # Find the referee table
        tables = ref_soup.find_all('table')
        referee_table = None
        
        for table in tables:
            headers = table.find_all('th')
            header_texts = [h.get_text().strip() for h in headers]
            if 'Referees' in header_texts:
                referee_table = table
                break
        
        if referee_table:
            # Extract all rows
            rows = referee_table.find_all('tr')
            
            for row in rows[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 5:
                    date = cells[0].get_text().strip()
                    home_team = cells[1].get_text().strip()
                    time = cells[2].get_text().strip()
                    away_team = cells[3].get_text().strip()
                    referee = cells[4].get_text().strip()
                    
                    # Normalize team names to match our data
                    home_team = _normalize_team_name(home_team)
                    away_team = _normalize_team_name(away_team)
                    
                    referees_data.append({
                        'Date': date,
                        'HomeTeam': home_team,
                        'AwayTeam': away_team,
                        'Referee': referee,
                        'AssistantReferee1': '',
                        'AssistantReferee2': '',
                        'FourthOfficial': '',
                        'VAR': '',
                        'AVAR': ''
                    })
        
        return referees_data if referees_data else None
        
    except Exception as e:
        print(f"Error scraping Playmaker: {e}")
        return None

def _normalize_team_name(team_name):
    """
    Normalize team names to match our historical data format
    """
    team_mapping = {
        'Manchester United': 'Man United',
        'Manchester City': 'Man City',
        'Newcastle United': 'Newcastle',
        'Nottingham Forest': 'Nott\'m Forest',
        'Tottenham Hotspur': 'Tottenham',
        'West Ham United': 'West Ham',
        'Wolverhampton Wanderers': 'Wolves',
        'Wolverhampton': 'Wolves',  # Add this mapping
        'Brighton & Hove Albion': 'Brighton',
        'West Bromwich Albion': 'West Brom',
        'Leeds United': 'Leeds',
        'Aston Villa': 'Aston Villa',
        'Crystal Palace': 'Crystal Palace',
        'Southampton': 'Southampton',
        'Everton': 'Everton',
        'Burnley': 'Burnley',
        'Fulham': 'Fulham',
        'Brentford': 'Brentford',
        'Chelsea': 'Chelsea',
        'Liverpool': 'Liverpool',
        'Arsenal': 'Arsenal',
        'Sunderland': 'Sunderland',
        'Bournemouth': 'Bournemouth'
    }
    
    return team_mapping.get(team_name, team_name)

# Test the scraper
if __name__ == "__main__":
    print("Testing Premier League referee scraper...")

    # Test the main scraping function
    referees = scrape_premier_league_referees()

    if not referees.empty:
        print(f"\nScraped {len(referees)} referee assignments:")
        print(referees.to_string(index=False))

        # Save to CSV for inspection
        output_path = 'data_files/scraped_referees_test.csv'
        referees.to_csv(output_path, index=False)
        print(f"\nSaved referee data to {output_path}")
    else:
        print("No referee data was scraped. This could be due to:")
        print("1. Premier League website not publishing referee assignments in advance")
        print("2. Anti-scraping measures blocking access")
        print("3. Referee assignments announced closer to match time")
        print("4. Need for different data source or API access")
