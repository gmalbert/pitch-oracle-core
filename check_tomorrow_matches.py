#!/usr/bin/env python3
"""
Check if there are Premier League matches tomorrow.
Returns exit code 0 if matches found, 1 if no matches.
Used by GitHub Actions to conditionally run data pipeline.
"""

import pandas as pd
from datetime import datetime, timedelta
from os import path
import sys

DATA_DIR = 'data_files/'

def check_tomorrow_matches():
    """Check if there are matches scheduled for tomorrow"""
    try:
        # Try to read existing upcoming fixtures
        fixtures_path = path.join(DATA_DIR, 'upcoming_fixtures.csv')
        if not path.exists(fixtures_path):
            print("No upcoming fixtures file found")
            return False

        df = pd.read_csv(fixtures_path)

        if len(df) == 0:
            print("No fixtures in upcoming fixtures file")
            return False

        # Get tomorrow's date
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"Checking for matches on {tomorrow}")

        # Filter for tomorrow's matches
        tomorrow_matches = df[df['Date'] == tomorrow]

        if len(tomorrow_matches) > 0:
            print(f"Found {len(tomorrow_matches)} matches tomorrow:")
            for _, match in tomorrow_matches.iterrows():
                print(f"  {match['HomeTeam']} vs {match['AwayTeam']} at {match['Time']}")
            return True
        else:
            print("No matches found for tomorrow")
            return False

    except Exception as e:
        print(f"Error checking for tomorrow's matches: {e}")
        return False

if __name__ == "__main__":
    has_matches = check_tomorrow_matches()
    sys.exit(0 if has_matches else 1)