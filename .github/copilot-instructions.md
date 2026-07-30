# Copilot Instructions — premier-league

## Purpose
Assist with analysis, predictions, and tooling for the Premier League dataset in this repo. Focus on ML modeling for match outcomes using historical football data.

## Architecture Overview
- **Data Pipeline**: Raw yearly CSVs downloaded from football-data.co.uk → `combine_raw_data.py` → `combined_historical_data.csv` → `prepare_model_data.py` → feature engineering → `combined_historical_data_with_calculations.csv`
- **Upcoming Fixtures**: `fetch_upcoming_fixtures.py` → queries ESPN API for next 30 days → `upcoming_fixtures.csv`
- **Referee Data**: `scrape_referees.py` → scrapes Playmaker Stats for referee assignments → `scraped_referees_test.csv`
- **App**: `premier-league-predictions.py` (Streamlit) loads processed data, trains XGBoost classifier for H/D/A predictions, displays model metrics, permutation feature importance, and referee statistics
- **Data Flow**: Match-level stats + team aggregates (goals, shots, differentials) + rolling form (last 5 points) + H2H history + rest days + referee statistics → ML features
- **Key Files**: 
  - `combine_raw_data.py`: Downloads and concatenates raw yearly CSVs from football-data.co.uk into combined historical data
  - `prepare_model_data.py`: Data wrangling, column renaming (e.g., `FTHG` → `FullTimeHomeGoals`), team stats calculation, rolling windows, feature engineering, referee statistics calculation
  - `fetch_upcoming_fixtures.py`: Fetches upcoming PL fixtures from ESPN API (site.api.espn.com) for next 30 days
  - `scrape_referees.py`: Scrapes referee assignments from Playmaker Stats website
  - `premier-league-predictions.py`: Model training, evaluation, referee statistics dashboard; uses tab-separated CSVs from `data_files/`

## Guidelines
- Prefer clear, minimal changes that follow existing functional style (no classes, plain scripts)
- Avoid changing filenames or moving data files unless requested
- When adding dependencies, update `requirements.txt` and include rationale
- Data processing: Handle missing values with `fillna(X.mean())`, encode categoricals to codes, clean column names for XGBoost (remove special chars, spaces to _)
- Modeling: Target encoding (H=0, D=1, A=2), drop leaky columns (results, goals), use permutation importance for feature selection
- Referee data: Normalize team names using `team_name_mapping.py`, calculate historical statistics from disciplinary data, merge referee assignments with upcoming fixtures

## Environment
- Python virtual environment (`venv/`); activate with `venv\Scripts\Activate.ps1`
- Activate the virtual environment before running scripts
- Dependencies: `xgboost`, `scikit-learn`, `streamlit`, `pandas`, `numpy`, `requests`, `beautifulsoup4`
- Data: Tab-separated CSVs in `data_files/`; downloaded from football-data.co.uk
- Additional data: Referee assignments (`scraped_referees_test.csv`), team name mappings (`team_name_mapping.py`)

## Running & Testing
- Update data: Run `python combine_raw_data.py` then `python prepare_model_data.py` to regenerate processed CSVs with latest data
- Process data: Run `python prepare_model_data.py` to generate processed CSVs
- Launch app: `streamlit run premier-league-predictions.py`
- Preference is to create .py scripts for tasks; avoid complex test frameworks
- Always compile using py_compile.py to check for syntax errors
- Model validation: Check MAE/accuracy on test set; inspect top features (e.g., betting odds, team form)
- No formal tests; validate by running app and checking outputs

## Developer Workflows
- Data updates: Run `python combine_raw_data.py` then `python prepare_model_data.py` to regenerate processed CSVs with latest data
- Fetch upcoming fixtures: Run `python fetch_upcoming_fixtures.py` to get next 30 days of PL matches from ESPN API
- Scrape referee assignments: Run `python scrape_referees.py` to get latest referee assignments from Playmaker Stats
- Feature engineering: Add calculations in `prepare_model_data.py` (e.g., rolling sums, H2H stats), merge into historical data
- Model tweaks: Modify XGBoost params or feature selection in Streamlit script

## Conventions & Patterns
- Column naming: Descriptive (e.g., `Bet365_HomeWinOdds`), consistent across scripts
- Team IDs: Lowercase, underscores, no apostrophes (e.g., "man_city")
- Date handling: `pd.to_datetime(..., dayfirst=True)` for UK format
- Rolling calculations: Use `groupby().rolling(window=5)` for form metrics
- Merges: Left join team stats onto match data for home/away features
- Referee data: Normalize team names using `team_name_mapping.py`, calculate historical statistics from disciplinary data, merge referee assignments with upcoming fixtures
- Feature engineering patterns:
  - Team aggregates: Calculate per-team averages/totals using `groupby().agg({'col': 'mean'})`
  - Rolling form: `groupby('Team')['Points'].rolling(window=5).sum()` for last 5 games
  - H2H history: Filter matches between same teams, sort by date, take last N
  - Rest days: Calculate gaps between matches using `shift(1)` and date differences
  - Betting features: Convert odds to implied probabilities, calculate market margins
- Data validation: Check for missing values, ensure date sorting before rolling operations
- API integration: Use requests with timeout/headers, handle JSON responses from ESPN API
- Scraping: Use BeautifulSoup for HTML parsing, handle rate limiting and errors gracefully

## Privacy & Secrets
- Do not commit secrets, API keys, or local config files. Add them to `.gitignore`.

If unsure
- Ask a clarifying question before making broad or destructive changes.