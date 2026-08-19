"""League-neutral page renderers used by the shared Pitch Oracle shell.

Pages intentionally use only ASCII-safe Python/module names. Navigation icons are
declared centrally in :mod:`pitch_oracle_core.navigation` for macOS/Docker safety.
"""

from __future__ import annotations

from datetime import datetime
from os import path
import os

import pandas as pd
import streamlit as st

from .config import LeagueConfig
from .predictions import add_goal_market_predictions
from .risk import (
    HIGH_RISK_MAX,
    LOW_RISK_MAX,
    MODERATE_RISK_MAX,
    calculate_prediction_risk,
    get_prediction_guidance,
    get_risk_category,
)


def _data_dir() -> str:
    return __import__("os").environ.get("PITCH_ORACLE_DATA_DIR", "data_files")


def _read_csv(filename: str) -> pd.DataFrame:
    filename = path.join(_data_dir(), filename)
    if not path.exists(filename):
        return pd.DataFrame()
    try:
        with open(filename, "r", encoding="utf-8-sig", errors="replace") as stream:
            header = stream.readline()
        delimiter = "\t" if header.count("\t") > header.count(",") else ","
        return pd.read_csv(filename, sep=delimiter)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def _historical() -> pd.DataFrame:
    for filename in (
        "combined_historical_data_with_calculations_new.csv",
        "combined_historical_data_with_calculations.csv",
        "combined_historical_data.csv",
    ):
        frame = _read_csv(filename)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _upcoming() -> pd.DataFrame:
    return _read_csv("upcoming_fixtures.csv")


def _referee_assignments() -> pd.DataFrame:
    """Load published referee assignments for upcoming fixtures."""
    frame = _read_csv("referees.csv")
    if frame.empty:
        return frame
    date_column = "Date" if "Date" in frame.columns else frame.columns[0]
    for column in ("Referee", "RefereeID", "RefereeCareerGames",
                   "RefereeCareerYellow", "RefereeCareerRed"):
        if column not in frame.columns:
            frame[column] = pd.NA
    frame[date_column] = pd.to_datetime(
        frame[date_column], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    return frame


def _merge_referee_assignments(
    frame: pd.DataFrame, assignments: pd.DataFrame
) -> pd.DataFrame:
    """Attach referee assignments to a fixture/prediction frame on team+date."""
    if frame.empty:
        return frame
    result = frame.copy()
    if "Referee" in result.columns:
        return result
    referee_columns = ["Referee", "RefereeCareerGames", "RefereeCareerYellow",
                       "RefereeCareerRed"]
    if assignments.empty:
        result["Referee"] = "Not yet assigned"
        for column in referee_columns[1:]:
            result[column] = pd.NA
        return result
    date_column = "Date" if "Date" in result.columns else result.columns[0]
    result[date_column] = pd.to_datetime(
        result[date_column], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    keys = ["HomeTeam", "AwayTeam", date_column]
    result = result.merge(
        assignments[keys + referee_columns],
        on=keys, how="left",
    )
    result["Referee"] = result["Referee"].fillna("Not yet assigned")
    return result


def _height(frame: pd.DataFrame, maximum: int = 620) -> int:
    return min(max(150, 38 + len(frame) * 35), maximum)


def _display_fixtures(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a compact, human-readable fixture table."""
    if frame.empty:
        return frame
    result = frame.copy()
    result = result.rename(columns={
        "Date": "Match date", "Time": "Kickoff", "HomeTeam": "Home team",
        "AwayTeam": "Away team", "Status": "Status",
    })
    if "Status" in result:
        result["Status"] = result["Status"].astype(str).str.replace("STATUS_", "", regex=False).str.title()
    preferred = ["Match date", "Kickoff", "Home team", "Away team", "Status"]
    return result[[column for column in preferred if column in result.columns]]


def _display_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return predictions with readable labels and percentage formatting."""
    if frame.empty:
        return frame
    result = frame.copy().rename(columns={
        "Date": "Match date", "Time": "Kickoff", "HomeTeam": "Home team",
        "AwayTeam": "Away team", "HomeWin_Prob": "Home win",
        "Draw_Prob": "Draw", "AwayWin_Prob": "Away win",
        "PredictedResult": "Model pick", "Status": "Status",
        "PredictionGeneratedAt": "Generated",
        "Confidence_Score": "Confidence", "Risk_Score": "Risk score",
        "Risk_Category": "Risk level", "Recommendation": "Recommendation",
        "ModelLean": "Model lean", "ModelLeanProbability": "Lean probability",
        "BetRecommendation": "Bet recommendation", "BetReason": "Bet rationale",
        "ExpectedTotalGoals": "Expected total goals", "Over2_5Prob": "Over 2.5",
        "Under2_5Prob": "Under 2.5", "BTTSProb": "BTTS",
    })
    for column in ("Home win", "Draw", "Away win"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").map(
                lambda value: f"{value:.1%}" if pd.notna(value) else "—"
            )
    if "Confidence" in result:
        result["Confidence"] = pd.to_numeric(result["Confidence"], errors="coerce").map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "—"
        )
    if "Lean probability" in result:
        result["Lean probability"] = pd.to_numeric(
            result["Lean probability"], errors="coerce"
        ).map(lambda value: f"{value:.1%}" if pd.notna(value) else "—")
    if "Risk score" in result:
        result["Risk score"] = pd.to_numeric(result["Risk score"], errors="coerce").round(1)
    for column in ("Over 2.5", "Under 2.5", "BTTS"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").map(
                lambda value: f"{value:.1%}" if pd.notna(value) else "—"
            )
    if "Expected total goals" in result:
        result["Expected total goals"] = pd.to_numeric(
            result["Expected total goals"], errors="coerce"
        ).round(2)
    if {"Home team", "Away team", "Home win", "Draw", "Away win"}.issubset(result.columns):
        for index, row in result.iterrows():
            team_probabilities = {
                "Home team": pd.to_numeric(str(row["Home win"]).rstrip("%"), errors="coerce"),
                "Away team": pd.to_numeric(str(row["Away win"]).rstrip("%"), errors="coerce"),
            }
            team_probabilities = {
                team: probability
                for team, probability in team_probabilities.items()
                if pd.notna(probability)
            }
            if team_probabilities:
                picked_team = max(team_probabilities, key=team_probabilities.get)
                result.at[index, picked_team] = f"▲ {result.at[index, picked_team]}"
    if "Status" in result:
        result["Status"] = result["Status"].astype(str).str.replace("STATUS_", "", regex=False).str.title()
    preferred = [
        "Match date", "Kickoff", "Home team", "Away team", "Home win", "Draw",
        "Away win", "Model pick", "Confidence", "Risk score", "Risk level",
        "Model lean", "Lean probability", "Recommendation", "Bet recommendation",
        "Bet rationale", "Expected total goals", "Over 2.5", "Under 2.5", "BTTS", "Status",
    ]
    return result[[column for column in preferred if column in result.columns]]


def _prediction_assessment(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calibrated ambiguity scores and cautious model guidance."""
    result = frame.copy()
    probability_columns = ["HomeWin_Prob", "Draw_Prob", "AwayWin_Prob"]
    if not set(probability_columns).issubset(result.columns):
        return result

    probabilities = result[probability_columns].apply(pd.to_numeric, errors="coerce")
    assessments = []
    for values in probabilities.itertuples(index=False, name=None):
        try:
            risk, confidence = calculate_prediction_risk(values)
            category, _ = get_risk_category(risk)
            recommendation, _ = get_prediction_guidance(values, risk)
        except ValueError:
            risk, confidence, category, recommendation = float("nan"), float("nan"), "Unavailable", "Unavailable"
        assessments.append((confidence, risk, category, recommendation))

    result[["Confidence_Score", "Risk_Score", "Risk_Category", "Recommendation"]] = assessments
    return result


def _prediction_column_config() -> dict:
    return {
        "Home team": st.column_config.TextColumn("Home team", width="medium"),
        "Away team": st.column_config.TextColumn("Away team", width="medium"),
        "Home win": st.column_config.TextColumn("Home win", width="small"),
        "Draw": st.column_config.TextColumn("Draw", width="small"),
        "Away win": st.column_config.TextColumn("Away win", width="small"),
        "Confidence": st.column_config.TextColumn("Confidence", width="small"),
        "Risk score": st.column_config.NumberColumn("Risk score", format="%.1f", width="small"),
        "Risk level": st.column_config.TextColumn("Risk level", width="medium"),
        "Recommendation": st.column_config.TextColumn("Recommendation", width="medium"),
    }


def _filter_predictions_by_risk(frame: pd.DataFrame, selection: str) -> pd.DataFrame:
    """Return the selected risk band without changing the cached prediction data."""
    if selection == "All matches" or "Risk_Score" not in frame:
        return frame.copy()

    risk = pd.to_numeric(frame["Risk_Score"], errors="coerce")
    if selection == "Low risk":
        return frame.loc[risk <= LOW_RISK_MAX].copy()
    if selection == "Moderate risk":
        return frame.loc[(risk > LOW_RISK_MAX) & (risk <= MODERATE_RISK_MAX)].copy()
    if selection == "High risk":
        return frame.loc[(risk > MODERATE_RISK_MAX) & (risk <= HIGH_RISK_MAX)].copy()
    if selection == "Critical risk":
        return frame.loc[risk > HIGH_RISK_MAX].copy()
    return frame.copy()


def _style_prediction_risk(row: pd.Series) -> list[str]:
    """Shade every prediction row according to its ambiguity/risk score."""
    risk = pd.to_numeric(pd.Series([row.get("Risk score")]), errors="coerce").iloc[0]
    if pd.isna(risk):
        return [""] * len(row)
    if risk <= LOW_RISK_MAX:
        style = "background-color: #d4edda; color: #155724"
    elif risk <= MODERATE_RISK_MAX:
        style = "background-color: #fff3cd; color: #856404"
    elif risk <= HIGH_RISK_MAX:
        style = "background-color: #ffe5b4; color: #7a4100"
    else:
        style = "background-color: #f8d7da; color: #721c24"
    return [style] * len(row)


def _style_prediction_cells(row: pd.Series) -> list[str]:
    """Shade risk and bold the highest-probability team."""
    styles = _style_prediction_risk(row)
    probabilities = {
        "Home team": row.get("Home win"),
        "Away team": row.get("Away win"),
    }
    numeric_probabilities = {
        column: pd.to_numeric(str(value).rstrip("%"), errors="coerce")
        for column, value in probabilities.items()
    }
    numeric_probabilities = {
        column: value for column, value in numeric_probabilities.items() if pd.notna(value)
    }
    picked_team = max(numeric_probabilities, key=numeric_probabilities.get) if numeric_probabilities else None
    for index, column in enumerate(row.index):
        if column == picked_team:
            # !important keeps the recommendation visible alongside the row's
            # risk coloring in Streamlit's pandas Styler rendering.
            styles[index] += "; font-weight: bold !important"
    return styles


def _prediction_weather(row: pd.Series) -> str:
    """Return a compact expected-weather summary when forecast data is present."""
    def first_value(*keys):
        for key in keys:
            value = row.get(key)
            if value is not None and not pd.isna(value) and str(value).strip() not in {"", "Unknown", "nan"}:
                return value
        return None

    condition = first_value("WeatherDescription", "WeatherCondition", "Weather", "Forecast")
    temperature = first_value("Temperature", "TemperatureC", "ForecastTemperature")
    precipitation = first_value("Precipitation", "PrecipitationMm", "ForecastPrecipitation")
    wind = first_value("WindSpeed", "WindSpeedMs", "ForecastWindSpeed")
    humidity = first_value("Humidity", "HumidityPct", "ForecastHumidity")

    details = []
    if condition is not None:
        details.append(str(condition))
    if temperature is not None:
        details.append(f"{float(temperature):.1f}°C")
    if precipitation is not None:
        details.append(f"{float(precipitation):.1f} mm rain")
    if wind is not None:
        details.append(f"{float(wind):.1f} m/s wind")
    if humidity is not None:
        details.append(f"{float(humidity):.0f}% humidity")
    if details:
        return " · ".join(details)
    match_date = pd.to_datetime(row.get("Date", row.get("MatchDate")), errors="coerce")
    if pd.notna(match_date) and match_date.normalize() > pd.Timestamp.now().normalize() + pd.Timedelta(days=16):
        return "Available closer to kickoff"
    return "Forecast temporarily unavailable"


def _prediction_commentary(row: pd.Series) -> str:
    """Generate a scannable, probability-grounded explanation for one fixture."""
    probabilities = {
        "Home win": float(row["HomeWin_Prob"]),
        "Draw": float(row["Draw_Prob"]),
        "Away win": float(row["AwayWin_Prob"]),
    }
    outcome, confidence = max(probabilities.items(), key=lambda item: item[1])
    home, away = row.get("HomeTeam", "Home"), row.get("AwayTeam", "Away")
    if outcome == "Home win":
        pick = f"{home} to win"
    elif outcome == "Away win":
        pick = f"{away} to win"
    else:
        pick = "a draw"
    risk_level = row.get("Risk_Category", "Unavailable")
    lines = [
        f"- **Model lean:** {pick} ({confidence:.1%})",
        f"- **Risk level:** {risk_level}",
    ]
    referee = row.get("Referee")
    if pd.notna(referee) and str(referee).strip():
        career_parts = []
        games = row.get("RefereeCareerGames")
        if pd.notna(games) and float(games) > 0:
            career_parts.append(f"{float(games):.0f} career matches")
        yellow = row.get("RefereeCareerYellow")
        red = row.get("RefereeCareerRed")
        if pd.notna(yellow) and float(yellow) > 0:
            career_parts.append(f"{float(yellow):.0f} yellow cards")
        if pd.notna(red) and float(red) > 0:
            career_parts.append(f"{float(red):.0f} red cards")
        referee_line = f"- **Referee:** {referee}"
        if career_parts:
            referee_line += " (" + ", ".join(career_parts) + ")"
        lines.append(referee_line)
    bet_recommendation = row.get("BetRecommendation")
    bet_reason = row.get("BetReason")
    if pd.notna(bet_recommendation):
        bet_line = f"- **Bet recommendation:** {bet_recommendation}"
        if pd.notna(bet_reason):
            bet_line += f" — {bet_reason}"
        lines.append(bet_line)
    market_labels = (
        ("Expected total goals", "ExpectedTotalGoals", "{:.2f}"),
        ("Over 2.5", "Over2_5Prob", "{:.1%}"),
        ("Under 2.5", "Under2_5Prob", "{:.1%}"),
        ("BTTS", "BTTSProb", "{:.1%}"),
    )
    market_values = []
    for label, key, format_string in market_labels:
        value = row.get(key)
        if pd.notna(value):
            market_values.append(f"{label}: {format_string.format(float(value))}")
    goal_outlook = " · ".join(market_values) if market_values else "Unavailable"
    lines.append(f"- **Goal outlook:** {goal_outlook}")
    weather = _prediction_weather(row)
    lines.append(f"- **Expected weather:** {weather}")
    return "\n".join(lines)


def _page_title(config: LeagueConfig, title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(f"{config.display_name} · {subtitle}")


def render_overview(config: LeagueConfig) -> None:
    from .navigation import render_sidebar_branding
    render_sidebar_branding(config, show_logo=False)
    logo = path.join(_data_dir(), "logo.png")
    title = "English Premier League - Overview" if config.key == "epl" else f"{config.display_name} - Overview"
    if path.exists(logo):
        # Leave a little breathing room above the centered header.  This keeps
        # the ball clear of the top edge while preserving the original scale.
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        logo_column, title_column = st.columns([2, 5], vertical_alignment="center")
        with logo_column:
            st.image(logo, width=200)
        with title_column:
            st.title(title)
    else:
        st.title(title)
    upcoming = _upcoming()
    historical = _historical()

    c1, c2, c3 = st.columns(3)
    c1.metric("Upcoming fixtures", len(upcoming))
    c2.metric("Historical matches", len(historical))
    c3.metric("Data refreshed", datetime.now().strftime("%b %d, %Y"))

    st.subheader("Upcoming fixtures")
    if upcoming.empty:
        st.info("No upcoming fixtures are available yet.")
    else:
        display = _display_fixtures(upcoming)
        st.dataframe(display, hide_index=True, width="stretch", column_config={
            "Match date": st.column_config.DateColumn("Match date", format="MMM D, YYYY"),
            "Kickoff": st.column_config.TextColumn("Kickoff", width="small"),
        })


def render_predictions(config: LeagueConfig) -> None:
    from .navigation import render_sidebar_branding
    render_sidebar_branding(config)
    _page_title(config, "Predictions", "Upcoming matches, probabilities, and risk")
    predictions = _read_csv("upcoming_predictions.csv")
    if predictions.empty:
        predictions = _upcoming()
    if predictions.empty:
        st.info("No upcoming prediction cache is available yet.")
        return
    add_goal_market_predictions(predictions)
    predictions = _merge_referee_assignments(
        predictions, _referee_assignments()
    )

    teams = sorted(
        set(predictions.get("HomeTeam", pd.Series(dtype=str)).dropna())
        | set(predictions.get("AwayTeam", pd.Series(dtype=str)).dropna())
    )
    selected_team = st.selectbox("Team", ["All teams", *teams])
    if selected_team != "All teams":
        predictions = predictions[
            (predictions.get("HomeTeam") == selected_team)
            | (predictions.get("AwayTeam") == selected_team)
        ]
    predictions = _prediction_assessment(predictions)
    risk_filter = st.radio(
        "Risk level",
        ["All matches", "Low risk", "Moderate risk", "High risk", "Critical risk"],
        horizontal=True,
        help="Risk measures how close the three predicted outcomes are; it is not a measure of betting value.",
    )
    predictions = _filter_predictions_by_risk(predictions, risk_filter)
    display = _display_predictions(predictions)
    st.caption("▲ = highest-probability team. Green = more decisive; yellow/orange/red = progressively more ambiguous.")

    metrics = st.columns(4)
    risk_scores = pd.to_numeric(predictions.get("Risk_Score"), errors="coerce")
    confidence_scores = pd.to_numeric(predictions.get("Confidence_Score"), errors="coerce")
    recommendations = predictions.get("Recommendation", pd.Series(dtype=str)).fillna("")
    metrics[0].metric("Matches shown", len(predictions))
    metrics[1].metric("Low risk", int((risk_scores <= LOW_RISK_MAX).sum()))
    metrics[2].metric("High confidence", int((confidence_scores >= 0.60).sum()))
    metrics[3].metric("Actionable leans", int(recommendations.str.contains("Strong|Consider", regex=True).sum()))

    with st.expander("Risk scoring methodology"):
        st.markdown(
            f"""- 🟢 **Low risk (0–{LOW_RISK_MAX:.0f})**: more decisive model prediction.
            - 🟡 **Moderate risk ({LOW_RISK_MAX:.0f}–{MODERATE_RISK_MAX:.0f})**: usable lean with meaningful uncertainty.
            - 🟠 **High risk ({MODERATE_RISK_MAX:.0f}–{HIGH_RISK_MAX:.0f})**: ambiguous outcome probabilities.
            - 🔴 **Critical risk (>{HIGH_RISK_MAX:.0f})**: close to a three-way toss-up.

            Risk combines the leading probability with its margin over the runner-up. It describes model ambiguity, not betting value."""
        )

    if display.empty:
        st.info("No matches match the selected filters.")
        return

    styled_display = display.style.apply(_style_prediction_cells, axis=1).format({
        "Expected total goals": "{:.2f}",
    })
    st.dataframe(
        styled_display,
        hide_index=True,
        height=_height(display),
        width="stretch",
        column_config=_prediction_column_config(),
    )
    st.download_button(
        "Download predictions as CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name=f"{config.key}_predictions_{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
    )

    st.subheader("Match breakdown")
    st.caption("Open a fixture for its probabilities, model assessment, goal outlook, and forecast.")
    required_probabilities = {"HomeWin_Prob", "Draw_Prob", "AwayWin_Prob"}
    if required_probabilities.issubset(predictions.columns):
        for _, row in predictions.iterrows():
            home, away = row.get("HomeTeam", "Home"), row.get("AwayTeam", "Away")
            date, kickoff = row.get("Date", ""), row.get("Time", "")
            with st.expander(
                f"{home} vs {away} — {date} {kickoff}",
                icon=":material/sports_soccer:",
            ):
                probabilities = st.columns(3)
                probabilities[0].metric("Home win", f"{float(row['HomeWin_Prob']):.1%}")
                probabilities[1].metric("Draw", f"{float(row['Draw_Prob']):.1%}")
                probabilities[2].metric("Away win", f"{float(row['AwayWin_Prob']):.1%}")
                referee = row.get("Referee")
                if pd.notna(referee) and str(referee).strip():
                    st.caption(f"Referee: {referee}")
                with st.container(border=True):
                    st.markdown(_prediction_commentary(row))


def _standings(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"HomeTeam", "AwayTeam", "FullTimeHomeGoals", "FullTimeAwayGoals"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    rows: dict[str, dict[str, int]] = {}
    for _, match in frame.iterrows():
        home, away = match["HomeTeam"], match["AwayTeam"]
        hg, ag = int(match["FullTimeHomeGoals"]), int(match["FullTimeAwayGoals"])
        for team in (home, away):
            rows.setdefault(team, {"Played": 0, "Wins": 0, "Draws": 0, "Losses": 0, "GF": 0, "GA": 0, "Points": 0})
        rows[home]["Played"] += 1; rows[away]["Played"] += 1
        rows[home]["GF"] += hg; rows[home]["GA"] += ag
        rows[away]["GF"] += ag; rows[away]["GA"] += hg
        if hg > ag:
            rows[home]["Wins"] += 1; rows[away]["Losses"] += 1; rows[home]["Points"] += 3
        elif ag > hg:
            rows[away]["Wins"] += 1; rows[home]["Losses"] += 1; rows[away]["Points"] += 3
        else:
            rows[home]["Draws"] += 1; rows[away]["Draws"] += 1
            rows[home]["Points"] += 1; rows[away]["Points"] += 1
    result = pd.DataFrame.from_dict(rows, orient="index").rename_axis("Team").reset_index()
    result["GD"] = result["GF"] - result["GA"]
    return result.sort_values(["Points", "GD", "GF"], ascending=False).reset_index(drop=True)


def render_standings(config: LeagueConfig) -> None:
    from .navigation import render_sidebar_branding
    render_sidebar_branding(config)
    _page_title(config, "Standings", "Current league table")
    table = _standings(_historical())
    if table.empty:
        st.info("Standings will appear when match-result data is available.")
    else:
        table.insert(0, "Pos", range(1, len(table) + 1))
        st.dataframe(table, hide_index=True, height=_height(table, 760), width="stretch")


_TEAM_DEEP_DIVE_VIEWS = {
    "Results": (
        "Match date", "Opponent", "Venue", "Result", "Score", "Shots", "Shots on target",
    ),
    "Performance": (
        "Match date", "Opponent", "Venue", "Result", "Rolling xG (L5)",
        "Venue form points (L5)", "Rolling goal difference (L5)", "Rest days",
    ),
    "Discipline": (
        "Match date", "Opponent", "Venue", "Result", "Corners", "Fouls",
        "Yellow cards", "Red cards",
    ),
}

_TEAM_DEEP_DIVE_DICTIONARY = (
    ("Match date", "Calendar date of the completed fixture.", "MatchDate", "All"),
    ("Opponent", "The selected team's opponent.", "HomeTeam / AwayTeam", "All"),
    ("Venue", "Whether the selected team played at home or away.", "Team role", "All"),
    ("Result", "Win, draw, or loss from the selected team's perspective.", "FullTimeResult", "All"),
    ("Score", "Full-time score with the selected team's goals shown first.", "FullTimeHomeGoals / FullTimeAwayGoals", "Results"),
    ("Shots", "Total shots attempted by the selected team.", "HomeShots / AwayShots", "Results"),
    ("Shots on target", "Selected-team shots that were on target.", "HomeShotsOnTarget / AwayShotsOnTarget", "Results"),
    ("Rolling xG (L5)", "Pre-match average expected-goals proxy over the previous five matches in the same venue role.", "HomexG_Avg_L5 / AwayxG_Avg_L5", "Performance"),
    ("Venue form points (L5)", "Points earned over the previous five home or away matches, out of 15.", "HomeTeamPointsLast5 / AwayTeamPointsLast5", "Performance"),
    ("Rolling goal difference (L5)", "Average goals scored minus goals conceded over the previous five matches in the same venue role.", "HomeGoalDiff_Avg_L5 / AwayGoalDiff_Avg_L5", "Performance"),
    ("Rest days", "Days since the selected team's previous fixture.", "HomeRestDays / AwayRestDays", "Performance"),
    ("Corners", "Corners won by the selected team.", "HomeCorners / AwayCorners", "Discipline"),
    ("Fouls", "Fouls committed by the selected team.", "HomeFouls / AwayFouls", "Discipline"),
    ("Yellow cards", "Yellow cards shown to the selected team.", "HomeYellowCards / AwayYellowCards", "Discipline"),
    ("Red cards", "Red cards shown to the selected team.", "HomeRedCards / AwayRedCards", "Discipline"),
)


def _team_deep_dive_dictionary() -> pd.DataFrame:
    """Return the user-facing definitions for the curated team match log."""
    return pd.DataFrame(
        _TEAM_DEEP_DIVE_DICTIONARY,
        columns=("Field", "Definition", "Source field", "View"),
    )


def _team_match_log(frame: pd.DataFrame, team: str, limit: int | None = 10) -> pd.DataFrame:
    """Convert match-level source data into a compact team-perspective log."""
    matches = frame[(frame.get("HomeTeam") == team) | (frame.get("AwayTeam") == team)].copy()
    if matches.empty:
        return pd.DataFrame(columns=tuple(dict.fromkeys(sum(_TEAM_DEEP_DIVE_VIEWS.values(), ()))))
    matches["_match_date"] = pd.to_datetime(matches.get("MatchDate"), errors="coerce")
    matches = matches.sort_values("_match_date", ascending=False, kind="stable")
    if limit is not None:
        matches = matches.head(limit)

    def number(match: pd.Series, key: str):
        return pd.to_numeric(pd.Series([match.get(key)]), errors="coerce").iloc[0]

    rows = []
    for _, match in matches.iterrows():
        is_home = match.get("HomeTeam") == team
        prefix = "Home" if is_home else "Away"
        other = "Away" if is_home else "Home"
        raw_result = match.get("FullTimeResult")
        result = (
            "D" if raw_result == "D"
            else "W" if (is_home and raw_result == "H") or (not is_home and raw_result == "A")
            else "L" if raw_result in {"H", "A"}
            else "—"
        )
        goals_for = number(match, f"FullTime{prefix}Goals")
        goals_against = number(match, f"FullTime{other}Goals")
        score = (
            f"{int(goals_for)}–{int(goals_against)}"
            if pd.notna(goals_for) and pd.notna(goals_against)
            else "—"
        )
        rows.append({
            "Match date": match["_match_date"],
            "Opponent": match.get(f"{other}Team", "—"),
            "Venue": "Home" if is_home else "Away",
            "Result": result,
            "Score": score,
            "Shots": number(match, f"{prefix}Shots"),
            "Shots on target": number(match, f"{prefix}ShotsOnTarget"),
            "Rolling xG (L5)": number(match, f"{prefix}xG_Avg_L5"),
            "Venue form points (L5)": number(match, f"{prefix}TeamPointsLast5"),
            "Rolling goal difference (L5)": number(match, f"{prefix}GoalDiff_Avg_L5"),
            "Rest days": number(match, f"{prefix}RestDays"),
            "Corners": number(match, f"{prefix}Corners"),
            "Fouls": number(match, f"{prefix}Fouls"),
            "Yellow cards": number(match, f"{prefix}YellowCards"),
            "Red cards": number(match, f"{prefix}RedCards"),
        })
    return pd.DataFrame(rows)


def _team_match_log_column_config() -> dict:
    """Format the compact match-log fields consistently across views."""
    integer_columns = (
        "Shots", "Shots on target", "Venue form points (L5)", "Rest days",
        "Corners", "Fouls", "Yellow cards", "Red cards",
    )
    config = {
        "Match date": st.column_config.DateColumn("Match date", format="MMM D, YYYY", width="small"),
        "Opponent": st.column_config.TextColumn("Opponent", width="medium", pinned=True),
        "Venue": st.column_config.TextColumn("Venue", width="small"),
        "Result": st.column_config.TextColumn("Result", width="small"),
        "Score": st.column_config.TextColumn("Score", width="small"),
        "Rolling xG (L5)": st.column_config.NumberColumn("Rolling xG (L5)", format="%.2f", width="small"),
        "Rolling goal difference (L5)": st.column_config.NumberColumn(
            "Rolling goal difference (L5)", format="%+.2f", width="medium"
        ),
    }
    config.update({
        column: st.column_config.NumberColumn(column, format="%d", width="small")
        for column in integer_columns
    })
    return config


def render_team_deep_dive(config: LeagueConfig) -> None:
    from .navigation import render_sidebar_branding
    render_sidebar_branding(config)
    _page_title(config, "Team Deep Dive", "Form, scoring, and historical performance")
    frame = _historical()
    teams = sorted(set(frame.get("HomeTeam", pd.Series(dtype=str))) | set(frame.get("AwayTeam", pd.Series(dtype=str))))
    if not teams:
        st.info("Team analysis will appear when historical data is available.")
        return
    team = st.selectbox("Select a team", teams)
    home = frame[frame["HomeTeam"] == team]
    away = frame[frame["AwayTeam"] == team]
    played = len(home) + len(away)
    gf = home.get("FullTimeHomeGoals", pd.Series(dtype=float)).sum() + away.get("FullTimeAwayGoals", pd.Series(dtype=float)).sum()
    ga = home.get("FullTimeAwayGoals", pd.Series(dtype=float)).sum() + away.get("FullTimeHomeGoals", pd.Series(dtype=float)).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", played); c2.metric("Goals for", int(gf)); c3.metric("Goals against", int(ga))
    st.subheader("Recent matches")
    st.caption("The default views exclude bookmaker fields, model internals, duplicate indicators, and training targets.")
    view = st.segmented_control(
        "Table view",
        options=tuple(_TEAM_DEEP_DIVE_VIEWS),
        default="Results",
        required=True,
        key="team_deep_dive_table_view",
    )
    recent = _team_match_log(frame, team)
    displayed = recent.loc[:, _TEAM_DEEP_DIVE_VIEWS[view]]
    st.dataframe(
        displayed,
        hide_index=True,
        height=_height(displayed),
        width="stretch",
        column_config=_team_match_log_column_config(),
        key="team_deep_dive_match_log",
    )
    with st.expander("Data dictionary", icon=":material/menu_book:"):
        st.caption("L5 means the five prior matches; rolling fields exclude the match shown to avoid look-ahead leakage.")
        dictionary = _team_deep_dive_dictionary()
        st.dataframe(
            dictionary,
            hide_index=True,
            width="stretch",
            height=_height(dictionary, 560),
            column_config={
                "Field": st.column_config.TextColumn("Field", width="medium", pinned=True),
                "Definition": st.column_config.TextColumn("Definition", width="large"),
                "Source field": st.column_config.TextColumn("Source field", width="large"),
                "View": st.column_config.TextColumn("View", width="small"),
            },
        )


def render_statistics(config: LeagueConfig) -> None:
    from .navigation import render_sidebar_branding
    render_sidebar_branding(config)
    _page_title(config, "Statistics", "League, team, and referee analysis")
    frame = _historical()
    if frame.empty:
        st.info("Statistics will appear when historical data is available.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", len(frame))
    if "FullTimeHomeGoals" in frame:
        c2.metric("Avg goals", f"{pd.to_numeric(frame['FullTimeHomeGoals'], errors='coerce').add(pd.to_numeric(frame['FullTimeAwayGoals'], errors='coerce')).mean():.2f}")
    if "FullTimeResult" in frame:
        c3.metric("Home wins", f"{(frame['FullTimeResult'] == 'H').mean():.1%}")
    st.subheader("League summary")
    total = len(frame)
    results = frame["FullTimeResult"].value_counts() if "FullTimeResult" in frame else pd.Series(dtype=int)
    home_goals = pd.to_numeric(frame.get("FullTimeHomeGoals"), errors="coerce")
    away_goals = pd.to_numeric(frame.get("FullTimeAwayGoals"), errors="coerce")
    summary = pd.DataFrame([
        {"Measure": "Home wins", "Value": f"{int(results.get('H', 0)):,}", "Share": f"{results.get('H', 0) / total:.1%}"},
        {"Measure": "Draws", "Value": f"{int(results.get('D', 0)):,}", "Share": f"{results.get('D', 0) / total:.1%}"},
        {"Measure": "Away wins", "Value": f"{int(results.get('A', 0)):,}", "Share": f"{results.get('A', 0) / total:.1%}"},
        {"Measure": "Average home goals", "Value": f"{home_goals.mean():.2f}", "Share": "—"},
        {"Measure": "Average away goals", "Value": f"{away_goals.mean():.2f}", "Share": "—"},
    ])
    st.dataframe(summary, hide_index=True, width="stretch")

    st.subheader("Most productive teams")
    home_scoring = frame.groupby("HomeTeam")["FullTimeHomeGoals"].sum()
    away_scoring = frame.groupby("AwayTeam")["FullTimeAwayGoals"].sum()
    team_scoring = home_scoring.add(away_scoring, fill_value=0).sort_values(ascending=False).head(10)
    scoring = team_scoring.rename_axis("Team").reset_index(name="Goals scored")
    st.dataframe(scoring, hide_index=True, width="stretch")

    st.subheader("Referee analysis")
    assignments = _referee_assignments()
    upcoming = _upcoming()
    if assignments.empty or upcoming.empty:
        st.caption(
            "Referee assignments will appear here once they are published "
            "(typically 1–3 days before kickoff)."
        )
        return
    matched = _merge_referee_assignments(upcoming, assignments)
    assigned = matched.loc[
        (matched["Referee"] != "Not yet assigned")
        & matched["Referee"].notna()
    ].copy()
    if assigned.empty:
        st.caption(
            "Referee assignments for the upcoming matchday are not published yet; "
            "check back closer to kickoff."
        )
        return
    st.caption(
        "Upcoming Primeira Liga matches with published referee assignments. "
        "Career totals are provided by the Bzzoiro feed."
    )
    summary = (
        assigned.groupby("Referee")
        .agg(
            matches=("Date", "count"),
            career_games=("RefereeCareerGames", "max"),
            career_yellow=("RefereeCareerYellow", "max"),
            career_red=("RefereeCareerRed", "max"),
        )
        .reset_index()
        .sort_values("matches", ascending=False)
    )
    display = summary.rename(columns={
        "Referee": "Referee",
        "matches": "Upcoming matches",
        "career_games": "Career games",
        "career_yellow": "Career yellows",
        "career_red": "Career reds",
    })
    st.dataframe(display, hide_index=True, width="stretch")
    with st.expander("Upcoming fixtures with referees"):
        fixture_columns = ["Date", "HomeTeam", "AwayTeam", "Referee"]
        st.dataframe(
            assigned[fixture_columns].rename(columns={
                "Date": "Match date", "HomeTeam": "Home team",
                "AwayTeam": "Away team", "Referee": "Referee",
            }),
            hide_index=True,
            width="stretch",
        )


def render_model_lab(config: LeagueConfig) -> None:
    from .navigation import render_sidebar_branding
    render_sidebar_branding(config)
    _page_title(config, "Model Lab", "Performance and feature inspection")
    metrics = _read_csv("poisson_metrics_history.csv")
    if metrics.empty:
        st.info("Model evaluation history is not available in this deployment.")
    else:
        metrics = metrics.copy()
        metrics["_evaluation_date"] = pd.to_datetime(metrics["date"], errors="coerce")
        metrics = metrics.sort_values("_evaluation_date", ascending=False)
        latest = metrics.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Outcome accuracy", f"{float(latest['outcome_acc']):.1%}")
        c2.metric("Home-goal MAE", f"{float(latest['home_mae']):.2f}")
        c3.metric("Away-goal MAE", f"{float(latest['away_mae']):.2f}")
        display = metrics.drop(columns="_evaluation_date").rename(columns={
            "date": "Evaluation date", "home_mae": "Home-goal MAE",
            "away_mae": "Away-goal MAE", "home_rmse": "Home-goal RMSE",
            "away_rmse": "Away-goal RMSE", "outcome_acc": "Outcome accuracy",
        }).copy()
        display["Evaluation date"] = pd.to_datetime(display["Evaluation date"], errors="coerce")
        display = display.sort_values("Evaluation date", ascending=False).head(12).copy()
        display["Evaluation date"] = display["Evaluation date"].dt.strftime("%b %d, %Y")
        for column in ("Home-goal MAE", "Away-goal MAE", "Home-goal RMSE", "Away-goal RMSE"):
            display[column] = pd.to_numeric(display[column], errors="coerce").round(3)
        display["Outcome accuracy"] = pd.to_numeric(display["Outcome accuracy"], errors="coerce").map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "—"
        )
        st.subheader("Evaluation history")
        st.dataframe(display, hide_index=True, width="stretch")
    st.caption("Metrics shown here are from the persisted evaluation history used by the deployment.")


def render_raw_data(config: LeagueConfig) -> None:
    from .navigation import render_sidebar_branding
    render_sidebar_branding(config)
    _page_title(config, "Raw Data", "Source files and cache inspection")
    data_path = _data_dir()
    if not path.exists(data_path):
        st.info("The data directory is not available yet.")
        return
    files = pd.DataFrame(
        [{"File": name, "Updated": datetime.fromtimestamp(path.getmtime(path.join(data_path, name)))}
         for name in sorted(__import__("os").listdir(data_path))
         if name.lower().endswith((".csv", ".parquet", ".json"))]
    )
    st.dataframe(files, hide_index=True, width="stretch")
