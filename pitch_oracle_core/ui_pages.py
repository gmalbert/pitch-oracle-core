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
from .risk import calculate_prediction_risk, get_prediction_guidance, get_risk_category


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
    if "Risk score" in result:
        result["Risk score"] = pd.to_numeric(result["Risk score"], errors="coerce").round(1)
    if "Status" in result:
        result["Status"] = result["Status"].astype(str).str.replace("STATUS_", "", regex=False).str.title()
    preferred = [
        "Match date", "Kickoff", "Home team", "Away team", "Home win", "Draw",
        "Away win", "Model pick", "Confidence", "Risk score", "Risk level",
        "Recommendation", "Status",
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


def _page_title(config: LeagueConfig, title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(f"{config.display_name} · {subtitle}")


def render_overview(config: LeagueConfig) -> None:
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
    display = _display_predictions(predictions)
    st.dataframe(
        display,
        hide_index=True,
        height=_height(display),
        width="stretch",
        column_config=_prediction_column_config(),
    )


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
    recent = pd.concat([home, away]).copy()
    if "MatchDate" in recent.columns:
        recent["_sort_date"] = pd.to_datetime(recent["MatchDate"], errors="coerce")
        recent = recent.sort_values("_sort_date", ascending=False).drop(columns="_sort_date").head(10)
    else:
        recent = recent.tail(10)
    st.dataframe(recent, hide_index=True, height=_height(recent), width="stretch")


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
