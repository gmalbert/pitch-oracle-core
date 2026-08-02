import numpy as np
import pandas as pd
import pytest
import torch.nn as nn

from models.lstm_predictor import FootballLSTM, LSTMPredictor
from models.neural_predictor import FootballNet


def test_lstm_returns_logits_for_cross_entropy_loss():
    model = FootballLSTM(input_size=17)
    assert not any(isinstance(layer, nn.Softmax) for layer in model.fc)


def test_feed_forward_network_also_returns_logits_for_cross_entropy_loss():
    model = FootballNet(input_size=4)
    assert not any(isinstance(layer, nn.Softmax) for layer in model.network)


def test_match_prediction_blends_both_teams_perspectives(monkeypatch):
    predictor = LSTMPredictor(sequence_length=2)
    calls = iter([
        np.array([[0.70, 0.20, 0.10]]),  # home team: win/draw/loss
        np.array([[0.40, 0.30, 0.30]]),  # away team: win/draw/loss
    ])
    monkeypatch.setattr(predictor, "predict_proba", lambda _: next(calls))

    rows = []
    for index in range(2):
        rows.append({
            "MatchDate": f"2025-01-0{index + 1}", "HomeTeam": "Home FC",
            "AwayTeam": "Other FC", "FullTimeResult": "H",
        })
        rows.append({
            "MatchDate": f"2025-01-0{index + 3}", "HomeTeam": "Other FC",
            "AwayTeam": "Away FC", "FullTimeResult": "A",
        })

    result = predictor.predict_match("Home FC", "Away FC", pd.DataFrame(rows), recent_matches=2)

    assert result["HomeWinProb"] == pytest.approx(0.50)
    assert result["DrawProb"] == pytest.approx(0.25)
    assert result["AwayWinProb"] == pytest.approx(0.25)
    assert sum(result[key] for key in ("HomeWinProb", "DrawProb", "AwayWinProb")) == pytest.approx(1)
