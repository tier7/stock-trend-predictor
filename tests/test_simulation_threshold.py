from datetime import date, timedelta

import pandas as pd
import pytest

import app as app_module
from services.simulator_service import run_backtest


def predictions_with_probabilities(probabilities):
    predictions = []
    for position, probability in enumerate(probabilities):
        session = date(2026, 1, 2) + timedelta(days=position)
        predictions.append({
            "prediction": int(probability > 0.5),
            "probability_up": probability,
            "next_date": session.isoformat(),
            "next_open": 100 + position,
            "target_end_date": (session + timedelta(days=1)).isoformat(),
            "target_end_open": 101 + position,
        })
    return predictions


def test_default_threshold_keeps_60_40_policy_and_middle_probabilities_hold():
    predictions = predictions_with_probabilities([0.6, 0.45, 0.4])
    result = run_backtest(predictions, fee_rate=0)
    assert result == run_backtest(predictions, fee_rate=0, confidence_threshold=0.6)
    assert [(trade["type"], trade["date"]) for trade in result["trades"]] == [
        ("BUY", "2026-01-02"), ("SELL", "2026-01-04"),
    ]
    assert result["buy_threshold"] == 0.6
    assert result["sell_threshold"] == 0.4


def test_higher_threshold_filters_trades_for_the_same_predictions():
    predictions = predictions_with_probabilities([0.65, 0.4, 0.75, 0.3])
    lower = run_backtest(predictions, fee_rate=0, confidence_threshold=0.6)
    higher = run_backtest(predictions, fee_rate=0, confidence_threshold=0.7)
    assert lower["total_transactions"] == 4
    assert higher["total_transactions"] == 2
    assert higher["trades"][0]["date"] == "2026-01-04"
    assert higher["trades"][1]["date"] == "2026-01-05"


def test_mirrored_threshold_includes_decimal_boundary():
    result = run_backtest(predictions_with_probabilities([0.68, 0.32]), confidence_threshold=0.68)
    assert [trade["type"] for trade in result["trades"]] == ["BUY", "SELL"]


def test_at_50_percent_a_tie_keeps_cash_or_shares_instead_of_switching():
    result = run_backtest(
        predictions_with_probabilities([0.5, 0.55, 0.5, 0.45, 0.5]), confidence_threshold=0.5,
    )
    assert [(trade["type"], trade["date"]) for trade in result["trades"]] == [
        ("BUY", "2026-01-03"), ("SELL", "2026-01-05"),
    ]


def test_100_percent_threshold_only_trades_at_certainty_boundaries():
    result = run_backtest(predictions_with_probabilities([0.99, 1, 0.01, 0]), confidence_threshold=1)
    assert [(trade["type"], trade["date"]) for trade in result["trades"]] == [
        ("BUY", "2026-01-03"), ("SELL", "2026-01-05"),
    ]


@pytest.mark.parametrize("threshold", [None, 0.49, 1.01, float("nan"), float("inf"), "invalid"])
def test_backtest_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="Próg pewności"):
        run_backtest(predictions_with_probabilities([0.7, 0.3]), confidence_threshold=threshold)


@pytest.mark.parametrize("mode", ["local", "global", "walk_forward"])
@pytest.mark.parametrize("threshold", [None, 0.73])
def test_api_passes_default_or_selected_threshold_to_each_simulator(client, monkeypatch, mode, threshold):
    received = []
    monkeypatch.setattr(app_module, "ensure_stock_data", lambda *args: True)
    monkeypatch.setattr(app_module, "get_stock_data", lambda *args: pd.DataFrame({"close": [100, 101]}))
    service_name = {
        "local": "train_and_predict", "global": "simulate_with_global_model",
        "walk_forward": "simulate_global_walk_forward",
    }[mode]

    def capture_simulation(*args, **kwargs):
        received.append(kwargs["confidence_threshold"])
        return {"selected_threshold": kwargs["confidence_threshold"]}

    monkeypatch.setattr(app_module, service_name, capture_simulation)
    parameters = {"start_date": "2026-01-01", "end_date": "2026-02-01"}
    if threshold is not None:
        parameters["confidence_threshold"] = threshold
    if mode == "local":
        response = client.get("/api/stock/AAPL/prediction", query_string=parameters)
    else:
        endpoint = "global-simulation" if mode == "global" else "global-walk-forward"
        response = client.post(f"/api/stock/AAPL/{endpoint}", json=parameters)
    expected = 0.6 if threshold is None else threshold
    assert response.status_code == 200
    assert received == [expected]
    assert response.get_json() == {"selected_threshold": expected}


@pytest.mark.parametrize("mode", ["local", "global", "walk_forward"])
@pytest.mark.parametrize("threshold", ["nan", "invalid", 0.49, 1.01])
def test_api_rejects_invalid_threshold_before_fetching_or_training(client, monkeypatch, mode, threshold):
    monkeypatch.setattr(app_module, "ensure_stock_data", lambda *args: pytest.fail("Reject threshold before data loading"))
    parameters = {"start_date": "2026-01-01", "end_date": "2026-02-01", "confidence_threshold": threshold}
    if mode == "local":
        response = client.get("/api/stock/AAPL/prediction", query_string=parameters)
    else:
        endpoint = "global-simulation" if mode == "global" else "global-walk-forward"
        response = client.post(f"/api/stock/AAPL/{endpoint}", json=parameters)
    assert response.status_code == 400
    assert "Próg pewności" in response.get_json()["error"]
