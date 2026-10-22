import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from services.ml import global_forest_service, model_service
from services.ml.dataset_service import FEATURE_COLUMNS
from services.ml.walk_forward_service import build_date_blocks, train


def prepared_data(ticker, test_dates):
    rows = []

    def add_row(date, target_end_date=None):
        date = pd.Timestamp(date).to_pydatetime()
        position = len(rows)
        row = {
            "ticker": ticker,
            "date": date.strftime("%Y-%m-%d"),
            "close": 100.0 + position,
            "next_date": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
            "next_open": 101.0 + position,
            "next_close": 102.0 + position,
            "target_end_date": target_end_date or (
                date + timedelta(days=2)
            ).strftime("%Y-%m-%d"),
            "target_end_open": 102.0 + position,
            "target": position % 2,
        }
        for feature in FEATURE_COLUMNS:
            row[feature] = float(position % 2) + position / 1000
        rows.append(row)

    for date in pd.date_range("2024-06-01", periods=120):
        add_row(date)
    add_row("2024-12-30", target_end_date="2025-01-03")
    for date in test_dates:
        add_row(date)
    return pd.DataFrame(rows)


@pytest.fixture
def shared_experiment(monkeypatch):
    stock = prepared_data("AAPL", ["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"])
    held_out = prepared_data("NVDA", ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"])
    combined = pd.concat([stock, held_out], ignore_index=True)
    test_df = combined[combined["date"] >= "2025-01-01"]
    date_blocks = build_date_blocks(test_df, 2)

    monkeypatch.setattr(model_service, "prepare_dataset", lambda data: data.copy())
    monkeypatch.setattr(global_forest_service, "connect_data", lambda tickers: combined.copy())
    monkeypatch.setattr(global_forest_service, "tickers", ["AAPL", "NVDA"])
    monkeypatch.setattr(global_forest_service, "train_tickers", ["AAPL"])
    monkeypatch.setattr(global_forest_service, "test_tickers", ["NVDA"])
    monkeypatch.setattr(global_forest_service, "RETRAIN_INTERVAL", 2)
    return stock, date_blocks


def test_local_and_global_share_training_cutoffs_and_predictions(monkeypatch, shared_experiment):
    stock, date_blocks = shared_experiment
    global_training = []
    local_training = []

    def global_train(frame, features):
        global_training.append(frame.copy())
        return train(frame, features)

    def local_train(frame, features):
        local_training.append(frame.copy())
        return train(frame, features)

    monkeypatch.setattr(global_forest_service, "train", global_train)
    monkeypatch.setattr(model_service, "train", local_train)

    seen, unseen = global_forest_service.run_experiment("2025-01-01", "2025-01-31")
    local_result = model_service.train_and_predict(
        stock, "2025-01-01", "2025-01-31", 10000, 0.001,
        date_blocks=date_blocks,
    )

    assert len(global_training) == len(local_training) == len(date_blocks)
    for global_frame, local_frame, block_dates in zip(global_training, local_training, date_blocks):
        pd.testing.assert_frame_equal(global_frame, local_frame)
        assert set(global_frame["ticker"]) == {"AAPL"}
        assert (global_frame["date"] < block_dates[0]).all()
        assert (global_frame["target_end_date"] <= block_dates[0]).all()

    assert "2024-12-30" not in local_training[0]["date"].tolist()
    seen = seen.sort_values("date")
    records = local_result["predictions"]
    assert [record["date"] for record in records] == seen["date"].tolist()
    assert [record["prediction"] for record in records] == seen["prediction"].tolist()
    np.testing.assert_allclose(
        [record["probability_up"] for record in records], seen["probability_up"],
    )
    assert set(unseen["ticker"]) == {"NVDA"}
    assert len(seen) == 4
    assert len(unseen) == 5
    assert local_result["accuracy"] == global_forest_service.evaluate(seen)["accuracy"]
    json.dumps(local_result)


@pytest.mark.parametrize("threshold,expected_trades", [(0.6, ["BUY", "SELL", "BUY"]), (0.9, [])])
def test_local_keeps_predictions_from_each_block_for_backtest(monkeypatch, shared_experiment, threshold, expected_trades):
    stock, date_blocks = shared_experiment
    probabilities = iter([0.8, 0.2, 0.8])

    class FixedForest:
        classes_ = np.array([0, 1])
        def __init__(self, probability_up):
            self.probability_up = probability_up

        def predict(self, features):
            return np.full(len(features), int(self.probability_up >= 0.5))

        def predict_proba(self, features):
            return np.tile([1 - self.probability_up, self.probability_up], (len(features), 1))

    monkeypatch.setattr(model_service, "train", lambda frame, features: FixedForest(next(probabilities)))
    result = model_service.train_and_predict(
        stock, "2025-01-01", "2025-01-31", 10000, 0,
        date_blocks=date_blocks,
        confidence_threshold=threshold,
    )

    assert [record["prediction"] for record in result["predictions"]] == [1, 0, 0, 1]
    assert [record["probability_up"] for record in result["predictions"]] == [0.8, 0.2, 0.2, 0.8]
    assert result["total_predictions"] == 4
    assert [trade["type"] for trade in result["model_summary"]["trades"]] == expected_trades
    assert result["model_summary"]["confidence_threshold"] == threshold
    json.dumps(result)


def test_local_rejects_incomplete_shared_schedule_before_training(monkeypatch, shared_experiment):
    stock, _ = shared_experiment

    def unexpected_train(frame, features):
        pytest.fail("An incomplete schedule must be rejected before training")

    monkeypatch.setattr(model_service, "train", unexpected_train)
    with pytest.raises(ValueError, match="cover all test dates"):
        model_service.train_and_predict(
            stock, "2025-01-01", "2025-01-31", 10000, 0,
            date_blocks=[["2025-01-03"]],
        )


def test_training_requires_both_classes_for_probability_column(shared_experiment):
    stock, _ = shared_experiment
    stock["target"] = 0
    with pytest.raises(ValueError, match="both target classes"):
        train(stock, FEATURE_COLUMNS)


def test_local_runs_from_raw_prices_through_indicators_and_backtest():
    dates = pd.bdate_range("2024-01-02", periods=220)
    positions = np.arange(len(dates))
    close = 100 + positions * 0.1 + 2 * np.cos(positions)
    open_prices = close + 0.25 * np.sin(positions)
    prices = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": open_prices,
        "high": np.maximum(open_prices, close) + 1,
        "low": np.minimum(open_prices, close) - 1,
        "close": close,
        "volume": 1000 + positions % 11 * 100,
    })

    result = model_service.train_and_predict(
        prices, dates[180].strftime("%Y-%m-%d"), dates[209].strftime("%Y-%m-%d"),
        10000, 0.001,
    )

    assert result["total_predictions"] == 28
    assert result["effective_test_end_date"] == dates[209].strftime("%Y-%m-%d")
    assert sum(result["confusion_matrix"].values()) == 28
    assert all(0 <= record["probability_up"] <= 1 for record in result["predictions"])
    assert len(result["model_summary"]["equity_history"]) == 29
    json.dumps(result)
