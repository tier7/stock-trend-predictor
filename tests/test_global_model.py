import json
from datetime import datetime, timedelta
from threading import Event

import numpy as np
import pandas as pd
import pytest

import app as app_module
from services.ml import global_forest_service as global_service
from services.ml import global_training_service
from services.ml.dataset_service import FEATURE_COLUMNS, prepare_dataset, prepare_features
from services.ml.global_training_service import GlobalTrainingJobs
from services.ml import model_store_service
from services.ml.model_store_service import list_global_models, load_global_model, save_global_model
from services.ml.walk_forward_service import train


# joblib 1.5.3 restores tree arrays correctly, but uses an API deprecated in NumPy 2.5.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning:joblib.numpy_pickle"
)


@pytest.fixture
def prices():
    dates = pd.bdate_range("2024-01-02", periods=260)
    positions = np.arange(len(dates))
    close = 100 + positions * 0.1 + 2 * np.cos(positions)
    open_prices = close + 0.25 * np.sin(positions)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": open_prices,
        "high": np.maximum(open_prices, close) + 1,
        "low": np.minimum(open_prices, close) - 1,
        "close": close,
        "volume": 1000 + positions % 11 * 100,
    })


@pytest.fixture
def training_data(monkeypatch, prices):
    fetched_tickers = []

    def get_stock_data(ticker, years=None):
        fetched_tickers.append(ticker)
        return prices.copy()

    monkeypatch.setattr(global_service, "train_tickers", ["AAPL", "MSFT"])
    monkeypatch.setattr(global_service, "ensure_stock_data", lambda ticker, interval: True)
    monkeypatch.setattr(global_service, "get_stock_data", get_stock_data)
    return fetched_tickers


@pytest.fixture
def saved_model(tmp_path, prices, training_data):
    cutoff = prices["date"].iloc[180]
    return global_service.train_global_model(cutoff, tmp_path)


def test_training_persists_only_mature_samples_from_training_tickers(
    monkeypatch, tmp_path, prices, training_data,
):
    cutoff = prices["date"].iloc[180]
    training_frames = []

    def capture_train(frame, features):
        training_frames.append(frame.copy())
        return train(frame, features)

    monkeypatch.setattr(global_service, "train", capture_train)
    metadata = global_service.train_global_model(cutoff, tmp_path)
    frame = training_frames[0]
    assert training_data == ["AAPL", "MSFT"]
    assert set(frame["ticker"]) == {"AAPL", "MSFT"}
    assert (frame["date"] < cutoff).all()
    assert (frame["target_end_date"] <= cutoff).all()
    assert metadata["training_rows"] == len(frame)
    assert metadata["last_target_end_date"] <= cutoff
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert list_global_models(tmp_path) == [metadata]
    model, restored_metadata = load_global_model(metadata["model_id"], tmp_path)
    assert restored_metadata == metadata
    np.testing.assert_array_equal(model.classes_, [0, 1])
    assert model.n_estimators == 100


def test_saved_model_simulates_and_predicts_latest_without_training(
    monkeypatch, tmp_path, prices, saved_model,
):
    monkeypatch.setattr(global_service, "train", lambda *args: pytest.fail("Inference must not train"))
    parameters = (
        prices, saved_model["model_id"], prices["date"].iloc[200], prices["date"].iloc[229],
        10000, 0.001, tmp_path,
    )
    result = global_service.simulate_with_global_model(*parameters)
    repeated = global_service.simulate_with_global_model(*parameters)
    assert result == repeated
    stricter = global_service.simulate_with_global_model(*parameters, confidence_threshold=0.85)
    assert stricter["predictions"] == result["predictions"]
    assert stricter["accuracy"] == result["accuracy"]
    assert stricter["buyandhold_summary"] == result["buyandhold_summary"]
    assert stricter["model_metadata"] == saved_model
    assert stricter["model_summary"]["buy_threshold"] == 0.85
    assert stricter["model_summary"]["sell_threshold"] == 0.15
    assert result["total_predictions"] == 28
    assert result["training_rows"] == saved_model["training_rows"]
    assert len(result["model_summary"]["equity_history"]) == 29
    assert result["model_metadata"] == saved_model
    assert sum(result["confusion_matrix"].values()) == 28
    latest = global_service.predict_latest_with_global_model(prices, saved_model["model_id"], tmp_path)
    assert latest["date"] == prices["date"].iloc[-1]
    assert 0 <= latest["probability_up"] <= 1
    assert latest["prediction"] in [0, 1]
    json.dumps(result)
    json.dumps(latest)


@pytest.mark.parametrize("selection", ["latest", "older", "missing"])
def test_active_model_api_selects_ready_version_without_loading_or_training(
    client, monkeypatch, tmp_path, saved_model, selection,
):
    model, metadata = load_global_model(saved_model["model_id"], tmp_path)
    newer_metadata = save_global_model(model, {
        **metadata,
        "created_at": (datetime.fromisoformat(metadata["created_at"]) + timedelta(days=1)).isoformat(),
    }, tmp_path)
    selected_id = {
        "latest": None,
        "older": saved_model["model_id"],
        "missing": "global_forest_2024-01-01_00000000",
    }[selection]
    monkeypatch.setitem(app_module.app.config, "GLOBAL_MODEL_DIR", tmp_path)
    monkeypatch.setitem(app_module.app.config, "GLOBAL_MODEL_ID", selected_id)
    monkeypatch.setattr(model_store_service.joblib, "load", lambda *args: pytest.fail("Only metadata is needed"))
    monkeypatch.setattr(global_service, "train", lambda *args: pytest.fail("Selecting a model must not train"))
    monkeypatch.setattr(app_module, "ensure_stock_data", lambda *args: pytest.fail("No stock data is needed"))
    response = client.get("/api/models/global/active")
    if selection == "missing":
        assert response.status_code == 404
        assert "GLOBAL_MODEL_ID" in response.get_json()["error"]
    else:
        assert response.status_code == 200
        expected = newer_metadata if selection == "latest" else saved_model
        assert response.get_json() == {"model": expected}


def test_active_model_api_reports_missing_model_without_starting_training(client, monkeypatch, tmp_path):
    monkeypatch.setitem(app_module.app.config, "GLOBAL_MODEL_DIR", tmp_path)
    monkeypatch.setitem(app_module.app.config, "GLOBAL_MODEL_ID", None)
    monkeypatch.setattr(global_service, "train", lambda *args: pytest.fail("Missing model must not trigger training"))
    response = client.get("/api/models/global/active")
    assert response.status_code == 404
    assert "model lokalny" in response.get_json()["error"]


def test_historical_simulation_rejects_overlap_before_prediction(
    monkeypatch, tmp_path, prices, saved_model,
):
    monkeypatch.setattr(global_service, "prepare_dataset", lambda data: pytest.fail("Reject overlap first"))
    with pytest.raises(ValueError, match="może zaczynać się od"):
        global_service.predict_with_global_model(
            prices, saved_model["model_id"], prices["date"].iloc[179], prices["date"].iloc[229], tmp_path,
        )


def test_features_keep_latest_rows_and_match_training_representation(prices):
    features = prepare_features(prices)
    dataset = prepare_dataset(prices)
    assert features["date"].iloc[-1] == prices["date"].iloc[-1]
    assert dataset["date"].iloc[-1] == prices["date"].iloc[-3]
    assert "target" not in features
    pd.testing.assert_frame_equal(dataset[FEATURE_COLUMNS], features.loc[dataset.index, FEATURE_COLUMNS])
    truncated = prepare_features(prices.iloc[:220])
    pd.testing.assert_frame_equal(truncated[FEATURE_COLUMNS], features.iloc[:220][FEATURE_COLUMNS])


def test_latest_rejects_missing_features_instead_of_using_older_session(tmp_path, prices, saved_model):
    incomplete = prices.copy()
    incomplete.loc[incomplete.index[-1], "close"] = np.nan
    with pytest.raises(ValueError, match="No complete features"):
        global_service.predict_latest_with_global_model(incomplete, saved_model["model_id"], tmp_path)


@pytest.mark.parametrize("model_id", ["../other", "C:\\other.joblib", "", None])
def test_model_id_cannot_select_arbitrary_file(tmp_path, model_id):
    with pytest.raises(ValueError, match="identyfikator"):
        load_global_model(model_id, tmp_path)


def test_failed_training_does_not_remove_saved_version(monkeypatch, tmp_path, saved_model):
    def fail_training(*args):
        raise ValueError("Brak próbek treningowych.")

    monkeypatch.setattr(global_training_service, "train_global_model", fail_training)
    jobs = GlobalTrainingJobs()
    jobs.start(saved_model["training_cutoff"], tmp_path)
    jobs._thread.join(timeout=10)
    assert not jobs._thread.is_alive()
    assert jobs.snapshot()["status"] == "failed"
    assert jobs.snapshot()["message"] == "Brak próbek treningowych."
    assert list_global_models(tmp_path) == [saved_model]


def test_duplicate_training_is_rejected_while_job_runs(monkeypatch, tmp_path):
    entered = Event()
    release = Event()

    def waiting_training(*args):
        entered.set()
        assert release.wait(timeout=10)
        return {"model_id": "test-model"}

    monkeypatch.setattr(global_training_service, "train_global_model", waiting_training)
    jobs = GlobalTrainingJobs()
    try:
        state = jobs.start("2025-01-01", tmp_path)
        assert state["status"] == "running"
        assert entered.wait(timeout=10)
        with pytest.raises(RuntimeError, match="już trwa"):
            jobs.start("2025-01-01", tmp_path)
    finally:
        release.set()
        jobs._thread.join(timeout=10)
    assert jobs.snapshot()["status"] == "completed"


def test_api_training_persistence_simulation_and_latest_forecast(
    client, monkeypatch, tmp_path, prices, training_data,
):
    jobs = GlobalTrainingJobs()
    monkeypatch.setattr(app_module, "global_training_jobs", jobs)
    monkeypatch.setitem(app_module.app.config, "GLOBAL_MODEL_DIR", tmp_path)
    monkeypatch.setattr(app_module, "ensure_stock_data", lambda ticker, interval: True)
    monkeypatch.setattr(app_module, "get_stock_data", lambda ticker, interval, years: prices.copy())
    before = client.get("/api/models/global")
    assert before.get_json() == {"models": [], "training": None}
    response = client.post("/api/models/global/train", json={"training_cutoff": prices["date"].iloc[180]})
    assert response.status_code == 202
    jobs._thread.join(timeout=10)
    assert not jobs._thread.is_alive()
    status = client.get("/api/models/global").get_json()
    assert status["training"]["status"] == "completed"
    assert len(status["models"]) == 1
    model_id = status["models"][0]["model_id"]
    monkeypatch.setattr(global_service, "train", lambda *args: pytest.fail("API inference must not train"))
    simulation = client.post("/api/stock/nvda/global-simulation", json={
        "model_id": model_id,
        "start_date": prices["date"].iloc[200],
        "end_date": prices["date"].iloc[229],
        "initial_cash": 25000,
        "fee_rate": 0,
    })
    assert simulation.status_code == 200
    assert simulation.get_json()["total_predictions"] == 28
    assert simulation.get_json()["model_summary"]["initial_cash"] == 25000
    latest = client.get("/api/stock/NVDA/global-prediction", query_string={"model_id": model_id})
    assert latest.status_code == 200
    assert latest.get_json()["date"] == prices["date"].iloc[-1]
    assert "NVDA" not in status["models"][0]["train_tickers"]


@pytest.mark.parametrize("cutoff", [None, "2025-13-01", "20250101", "2999-01-01"])
def test_training_api_rejects_invalid_cutoff_without_starting_job(client, monkeypatch, cutoff):
    jobs = GlobalTrainingJobs()
    monkeypatch.setattr(app_module, "global_training_jobs", jobs)
    response = client.post("/api/models/global/train", json={"training_cutoff": cutoff})
    assert response.status_code == 400
    assert jobs.snapshot() is None


@pytest.mark.parametrize("field,value", [("initial_cash", 0), ("fee_rate", -1), ("initial_cash", "nan"), ("fee_rate", "inf")])
def test_global_simulation_api_rejects_invalid_money_before_loading_data(client, monkeypatch, field, value):
    monkeypatch.setattr(app_module, "ensure_stock_data", lambda *args: pytest.fail("Validate money first"))
    response = client.post("/api/stock/AAPL/global-simulation", json={field: value})
    assert response.status_code == 400


@pytest.mark.parametrize("ticker", ["AAPL", "NVDA", "ALE.WA"])
def test_walk_forward_simulates_selected_stock_and_retrains_every_20_sessions(
    monkeypatch, prices, training_data, ticker,
):
    training_frames = []

    def capture_train(frame, features):
        training_frames.append(frame.copy())
        return train(frame, features)

    monkeypatch.setattr(global_service, "train", capture_train)
    monkeypatch.setattr(global_service, "save_global_model", lambda *args: pytest.fail("Simulation must not save models"))
    monkeypatch.setattr(global_service, "load_global_model", lambda *args: pytest.fail("No saved model is required"))
    start = prices["date"].iloc[180]
    end = prices["date"].iloc[226]
    result = global_service.simulate_global_walk_forward(prices, ticker, start, end, 25000, 0.001)

    assert training_data == ["AAPL", "MSFT"]
    assert result["model_type"] == "global_walk_forward"
    assert result["retrain_interval"] == 20
    assert result["training_runs"] == len(training_frames) == 3
    assert [block["test_rows"] for block in result["training_blocks"]] == [20, 20, 5]
    assert result["total_predictions"] == 45
    assert result["train_tickers"] == ["AAPL", "MSFT"]
    assert result["ticker_seen_during_training"] is (ticker == "AAPL")
    assert result["model_summary"]["initial_cash"] == 25000
    assert result["buyandhold_summary"]["buy_fee"] == pytest.approx(25000 / 1.001 * 0.001)
    assert result["effective_test_end_date"] == end
    assert sum(result["confusion_matrix"].values()) == 45
    assert len(result["model_summary"]["equity_history"]) == 46
    assert all(0 <= row["probability_up"] <= 1 for row in result["predictions"])
    assert [row["date"] for row in result["predictions"]] == prices["date"].iloc[180:225].tolist()

    for frame, block in zip(training_frames, result["training_blocks"]):
        assert set(frame["ticker"]) == {"AAPL", "MSFT"}
        assert (frame["date"] < block["start_date"]).all()
        assert (frame["target_end_date"] <= block["start_date"]).all()
        assert len(frame) == block["training_rows"]
    assert all(len(earlier) < len(later) for earlier, later in zip(training_frames, training_frames[1:]))
    json.dumps(result)


def test_walk_forward_api_runs_without_saved_model(client, monkeypatch, prices, training_data):
    monkeypatch.setattr(app_module, "ensure_stock_data", lambda *args: True)
    monkeypatch.setattr(app_module, "get_stock_data", lambda *args: prices.copy())
    response = client.post("/api/stock/nvda/global-walk-forward", json={
        "start_date": prices["date"].iloc[180],
        "end_date": prices["date"].iloc[226],
        "initial_cash": 25000,
        "fee_rate": 0,
    })
    assert response.status_code == 200
    result = response.get_json()
    assert result["training_runs"] == 3
    assert result["total_predictions"] == 45
    assert result["ticker_seen_during_training"] is False
    assert result["model_summary"]["initial_cash"] == 25000
    assert result["buyandhold_summary"]["buy_fee"] == 0
    assert all(trade["fee"] == 0 for trade in result["model_summary"]["trades"])


@pytest.mark.parametrize("parameters", [
    {}, {"start_date": "2025-01-01"},
    {"start_date": "20250101", "end_date": "2025-12-31"},
    {"start_date": "2025-02-30", "end_date": "2025-12-31"},
    {"start_date": "2025-12-31", "end_date": "2025-01-01"},
    {"start_date": "2025-01-01", "end_date": "2025-01-01"},
])
def test_walk_forward_api_validates_dates_before_loading_data(client, monkeypatch, parameters):
    monkeypatch.setattr(app_module, "ensure_stock_data", lambda *args: pytest.fail("Validate dates first"))
    response = client.post("/api/stock/AAPL/global-walk-forward", json=parameters)
    assert response.status_code == 400


@pytest.mark.parametrize("field,value", [("initial_cash", 0), ("fee_rate", -1), ("initial_cash", "nan"), ("fee_rate", "inf")])
def test_walk_forward_api_validates_money_before_loading_data(client, monkeypatch, field, value):
    monkeypatch.setattr(app_module, "ensure_stock_data", lambda *args: pytest.fail("Validate money first"))
    response = client.post("/api/stock/AAPL/global-walk-forward", json={
        "start_date": "2025-01-01", "end_date": "2025-12-31", field: value,
    })
    assert response.status_code == 400


def test_walk_forward_api_handles_missing_stock_and_training_data(client, monkeypatch, prices):
    parameters = {"start_date": prices["date"].iloc[180], "end_date": prices["date"].iloc[226]}
    monkeypatch.setattr(app_module, "ensure_stock_data", lambda *args: False)
    assert client.post("/api/stock/AAPL/global-walk-forward", json=parameters).status_code == 404
    monkeypatch.setattr(app_module, "ensure_stock_data", lambda *args: True)
    monkeypatch.setattr(app_module, "get_stock_data", lambda *args: pd.DataFrame())
    assert client.post("/api/stock/AAPL/global-walk-forward", json=parameters).status_code == 404
    monkeypatch.setattr(app_module, "get_stock_data", lambda *args: prices.copy())

    def fail_training_data(*args):
        raise ValueError("Brak danych spółki MSFT.")

    monkeypatch.setattr(global_service, "connect_data", fail_training_data)
    response = client.post("/api/stock/AAPL/global-walk-forward", json=parameters)
    assert response.status_code == 400
    assert response.get_json() == {"error": "Brak danych spółki MSFT."}


def test_walk_forward_rejects_insufficient_history_before_fitting(monkeypatch, prices, training_data):
    monkeypatch.setattr(global_service, "train", lambda *args: pytest.fail("Cannot fit without enough history"))
    with pytest.raises(ValueError, match="Za mało danych treningowych"):
        global_service.simulate_global_walk_forward(
            prices, "NVDA", prices["date"].iloc[40], prices["date"].iloc[70], 10000, 0.001,
        )


def test_walk_forward_rejects_empty_test_period_before_loading_40_stocks(monkeypatch, prices):
    monkeypatch.setattr(global_service, "connect_data", lambda *args: pytest.fail("Validate test period first"))
    with pytest.raises(ValueError, match="Brak danych do symulacji"):
        global_service.simulate_global_walk_forward(prices, "NVDA", "2020-01-01", "2020-12-31", 10000, 0.001)
