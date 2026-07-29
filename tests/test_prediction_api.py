import pytest
import app as app_module
import pandas as pd



@pytest.fixture
def available_stock_data(monkeypatch):
    fake_dataframe = pd.DataFrame({
        "close": [100, 101]
    })

    def fake_ensure_stock_data(ticker, interval):
        return True

    def fake_get_stock_data(ticker, interval, start_date):
        return fake_dataframe

    monkeypatch.setattr(
        app_module,
        "ensure_stock_data",
        fake_ensure_stock_data
    )
    monkeypatch.setattr(
        app_module,
        "get_stock_data",
        fake_get_stock_data
    )

    return fake_dataframe

@pytest.mark.parametrize(
    "query_string",
    [
        {},
        {"start_date": "2026-01-01"},
        {"end_date": "2026-12-31"},
    ],
    ids=[
        "missing-both-dates",
        "missing-end-date",
        "missing-start-date",
    ]
)


def test_prediction_endpoint_requires_both_dates(client, query_string):
    response = client.get("/api/stock/AAPL/prediction", query_string=query_string)

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Missing start or end date"
    }

def test_prediction_endpoint_returns_404_when_stock_data_cannot_be_fetched(client,monkeypatch):
    monkeypatch.setattr(app_module,"ensure_stock_data", lambda ticker, interval: False)

    response = client.get("/api/stock/aapl/prediction",
        query_string={
            "start_date": "2026-01-01",
            "end_date": "2026-06-01",
        }
    )

    assert response.status_code == 404
    assert response.get_json() == {"error": "No data found for ticker AAPL"}

def test_prediction_endpoint_returns_404_when_database_is_empty(client,monkeypatch):
    def fake_ensure_stock_data(ticker, interval):
        return True

    def fake_get_stock_data(ticker, interval, start_date):
        return pd.DataFrame()

    monkeypatch.setattr(app_module,"ensure_stock_data",fake_ensure_stock_data)
    monkeypatch.setattr(app_module,"get_stock_data",fake_get_stock_data)

    response = client.get(
        "/api/stock/AAPL/prediction",
        query_string={
            "start_date": "2026-01-01",
            "end_date": "2026-06-01",
        }
    )

    assert response.status_code == 404
    assert response.get_json() == {
        "error": "No data found for ticker AAPL"
    }

def test_prediction_endpoint_returns_model_result(client, monkeypatch, available_stock_data):


    expected_result = {
        "accuracy": 0.55,
        "total_predictions": 2,
        "predictions": [],
    }

    received_arguments = {}


    def fake_train_and_predict(data,start_date,end_date,initial_cash,fee_rate):
        received_arguments["data"] = data
        received_arguments["start_date"] = start_date
        received_arguments["end_date"] = end_date
        received_arguments["initial_cash"] = initial_cash
        received_arguments["fee_rate"] = fee_rate
        return expected_result

    monkeypatch.setattr(app_module,"train_and_predict",fake_train_and_predict)

    response = client.get("/api/stock/AAPL/prediction",
        query_string={
            "start_date": "2026-01-01",
            "end_date": "2026-06-01",
            "initial_cash": "25000",
            "fee_rate": "0.0025",
        }
    )

    assert response.status_code == 200
    assert response.get_json() == expected_result

    assert received_arguments["data"] is available_stock_data
    assert received_arguments["start_date"] == "2026-01-01"
    assert received_arguments["end_date"] == "2026-06-01"
    assert received_arguments["initial_cash"] == pytest.approx(25000)
    assert received_arguments["fee_rate"] == pytest.approx(0.0025)

def test_prediction_endpoint_returns_400_for_model_validation_error(client,monkeypatch,available_stock_data):


    def fake_train_and_predict(data,start_date,end_date,initial_cash,fee_rate):
        raise ValueError("End date must be later than start date.")

    monkeypatch.setattr(app_module,"train_and_predict",fake_train_and_predict)

    response = client.get(
        "/api/stock/AAPL/prediction",
        query_string={
            "start_date": "2026-06-01",
            "end_date": "2026-01-01",
        }
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "End date must be later than start date."
    }

def test_prediction_endpoint_uses_default_backtest_parameters(client,monkeypatch,available_stock_data):
    received_arguments = {}

    def fake_train_and_predict(data,start_date,end_date,initial_cash,fee_rate):
        received_arguments["initial_cash"] = initial_cash
        received_arguments["fee_rate"] = fee_rate
        return {"predictions": []}

    monkeypatch.setattr(
        app_module,
        "train_and_predict",
        fake_train_and_predict
    )

    response = client.get(
        "/api/stock/AAPL/prediction",
        query_string={
            "start_date": "2026-01-01",
            "end_date": "2026-06-01",
        }
    )

    assert response.status_code == 200
    assert received_arguments["initial_cash"] == 10000
    assert received_arguments["fee_rate"] == 0.001

def test_prediction_endpoint_rejects_negative_initial_cash(client,monkeypatch):

    def fake_ensure_stock_data(ticker, interval):
        return False

    monkeypatch.setattr(
        app_module,
        "ensure_stock_data",
        fake_ensure_stock_data
    )

    response = client.get(
        "/api/stock/AAPL/prediction",
        query_string={
            "start_date": "2026-01-01",
            "end_date": "2026-06-01",
            "initial_cash": "-1000",
        }
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Initial cash must be greater than zero."
    }

def test_prediction_endpoint_rejects_negative_fee_rate(client,monkeypatch):
    def fake_ensure_stock_data(ticker, interval):
        return False

    monkeypatch.setattr(
        app_module,
        "ensure_stock_data",
        fake_ensure_stock_data
    )

    response = client.get(
        "/api/stock/AAPL/prediction",
        query_string={
            "start_date": "2026-01-01",
            "end_date": "2026-06-01",
            "fee_rate": "-0.001",
        }
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Fee rate cannot be negative."
    }

def test_prediction_endpoint_accepts_zero_fee_rate(client,monkeypatch, available_stock_data):
    fake_dataframe = pd.DataFrame({
        "close": [100]
    })

    received_arguments = {}

    def fake_train_and_predict(
        data,
        start_date,
        end_date,
        initial_cash,
        fee_rate
    ):
        received_arguments["fee_rate"] = fee_rate
        return {"predictions": []}


    monkeypatch.setattr(
        app_module,
        "train_and_predict",
        fake_train_and_predict
    )

    response = client.get(
        "/api/stock/AAPL/prediction",
        query_string={
            "start_date": "2026-01-01",
            "end_date": "2026-06-01",
            "fee_rate": "0",
        }
    )

    assert response.status_code == 200
    assert received_arguments["fee_rate"] == 0.0

def test_prediction_endpoint_rejects_non_numeric_initial_cash(
    client,
    monkeypatch
):
    def fake_ensure_stock_data(ticker, interval):
        return False

    monkeypatch.setattr(
        app_module,
        "ensure_stock_data",
        fake_ensure_stock_data
    )

    response = client.get(
        "/api/stock/AAPL/prediction",
        query_string={
            "start_date": "2026-01-01",
            "end_date": "2026-06-01",
            "initial_cash": "abc",
        }
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Initial cash must be a number."
    }

def test_prediction_endpoint_rejects_non_numeric_fee_rate(client,monkeypatch):
    def fake_ensure_stock_data(ticker, interval):
        return False

    monkeypatch.setattr(
        app_module,
        "ensure_stock_data",
        fake_ensure_stock_data
    )

    response = client.get(
        "/api/stock/AAPL/prediction",
        query_string={
            "start_date": "2026-01-01",
            "end_date": "2026-06-01",
            "fee_rate": "abc",
        }
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Fee rate must be a number."
    }