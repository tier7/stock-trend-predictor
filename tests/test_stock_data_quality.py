from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import app as app_module
from services import data_fetcher, data_updater, db
from services.ml.dataset_service import prepare_dataset, prepare_features
from services.stock_data_quality_service import filter_valid_ohlc, valid_ohlc_mask


def stock_rows():
    return pd.DataFrame({
        "ticker": ["ALE.WA"] * 2,
        "interval": ["1d"] * 2,
        "date": ["2026-10-08", "2026-10-09"],
        "open": [50.0, 0.0], "high": [51.0, 0.0],
        "low": [49.0, 0.0], "close": [50.5, 51.75],
        "volume": [1000, 0], "dividends": [0.0, 0.0],
        "stock_splits": [0.0, 0.0],
    })


@pytest.mark.parametrize("prices", [
    [0, 0, 0, 51.75], [50, 51, 49, np.nan],
    [50, np.inf, 49, 50.5], [-50, 51, 49, 50.5],
    [52, 51, 49, 50.5], [50, 51, 50.6, 50.5],
    [50, 49, 51, 50.5],
])
def test_invalid_candles_are_excluded(prices):
    frame = stock_rows().iloc[:1].copy()
    frame.loc[frame.index[0], ["open", "high", "low", "close"]] = prices
    assert filter_valid_ohlc(frame).empty


def test_zero_volume_and_float32_rounding_do_not_invalidate_prices():
    frame = stock_rows()
    frame.loc[0, "volume"] = 0
    frame.loc[1, "volume"] = 1000
    frame.loc[1, ["open", "high", "low", "close"]] = [50, 49.999999, 50.000001, 50]
    assert valid_ohlc_mask(frame).all()


def test_flat_zero_volume_placeholder_cannot_overwrite_existing_prices(monkeypatch):
    frame = stock_rows().iloc[:1].copy()
    frame.loc[0, ["open", "high", "low", "close"]] = 50.0
    frame.loc[0, "volume"] = 0
    connect = Mock()
    monkeypatch.setattr(db, "get_connection", connect)
    db.save_stock_data(frame)
    connect.assert_not_called()


@pytest.mark.parametrize("start_date", [None, "2026-09-09"])
def test_fetch_requests_repair_and_excludes_unrepaired_candle(monkeypatch, start_date):
    raw = stock_rows().drop(columns=["ticker", "interval"]).rename(columns={
        "date": "Date", "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume", "dividends": "Dividends",
        "stock_splits": "Stock Splits",
    })
    raw["Date"] = pd.to_datetime(raw["Date"])
    download = Mock(return_value=raw.set_index("Date"))
    monkeypatch.setattr(data_fetcher.yf, "download", download)
    fetched = data_fetcher.fetch_stock_data("ALE.WA", start_date=start_date)
    assert download.call_args.kwargs["repair"] is True
    assert fetched["date"].tolist() == ["2026-10-08"]
    assert fetched.iloc[0]["open"] == 50


def test_save_skips_bad_candle_but_upserts_corrected_date(monkeypatch):
    connection = Mock()
    monkeypatch.setattr(db, "get_connection", Mock(return_value=connection))
    frame = stock_rows()
    db.save_stock_data(frame)
    query, rows = connection.cursor.return_value.executemany.call_args.args
    assert "ON CONFLICT" in query and "open = EXCLUDED.open" in query
    assert len(rows) == 1 and rows[0][2] == "2026-10-08"
    frame.loc[1, ["open", "high", "low"]] = [51.0, 52.0, 50.0]
    db.save_stock_data(frame.iloc[1:])
    assert connection.cursor.return_value.executemany.call_args.args[1][0][2] == "2026-10-09"


def test_save_does_not_connect_for_invalid_data(monkeypatch):
    connect = Mock()
    monkeypatch.setattr(db, "get_connection", connect)
    db.save_stock_data(stock_rows().iloc[1:])
    connect.assert_not_called()


def updater_mocks(monkeypatch, latest_dates, fetched):
    connection = Mock()
    connection.cursor.return_value.fetchone.return_value = latest_dates
    monkeypatch.setattr(data_updater, "get_connection", Mock(return_value=connection))
    fetch = Mock(return_value=fetched)
    save = Mock()
    monkeypatch.setattr(data_updater, "fetch_stock_data", fetch)
    monkeypatch.setattr(data_updater, "save_stock_data", save)
    return fetch, save


def test_updater_revisits_recent_rows_including_invalid_latest_date(monkeypatch):
    frame = stock_rows().iloc[:1]
    fetch, save = updater_mocks(monkeypatch, ("2026-10-09", "2026-10-08"), frame)
    assert data_updater.ensure_stock_data("ale.wa") is True
    fetch.assert_called_once_with(ticker="ALE.WA", interval="1d", start_date="2026-09-09")
    save.assert_called_once_with(frame)


def test_invalid_candle_today_does_not_suppress_refresh(monkeypatch):
    today = datetime.today().strftime("%Y-%m-%d")
    fetch, _ = updater_mocks(monkeypatch, (today, "2026-01-01"), pd.DataFrame())
    assert data_updater.ensure_stock_data("ALE.WA") is True
    fetch.assert_called_once()


def test_valid_candle_today_does_not_need_download(monkeypatch):
    today = datetime.today().strftime("%Y-%m-%d")
    fetch, _ = updater_mocks(monkeypatch, (today, today), pd.DataFrame())
    assert data_updater.ensure_stock_data("ALE.WA") is True
    fetch.assert_not_called()


@pytest.mark.parametrize("last_valid, expected", [("2026-10-08", True), (None, False)])
def test_offline_refresh_requires_existing_valid_history(monkeypatch, last_valid, expected):
    updater_mocks(monkeypatch, ("2026-10-09", last_valid), pd.DataFrame())
    assert data_updater.ensure_stock_data("ALE.WA") is expected


@pytest.mark.parametrize("interval", ["1d", "1w", "1mo"])
def test_chart_api_excludes_bad_candles_before_resampling(client, monkeypatch, interval):
    monkeypatch.setattr(app_module, "ensure_stock_data", lambda *args: True)
    monkeypatch.setattr(app_module, "get_stock_data", lambda *args: stock_rows())
    response = client.get(f"/api/stock/ALE.WA/candles?interval={interval}")
    assert response.status_code == 200
    assert response.get_json() == [{
        "time": "2026-10-08", "open": 50.0, "high": 51.0,
        "low": 49.0, "close": 50.5,
    }]


def test_invalid_session_does_not_shift_model_horizon_to_later_session():
    positions = np.arange(150)
    close = 100 + positions * 0.1 + np.sin(positions)
    frame = pd.DataFrame({
        "date": pd.bdate_range("2025-01-01", periods=len(positions)).strftime("%Y-%m-%d"),
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000,
    })
    frame.loc[100, ["open", "high", "low"]] = 0
    features = prepare_features(frame)
    assert features["date"].tolist() == frame["date"].tolist()
    assert features.loc[100, ["open", "high", "low", "close"]].isna().all()
    prepared = prepare_dataset(frame)
    assert not set(frame.loc[98:100, "date"]) & set(prepared["date"])
    previous = prepared[prepared["date"] == frame.loc[97, "date"]].iloc[0]
    assert previous["next_date"] == frame.loc[98, "date"]
    assert previous["target_end_date"] == frame.loc[99, "date"]
