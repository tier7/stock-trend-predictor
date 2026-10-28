import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from services.ml.feature_schema import MARKET_FEATURE_COLUMNS
from services.ml.prediction_time_service import MARKET_CLOSE_BUFFER_MINUTES

SUPPORTED_US_EXCHANGES = (
    "NMS", "NGM", "NCM", "NYQ", "ASE", "PCX", "BTS",
)

MARKET_FEATURE_POLICY = {
    "version": 1,
    "profile": "us_equities",
    "source": "yfinance",
    "benchmark_ticker": "SPY",
    "volatility_ticker": "^VIX",
    "benchmark_calendar": "NYSE",
    "volatility_calendar": "CBOE_Index_Options",
    "close_buffer_minutes": MARKET_CLOSE_BUFFER_MINUTES,
    "supported_exchanges": list(SUPPORTED_US_EXCHANGES),
    "exchange_timezone": "America/New_York",
    "instrument_type": "EQUITY",
    "currency": "USD",
    "price_basis": "database_close",
    "return_windows": [1, 5],
    "volatility_window": 20,
    "volatility_ddof": 1,
    "volatility_annualized": False,
    "sma_window": 20,
    "vix_change": "difference_in_index_points",
    "alignment": "same_session_date",
    "missing_values": "keep_nan"
}


def validate_market_metadata(ticker, metadata):
    exchange = metadata.get("exchangeName")

    if exchange not in SUPPORTED_US_EXCHANGES:
        raise ValueError(
            f"{ticker}: market features do not support exchange {exchange}")

    if metadata.get("exchangeTimezoneName") != "America/New_York":
        raise ValueError(
            f"{ticker}: market features require America/New_York timezone")

    if metadata.get("instrumentType") != "EQUITY":
        raise ValueError(
            f"{ticker}: market features currently support equities only")

    if metadata.get("currency") != "USD":
        raise ValueError(
            f"{ticker}: market features currently require USD quotations")


def prepare_market_prices(data, source_name):
    if data.empty:
        raise ValueError(f"{source_name}: empty price data")

    missing_columns = [column for column in ("date", "close") if column not in data.columns]

    if missing_columns:
        raise ValueError(f"{source_name}: missing columns: {missing_columns}")

    prices = data[["date", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="raise").dt.normalize()
    prices["close"] = pd.to_numeric(prices["close"], errors="raise")

    if prices["date"].isna().any():
        raise ValueError(f"{source_name}: missing session dates")

    if prices["date"].duplicated().any():
        raise ValueError(f"{source_name}: duplicate session dates")

    values = prices["close"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{source_name}: invalid closing prices")

    return prices.sort_values("date").reset_index(drop=True)

def build_market_features(spy_data, vix_data):
    spy = prepare_market_prices(spy_data, "SPY").set_index("date")["close"]
    vix = prepare_market_prices(vix_data, "VIX").set_index("date")["close"]

    calendar = mcal.get_calendar("NYSE")
    sessions = calendar.valid_days(start_date=min(spy.index.min(), vix.index.min()), end_date=max(spy.index.max(), vix.index.max()), tz=None).rename("date")

    if (not spy.index.isin(sessions).all() or not vix.index.isin(sessions).all()):
        raise ValueError("Market prices contain non session dates")

    spy = spy.reindex(sessions)
    vix = vix.reindex(sessions)
    daily_returns = spy.pct_change(fill_method=None)
    sma_20 = spy.rolling(window=20, min_periods=20).mean()

    features = pd.DataFrame(index=sessions)

    features["market_return_1d"] = daily_returns
    features["market_return_5d"] = spy / spy.shift(5) - 1
    features["market_volatility_20"] = daily_returns.rolling(window=20, min_periods=20).std(ddof=1)
    features["market_price_vs_sma_20"] = spy / sma_20 - 1

    features["vix_level"] = vix
    features["vix_change_1d"] = vix.diff()

    return features.reset_index()[["date"] + MARKET_FEATURE_COLUMNS]


def filter_completed_market_prices(data, source_name, calendar_name, as_of):
    prices = prepare_market_prices(data, source_name)
    snapshot_at = pd.to_datetime(as_of, utc=True)

    if pd.isna(snapshot_at):
        raise ValueError("as_of cannot be missing")

    calendar = mcal.get_calendar(calendar_name)
    schedule = calendar.schedule(start_date=prices["date"].min(),end_date=prices["date"].max(),)

    if not prices["date"].isin(schedule.index).all():
        raise ValueError(f"{source_name}: prices contain non-session dates")

    available_at = schedule["market_close"] + pd.Timedelta(minutes=MARKET_CLOSE_BUFFER_MINUTES)
    completed_dates = schedule.index[available_at <= snapshot_at]

    result = prices.loc[prices["date"].isin(completed_dates)].copy()
    if result.empty:
        raise ValueError(f"{source_name}: no completed sessions at as_of")

    return result

