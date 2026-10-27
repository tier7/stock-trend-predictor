from datetime import date

import pandas as pd

from services.data_updater import ensure_stock_data
from services.db import get_stock_data
from services.ml.dataset_service import prepare_dataset, prepare_latest_features
from services.ml.finbert_service import FinbertAnalyzer
from services.news_fetcher import ensure_news_data
from services.ml.feature_schema import get_feature_columns
from services.ml.news_feature_service import normalize_training_news_range
from services.ml.market_feature_service import build_market_features, filter_completed_market_prices
from services.ml.prediction_time_service import get_latest_prediction_window
from services.data_fetcher import fetch_market_metadata
from services.ml.market_feature_service import validate_market_metadata


def load_market_features(progress_callback=None, *, as_of=None):
    now = pd.Timestamp.now(tz="UTC")
    snapshot_at = (now if as_of is None else pd.to_datetime(as_of, utc=True))

    if pd.isna(snapshot_at) or snapshot_at > now:
        raise ValueError("as_of must be a valid time not in the future")

    sources = {}


    for ticker, calendar_name in (("SPY", "NYSE"), ("^VIX", "CBOE_Index_Options")):
        if progress_callback:
            progress_callback(f"Prepareing data for {ticker}")

        if not ensure_stock_data(ticker, "1d"):
            raise ValueError(f"No market data for {ticker}")

        data = get_stock_data(ticker, interval="1d", years=None)

        if data.empty:
            raise ValueError(f"No market data in database for {ticker}")

        sources[ticker] = filter_completed_market_prices(data, ticker, calendar_name, snapshot_at)

    market_features = build_market_features(sources["SPY"], sources["^VIX"])
    return market_features


def connect_data(tickers, progress_callback=None, *, horizon=1, feature_columns=None, feature_groups=("technical",), news_start=None, news_end=None, prepare_news=True):
    datasets = []

    uses_news = "news" in feature_groups
    analyzer = None
    news_as_of = None

    if uses_news:
        if news_start is None or news_end is None:
            raise ValueError("News features require news_start and news_end")

        if prepare_news:
            analyzer = FinbertAnalyzer()
            news_as_of = pd.Timestamp.now(tz="UTC")

    market_features = None

    if "market" in feature_groups:
        for ticker in tickers:
            metadata = fetch_market_metadata(ticker)
            validate_market_metadata(ticker, metadata)
        market_features = load_market_features(progress_callback=progress_callback)

    for ticker in tickers:
        if not ensure_stock_data(ticker, "1d"):
            raise ValueError(f"Brak danych spółki {ticker}.")
        data = get_stock_data(ticker,years=None)
        if data.empty:
            raise ValueError(f"Brak danych spółki {ticker} w bazie.")

        ticker_news_start = news_start
        ticker_news_end = news_end

        if uses_news and prepare_news:
            news_result = ensure_news_data(
                ticker,
                news_start,
                news_end,
                analyzer=analyzer,
                as_of=news_as_of,
                progress_callback=progress_callback)

            ticker_news_start = news_result["start_date"]
            ticker_news_end = news_result["end_date"]

        data = prepare_dataset(data, horizon=horizon, feature_columns=feature_columns, feature_groups=feature_groups, ticker=ticker, news_start=ticker_news_start, news_end=ticker_news_end, market_features=market_features)
        if data.empty:
            raise ValueError(f"Za mało danych do przygotowania cech spółki {ticker}.")
        data["ticker"] = ticker
        datasets.append(data)
        if progress_callback:
            progress_callback(f"Przygotowano dane {ticker} ({len(datasets)}/{len(tickers)}).")

    combinedd = pd.concat(datasets, ignore_index=True)
    return combinedd


def validate_date(value, label):
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"{label}: podaj datę w formacie RRRR-MM-DD.")
    return value


def validate_training_cutoff(training_cutoff):
    validate_date(training_cutoff, "Data graniczna uczenia")

    if training_cutoff > date.today().isoformat():
        raise ValueError("Data graniczna uczenia nie może być w przyszłości.")

    return training_cutoff


def validate_simulation_dates(start_date, end_date):
    validate_date(start_date, "Początek symulacji")
    validate_date(end_date, "Koniec symulacji")
    if start_date >= end_date:
        raise ValueError("Data końcowa musi być późniejsza od początkowej.")



def split_dataset_by_date(data, start_date, end_date):
    train_df = data[
        (data["date"] < start_date) & (data["target_end_date"] <= start_date)
    ]
    test_df = data[
        (data["date"] >= start_date) & (data["target_end_date"] <= end_date)
    ]
    return train_df, test_df

def prepare_prediction_data(data, start_date, end_date, config, feature_columns, training_cutoff, *, ticker=None, prepare_news=True, progress_callback=None):
    validate_simulation_dates(start_date, end_date)
    validate_training_cutoff(training_cutoff)

    if data.empty:
        raise ValueError("Data cannot be empty.")

    if start_date < training_cutoff:
        raise ValueError("The evaluation of model must begin after experiencing it")

    news_start = start_date
    news_end = end_date

    if "news" in config.feature_groups:
        if not ticker:
            raise ValueError("News features require ticker")
        if prepare_news == True:
            news_info = ensure_news_data(ticker, news_start, news_end, progress_callback=progress_callback)
            news_start = news_info["start_date"]
            news_end = news_info["end_date"]

    market_features = None
    if "market" in config.feature_groups:
        if not ticker:
            raise ValueError("Market features require ticker")

        metadata = fetch_market_metadata(ticker)
        validate_market_metadata(ticker, metadata)
        market_features = load_market_features(progress_callback=progress_callback)

    prepared = prepare_dataset(
        data,
        horizon=config.horizon,
        feature_columns=feature_columns,
        feature_groups=config.feature_groups,
        ticker=ticker,
        news_start=news_start,
        news_end=news_end,
        market_features=market_features
        )

    _, test_df = split_dataset_by_date(prepared, start_date, end_date)

    if test_df.empty:
        raise ValueError("No data in the selected prediction period")

    return test_df

def prepare_latest_prediction_data(data,config,feature_columns,training_cutoff, *,ticker=None,as_of=None, prepare_news=True,progress_callback=None):
    if data.empty:
        raise ValueError("Data cannot be empty.")

    validate_training_cutoff(training_cutoff)
    uses_news = "news" in config.feature_groups
    uses_market = "market" in config.feature_groups

    if uses_news and not ticker:
        raise ValueError("News features require ticker")
    if uses_market:
        if not ticker:
            raise ValueError("Market features require ticker")

        metadata = fetch_market_metadata(ticker)
        validate_market_metadata(ticker, metadata)

    prediction_window = {}

    if uses_news or uses_market:
        latest_price_date = pd.to_datetime(data["date"]).max()
        prediction_window = get_latest_prediction_window(latest_price_date, config.feature_groups, as_of=as_of)

    news_start = None
    news_end = None

    if uses_news:
        news_start = prediction_window["previous_close"]
        news_end = prediction_window["as_of"]

        if prepare_news:
            news_result = ensure_news_data(ticker, news_start, news_end, refresh=True, as_of=prediction_window["as_of"],progress_callback=progress_callback)
            news_start = news_result["start_date"]
            news_end = news_result["end_date"]

    market_features = None

    if uses_market:
        market_features = load_market_features(progress_callback=progress_callback, as_of=prediction_window["as_of"])

    latest = prepare_latest_features(data, feature_columns, feature_groups=config.feature_groups, ticker=ticker, news_start=news_start, news_end=news_end, market_features=market_features)
    forecast_date = str(latest["date"].iloc[0])

    if forecast_date < training_cutoff:
        raise ValueError("Last session date is earlier than the training cutoff")

    return latest, prediction_window


def prepare_training_data(tickers, validation_start, training_cutoff, config, *, news_start=None, news_end=None, prepare_news=True):
    validate_training_cutoff(training_cutoff)
    validate_simulation_dates(validation_start, training_cutoff)

    if config.scope != "global":
        raise ValueError("Expected a global model configuration")

    if "news" in config.feature_groups:
        news_start, news_end = normalize_training_news_range(news_start,news_end,training_cutoff)

    feature_columns = get_feature_columns(config.feature_groups)
    data = connect_data(tickers, horizon=config.horizon, feature_columns=feature_columns, feature_groups=config.feature_groups,news_start=news_start, news_end=news_end, prepare_news=prepare_news)
    train_df, validation_df = split_dataset_by_date(data, validation_start, training_cutoff)

    if train_df.empty or validation_df.empty:
        raise ValueError("train_df and validation_df must not be empty")

    print(f"Tickers: {len(tickers)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    return data, train_df, validation_df