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
from services.ml.sector_feature_service import (resolve_sector_benchmarks,build_sector_features)

def load_context_prices(source_calendars, progress_callback=None, *, as_of=None,):
    if not source_calendars:
        raise ValueError("Price sources cannot be empty")

    now = pd.Timestamp.now(tz="UTC")
    snapshot_at = (now if as_of is None else pd.to_datetime(as_of, utc=True))

    if pd.isna(snapshot_at) or snapshot_at > now:
        raise ValueError("as_of must be a valid time not in the future")

    prices = {}

    for ticker, calendar_name in source_calendars.items():
        if progress_callback:
            progress_callback(f"Przygotowywanie danych {ticker}.")

        if not ensure_stock_data(ticker, "1d"):
            raise ValueError(f"No price data for {ticker}")

        data = get_stock_data(
            ticker,
            interval="1d",
            years=None)

        if data.empty:
            raise ValueError(f"No price data in database for {ticker}")

        prices[ticker] = filter_completed_market_prices(data,ticker,calendar_name,snapshot_at)
    return prices

def connect_data(tickers, progress_callback=None, *, horizon=1,feature_columns=None, feature_groups=("technical",),news_start=None, news_end=None, prepare_news=True,sector_assignments=None,):
    tickers = list(tickers)

    if not tickers:
        raise ValueError("Tickers cannot be empty")

    datasets = []

    snapshot_at = pd.Timestamp.now(tz="UTC")
    uses_news = "news" in feature_groups
    uses_market = "market" in feature_groups
    uses_sector = "sector" in feature_groups
    analyzer = None

    if uses_news:
        if news_start is None or news_end is None:
            raise ValueError("News features require news_start and news_end")

        if prepare_news:
            analyzer = FinbertAnalyzer()

    sector_benchmarks = {}

    if uses_sector:
        sector_benchmarks = resolve_sector_benchmarks(tickers,sector_assignments)

    if uses_market or uses_sector:
        for ticker in tickers:
            metadata = fetch_market_metadata(ticker)
            validate_market_metadata(ticker, metadata)

    context_sources = {}
    if uses_market:
        context_sources.update({"SPY": "NYSE","^VIX": "CBOE_Index_Options"})

    if uses_sector:
        context_sources.update({benchmark: "NYSE" for benchmark in sorted(set(sector_benchmarks.values()))})

    context_prices = {}

    if context_sources:
        context_prices = load_context_prices(context_sources,progress_callback=progress_callback,as_of=snapshot_at)

    market_features = None
    if uses_market:
        market_features = build_market_features(context_prices["SPY"],context_prices["^VIX"])

    for ticker in tickers:
        if not ensure_stock_data(ticker, "1d"):
            raise ValueError(f"Brak danych spółki {ticker}.")

        data = get_stock_data(ticker, years=None)

        if data.empty:
            raise ValueError(f"Brak danych spółki {ticker} w bazie.")

        ticker_news_start = news_start
        ticker_news_end = news_end

        if uses_news and prepare_news:
            news_result = ensure_news_data(ticker,news_start, news_end,analyzer=analyzer,as_of=snapshot_at,progress_callback=progress_callback)
            ticker_news_start = news_result["start_date"]
            ticker_news_end = news_result["end_date"]

        sector_features = None

        if uses_sector:
            benchmark_ticker = sector_benchmarks[ticker.strip().upper()]
            sector_features = build_sector_features(data,context_prices[benchmark_ticker])

        data = prepare_dataset(data,horizon=horizon,feature_columns=feature_columns,feature_groups=feature_groups,ticker=ticker,news_start=ticker_news_start,news_end=ticker_news_end,market_features=market_features,sector_features=sector_features)

        if data.empty:
            raise ValueError(
                f"No data for ticker {ticker}.")

        data["ticker"] = ticker

        if uses_sector:
            data["sector_benchmark"] = benchmark_ticker
        datasets.append(data)

        if progress_callback:
            progress_callback(f"Przygotowano dane {ticker} "f"({len(datasets)}/{len(tickers)}).")

    return pd.concat(datasets, ignore_index=True)


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

def prepare_prediction_data(data, start_date, end_date, config, feature_columns, training_cutoff, *, ticker=None, prepare_news=True, progress_callback=None, sector_assignments=None, saved_benchmarks=None):
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

    uses_market = "market" in config.feature_groups
    uses_sector = "sector" in config.feature_groups

    context_sources = {}
    benchmark_ticker = None

    if uses_market or uses_sector:
        if not ticker:
            raise ValueError("Context features require ticker")

        market_metadata = fetch_market_metadata(ticker)
        validate_market_metadata(ticker, market_metadata)

    if uses_market:
        context_sources.update({"SPY": "NYSE","^VIX": "CBOE_Index_Options",})

    if uses_sector:
        benchmarks = resolve_sector_benchmarks([ticker],sector_assignments,saved_benchmarks=saved_benchmarks)
        benchmark_ticker = benchmarks[ticker.strip().upper()]
        context_sources[benchmark_ticker] = "NYSE"

    context_prices = {}

    if context_sources:
        context_prices = load_context_prices(context_sources,progress_callback=progress_callback)

    market_features = None
    sector_features = None

    if uses_market:
        market_features = build_market_features(context_prices["SPY"],context_prices["^VIX"])

    if uses_sector:
        sector_features = build_sector_features(data,context_prices[benchmark_ticker])

    prepared = prepare_dataset(
        data,
        horizon=config.horizon,
        feature_columns=feature_columns,
        feature_groups=config.feature_groups,
        ticker=ticker,
        news_start=news_start,
        news_end=news_end,
        market_features=market_features,
        sector_features=sector_features
        )

    _, test_df = split_dataset_by_date(prepared, start_date, end_date)

    if test_df.empty:
        raise ValueError("No data in the selected prediction period")

    if uses_sector:
        test_df = test_df.copy()
        test_df["sector_benchmark"] = benchmark_ticker

    return test_df

def prepare_latest_prediction_data(data, config, feature_columns, training_cutoff, *, ticker=None, as_of=None, prepare_news=True, progress_callback=None, sector_assignments=None, saved_benchmarks=None):
    if data.empty:
        raise ValueError("Data cannot be empty")

    validate_training_cutoff(training_cutoff)

    uses_news = "news" in config.feature_groups
    uses_market = "market" in config.feature_groups
    uses_sector = "sector" in config.feature_groups

    if (uses_news or uses_market or uses_sector) and not ticker:
        raise ValueError("Context features require ticker")

    if uses_market or uses_sector:
        market_metadata = fetch_market_metadata(ticker)
        validate_market_metadata(ticker, market_metadata)

    context_sources = {}
    benchmark_ticker = None

    if uses_market:
        context_sources.update({"SPY": "NYSE", "^VIX": "CBOE_Index_Options"})

    if uses_sector:
        benchmarks = resolve_sector_benchmarks([ticker], sector_assignments, saved_benchmarks=saved_benchmarks)
        benchmark_ticker = benchmarks[ticker.strip().upper()]
        context_sources[benchmark_ticker] = "NYSE"

    prediction_window = {}

    if uses_news or uses_market or uses_sector:
        latest_price_date = pd.to_datetime(data["date"]).max()
        prediction_window = get_latest_prediction_window(latest_price_date, config.feature_groups, as_of=as_of)

    news_start = None
    news_end = None

    if uses_news:
        news_start = prediction_window["previous_close"]
        news_end = prediction_window["as_of"]

        if prepare_news:
            news_result = ensure_news_data(ticker, news_start, news_end, refresh=True, as_of=prediction_window["as_of"], progress_callback=progress_callback)
            news_start = news_result["start_date"]
            news_end = news_result["end_date"]

    context_prices = {}

    if context_sources:
        context_prices = load_context_prices(context_sources, progress_callback=progress_callback, as_of=prediction_window["as_of"])

    market_features = None
    sector_features = None

    if uses_market:
        market_features = build_market_features(context_prices["SPY"], context_prices["^VIX"])

    if uses_sector:
        sector_features = build_sector_features(data, context_prices[benchmark_ticker])

    latest = prepare_latest_features(data, feature_columns, feature_groups=config.feature_groups, ticker=ticker, news_start=news_start, news_end=news_end, market_features=market_features, sector_features=sector_features)
    forecast_date = str(latest["date"].iloc[0])

    if forecast_date < training_cutoff:
        raise ValueError("Last session date is earlier than the training cutoff")

    if uses_sector:
        latest["sector_benchmark"] = benchmark_ticker

    return latest, prediction_window


def prepare_training_data(tickers, validation_start, training_cutoff, config, *, news_start=None, news_end=None, prepare_news=True, sector_assignments=None):
    validate_training_cutoff(training_cutoff)
    validate_simulation_dates(validation_start, training_cutoff)

    if config.scope != "global":
        raise ValueError("Expected a global model configuration")

    if "news" in config.feature_groups:
        news_start, news_end = normalize_training_news_range(news_start,news_end,training_cutoff)

    feature_columns = get_feature_columns(config.feature_groups)
    data = connect_data(tickers, horizon=config.horizon, feature_columns=feature_columns, feature_groups=config.feature_groups,news_start=news_start, news_end=news_end, prepare_news=prepare_news, sector_assignments=sector_assignments)
    train_df, validation_df = split_dataset_by_date(data, validation_start, training_cutoff)

    if train_df.empty or validation_df.empty:
        raise ValueError("train_df and validation_df must not be empty")

    print(f"Tickers: {len(tickers)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    return data, train_df, validation_df
