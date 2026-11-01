from datetime import date

import pandas as pd

from services.data_fetcher import fetch_market_metadata
from services.data_updater import ensure_stock_data
from services.db import get_stock_data, get_news_imports, get_news_without_sentiment, get_news_with_sentiment
from services.ml.dataset_service import prepare_dataset, prepare_latest_features
from services.ml.feature_schema import get_feature_columns
from services.ml.finbert_service import FinbertAnalyzer, MODEL_NAME, MODEL_REVISION
from services.ml.market_feature_service import build_market_features, filter_completed_market_prices, validate_market_metadata
from services.ml.news_feature_service import (NEWS_FEATURE_POLICY, prepare_news_dataframe, build_preopen_news_features,normalize_training_news_range, get_news_history_start)
from services.ml.prediction_time_service import build_market_schedule, get_latest_prediction_window
from services.ml.sector_feature_service import resolve_sector_benchmarks, build_sector_features
from services.news_fetcher import ensure_news_data, classify_news_relevance, find_missing_news_ranges


def load_context_prices(source_calendars, progress_callback=None, *, as_of=None):
    if not source_calendars:
        raise ValueError("Price sources cannot be empty")

    now = pd.Timestamp.now(tz="UTC")
    snapshot_at = now if as_of is None else pd.to_datetime(as_of, utc=True)
    if pd.isna(snapshot_at) or snapshot_at > now:
        raise ValueError("as_of must be a valid time not in the future")

    prices = {}
    for ticker, calendar_name in source_calendars.items():
        if progress_callback:
            progress_callback(f"Preparing data {ticker}.")
        if not ensure_stock_data(ticker, "1d"):
            raise ValueError(f"No price data for {ticker}")

        data = get_stock_data(ticker, interval="1d", years=None)
        if data.empty:
            raise ValueError(f"No price data in database for {ticker}")
        prices[ticker] = filter_completed_market_prices(data, ticker, calendar_name, snapshot_at)
    return prices


def load_context_data(tickers, feature_groups, *, sector_assignments=None, saved_benchmarks=None, as_of=None, progress_callback=None):
    tickers = list(tickers)
    uses_market = "market" in feature_groups
    uses_sector = "sector" in feature_groups
    sector_benchmarks = {}

    if uses_sector:
        sector_benchmarks = resolve_sector_benchmarks(tickers, sector_assignments, saved_benchmarks=saved_benchmarks)

    if uses_market or uses_sector:
        if not tickers or any(not ticker for ticker in tickers):
            raise ValueError("Context features require ticker")
        for ticker in tickers:
            validate_market_metadata(ticker, fetch_market_metadata(ticker))

    sources = {}
    if uses_market:
        sources.update({"SPY": "NYSE", "^VIX": "CBOE_Index_Options"})
    if uses_sector:
        sources.update({benchmark: "NYSE" for benchmark in sorted(set(sector_benchmarks.values()))})

    prices = load_context_prices(sources, progress_callback=progress_callback, as_of=as_of) if sources else {}
    market_features = build_market_features(prices["SPY"], prices["^VIX"]) if uses_market else None
    return market_features, sector_benchmarks, prices


def load_scored_news_range(ticker, start_date, end_date):
    ticker = ticker.upper()
    range_start = pd.to_datetime(start_date, utc=True)
    range_end = pd.to_datetime(end_date, utc=True)
    if pd.isna(range_start) or pd.isna(range_end):
        raise ValueError("Range dates cannot be missing")
    if range_start >= range_end:
        raise ValueError("Start date must be before end date")

    imports = get_news_imports(ticker, range_start, range_end)
    missing_ranges = find_missing_news_ranges(imports, range_start, range_end)
    if missing_ranges:
        first_missing = missing_ranges[0]
        raise ValueError(f"{ticker}: Missing news coverage between {first_missing['start_date']} and {first_missing['end_date']}")

    pending = get_news_without_sentiment(ticker, MODEL_NAME, MODEL_REVISION, start_date=range_start, end_date=range_end)
    for article in pending:
        if classify_news_relevance(article, ticker)["use_for_model"]:
            raise ValueError(f"{ticker}: First calculate the missing FinBERT scores")

    articles = get_news_with_sentiment(ticker, MODEL_NAME, MODEL_REVISION)
    articles = [article for article in articles if classify_news_relevance(article, ticker)["use_for_model"]]
    news_df = prepare_news_dataframe(articles)
    if not news_df.empty:
        news_df = news_df.loc[(news_df["available_at"] >= range_start) & (news_df["available_at"] < range_end)].copy()
    return news_df


def load_news_features(ticker, start_date, end_date, *, prepare_news=True, refresh=False,as_of=None, analyzer=None, progress_callback=None):
    now = pd.Timestamp.now(tz="UTC")
    snapshot_at = now if as_of is None else pd.to_datetime(as_of, utc=True)
    if pd.isna(snapshot_at) or snapshot_at > now:
        raise ValueError("as_of must be a valid time not in the future")

    range_start = pd.to_datetime(start_date, utc=True)
    range_end = pd.to_datetime(end_date, utc=True)
    minutes_before_open = NEWS_FEATURE_POLICY["minutes_before_open"]
    history_start = get_news_history_start(range_start, range_end, minutes_before_open)
    if prepare_news:
        news_result = ensure_news_data(
            ticker, range_start.isoformat(), range_end.isoformat(),
            analyzer=analyzer, refresh=refresh, as_of=snapshot_at,
            progress_callback=progress_callback, history_start=history_start)
        range_start = pd.to_datetime(news_result["start_date"], utc=True)
        range_end = pd.to_datetime(news_result["end_date"], utc=True)

    news = load_scored_news_range(ticker, history_start, range_end.isoformat())
    calendar_start = (pd.to_datetime(history_start, utc=True) - pd.DateOffset(days=14)).date().isoformat()
    calendar_end = (range_end + pd.DateOffset(days=14)).date().isoformat()
    schedule = build_market_schedule(calendar_start, calendar_end)
    return build_preopen_news_features(news, schedule, range_start, range_end, minutes_before_open=minutes_before_open, history_start=history_start)


def connect_data(tickers, progress_callback=None, *, horizon=1, feature_columns=None,
    feature_groups=("technical",), news_start=None, news_end=None,
    prepare_news=True, sector_assignments=None):

    tickers = list(tickers)
    if not tickers:
        raise ValueError("Tickers cannot be empty")

    snapshot_at = pd.Timestamp.now(tz="UTC")
    uses_news = "news" in feature_groups
    if uses_news and (news_start is None or news_end is None):
        raise ValueError("News features require news_start and news_end")
    analyzer = FinbertAnalyzer() if uses_news and prepare_news else None
    market_features, sector_benchmarks, prices = load_context_data(tickers, feature_groups, sector_assignments=sector_assignments, as_of=snapshot_at, progress_callback=progress_callback)

    datasets = []
    for ticker in tickers:
        if not ensure_stock_data(ticker, "1d"):
            raise ValueError(f"No data for ticker {ticker}.")
        data = get_stock_data(ticker, years=None)
        if data.empty:
            raise ValueError(f"No data for {ticker} in database.")

        news_features = None
        if uses_news:
            news_features = load_news_features(ticker, news_start, news_end, prepare_news=prepare_news, as_of=snapshot_at, analyzer=analyzer, progress_callback=progress_callback)

        sector_features = None
        if sector_benchmarks:
            benchmark = sector_benchmarks[ticker.strip().upper()]
            sector_features = build_sector_features(data, prices[benchmark])

        prepared = prepare_dataset(data, horizon=horizon, feature_columns=feature_columns, feature_groups=feature_groups,
            news_features=news_features, market_features=market_features, sector_features=sector_features)

        if prepared.empty:
            raise ValueError(f"No data for ticker {ticker}.")
        prepared["ticker"] = ticker
        if sector_benchmarks:
            prepared["sector_benchmark"] = benchmark
        datasets.append(prepared)
        if progress_callback:
            progress_callback(f"Przygotowano dane {ticker} ({len(datasets)}/{len(tickers)}).")

    return pd.concat(datasets, ignore_index=True)


def validate_date(value, label):
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"{label}: Invalid date format use RRRR-MM-DD.")
    return value


def validate_training_cutoff(training_cutoff):
    validate_date(training_cutoff, "Training cutoff")
    if training_cutoff > date.today().isoformat():
        raise ValueError("Training cutoff can not be in feature.")
    return training_cutoff


def validate_simulation_dates(start_date, end_date):
    validate_date(start_date, "Simulation start date")
    validate_date(end_date, "Simulation end date")
    if start_date >= end_date:
        raise ValueError("End date must be greater than start date.")


def select_training_rows(data, cutoff):
    return data.loc[(data["date"] < cutoff) & (data["target_end_date"] <= cutoff)].copy()


def split_dataset_by_date(data, start_date, end_date):
    train_df = select_training_rows(data, start_date)
    test_df = data.loc[(data["date"] >= start_date) & (data["target_end_date"] <= end_date)].copy()
    return train_df, test_df


def prepare_prediction_data(data, start_date, end_date, config, feature_columns, training_cutoff, *,
    ticker=None, prepare_news=True, progress_callback=None,
    sector_assignments=None, saved_benchmarks=None):

    validate_simulation_dates(start_date, end_date)
    validate_training_cutoff(training_cutoff)
    if data.empty:
        raise ValueError("Data cannot be empty.")
    if start_date < training_cutoff:
        raise ValueError("The evaluation of model must begin after experiencing it")
    if any(group in config.feature_groups for group in ("news", "market", "sector")) and not ticker:
        raise ValueError("Context features require ticker")

    snapshot_at = pd.Timestamp.now(tz="UTC")
    market_features, sector_benchmarks, prices = load_context_data( [ticker], config.feature_groups, sector_assignments=sector_assignments, saved_benchmarks=saved_benchmarks, as_of=snapshot_at, progress_callback=progress_callback)
    news_features = None
    if "news" in config.feature_groups:
        news_features = load_news_features(ticker, start_date, end_date, prepare_news=prepare_news, as_of=snapshot_at, progress_callback=progress_callback)

    sector_features = None
    if sector_benchmarks:
        benchmark = sector_benchmarks[ticker.strip().upper()]
        sector_features = build_sector_features(data, prices[benchmark])

    prepared = prepare_dataset(
        data, horizon=config.horizon, feature_columns=feature_columns, feature_groups=config.feature_groups,
        news_features=news_features, market_features=market_features, sector_features=sector_features)
    _, test_df = split_dataset_by_date(prepared, start_date, end_date)
    if test_df.empty:
        raise ValueError("No data in the selected prediction period")
    if sector_benchmarks:
        test_df["sector_benchmark"] = benchmark
    return test_df


def prepare_latest_prediction_data(data, config, feature_columns, training_cutoff, *, ticker=None, as_of=None,
    prepare_news=True, progress_callback=None, sector_assignments=None, saved_benchmarks=None):

    if data.empty:
        raise ValueError("Data cannot be empty")
    validate_training_cutoff(training_cutoff)

    uses_context = any(group in config.feature_groups for group in ("news", "market", "sector"))
    if uses_context and not ticker:
        raise ValueError("Context features require ticker")
    prediction_window = {}
    if uses_context:
        latest_price_date = pd.to_datetime(data["date"]).max()
        prediction_window = get_latest_prediction_window(latest_price_date, config.feature_groups, as_of=as_of)

    market_features, sector_benchmarks, prices = load_context_data(
        [ticker], config.feature_groups, sector_assignments=sector_assignments,
        saved_benchmarks=saved_benchmarks, as_of=prediction_window.get("as_of"), progress_callback=progress_callback)
    news_features = None
    if "news" in config.feature_groups:
        news_features = load_news_features(
            ticker, prediction_window["previous_close"], prediction_window["as_of"],
            prepare_news=prepare_news, refresh=True, as_of=prediction_window["as_of"],
            progress_callback=progress_callback)

    sector_features = None
    if sector_benchmarks:
        benchmark = sector_benchmarks[ticker.strip().upper()]
        sector_features = build_sector_features(data, prices[benchmark])

    latest = prepare_latest_features(
        data, feature_columns, feature_groups=config.feature_groups, news_features=news_features,
        market_features=market_features, sector_features=sector_features)
    if str(latest["date"].iloc[0]) < training_cutoff:
        raise ValueError("Last session date is earlier than the training cutoff")
    if sector_benchmarks:
        latest["sector_benchmark"] = benchmark
    return latest, prediction_window


def prepare_training_data(
    tickers, validation_start, training_cutoff, config, *, news_start=None,
    news_end=None, prepare_news=True, sector_assignments=None, progress_callback=None
):
    validate_training_cutoff(training_cutoff)
    tickers = list(tickers)
    if not tickers:
        raise ValueError("Tickers cannot be empty")
    if config.scope != "global":
        raise ValueError("Expected a global model configuration")
    if validation_start is not None:
        validate_simulation_dates(validation_start, training_cutoff)
    elif config.algorithm != "random_forest":
        raise ValueError("MLP and LightGBM require validation_start")

    if "news" in config.feature_groups:
        news_start, news_end = normalize_training_news_range(news_start, news_end, training_cutoff)
    feature_columns = get_feature_columns(config.feature_groups)
    data = connect_data(
        tickers, progress_callback=progress_callback, horizon=config.horizon,
        feature_columns=feature_columns, feature_groups=config.feature_groups,
        news_start=news_start, news_end=news_end, prepare_news=prepare_news,
        sector_assignments=sector_assignments
    )
    final_train_df = select_training_rows(data, training_cutoff)
    missing_tickers = sorted(set(tickers) - set(final_train_df["ticker"]))
    if missing_tickers:
        raise ValueError(f"No training data before choosen date: {missing_tickers}.")

    selection_train_df = None
    validation_df = None
    if validation_start is not None:
        selection_train_df, validation_df = split_dataset_by_date(final_train_df, validation_start, training_cutoff)
        if selection_train_df.empty or validation_df.empty:
            raise ValueError("train_df and validation_df must not be empty")

    return {
        "final_train_df": final_train_df,
        "selection_train_df": selection_train_df,
        "validation_df": validation_df,
        "feature_columns": feature_columns,
        "news_start": news_start,
        "news_end": news_end
    }
