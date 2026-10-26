from datetime import date

import pandas as pd

from services.data_updater import ensure_stock_data
from services.db import get_stock_data
from services.ml.dataset_service import prepare_dataset, prepare_latest_features
from services.ml.finbert_service import FinbertAnalyzer
from services.ml.news_feature_service import get_latest_news_window
from services.news_fetcher import ensure_news_data



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
                progress_callback=progress_callback,)

            ticker_news_start = news_result["start_date"]
            ticker_news_end = news_result["end_date"]

        data = prepare_dataset(data, horizon=horizon, feature_columns=feature_columns, feature_groups=feature_groups, ticker=ticker, news_start=ticker_news_start, news_end=ticker_news_end)
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

    prepared = prepare_dataset(
        data,
        horizon=config.horizon,
        feature_columns=feature_columns,
        feature_groups=config.feature_groups,
        ticker=ticker,
        news_start=news_start,
        news_end=news_end)

    _, test_df = split_dataset_by_date(prepared, start_date, end_date)

    if test_df.empty:
        raise ValueError("No data in the selected prediction period")

    return test_df

def prepare_latest_prediction_data(data,config,feature_columns,training_cutoff, *,ticker=None,as_of=None, prepare_news=True,progress_callback=None):
    if data.empty:
        raise ValueError("Data cannot be empty.")

    validate_training_cutoff(training_cutoff)

    news_window = {}
    if "news" in config.feature_groups:
        if not ticker:
            raise ValueError("News features require ticker")

        latest_price_date = pd.to_datetime(data["date"]).max()
        news_window = get_latest_news_window(latest_price_date, as_of=as_of)

        if prepare_news:
            news_result = ensure_news_data(ticker, news_window["news_start"], news_window["news_end"], refresh=True, as_of=news_window["news_end"], progress_callback=progress_callback)
            news_window["news_start"] = news_result["start_date"]
            news_window["news_end"] = news_result["end_date"]

    latest = prepare_latest_features(data, feature_columns, feature_groups=config.feature_groups, ticker=ticker, news_start=news_window.get("news_start"), news_end=news_window.get("news_end"))

    forecast_date = str(latest["date"].iloc[0])

    if forecast_date < training_cutoff:
        raise ValueError("Last session date is earlier than the training cutoff")

    return latest, news_window