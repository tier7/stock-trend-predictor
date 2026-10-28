from datetime import date, datetime, timezone

import sklearn

from services.db import get_stock_data
from services.ml.data_preparation_service import validate_training_cutoff, connect_data, validate_simulation_dates
from services.ml.dataset_service import FEATURE_COLUMNS, prepare_dataset, prepare_latest_features
from services.ml.model_service import build_backtest_result
from services.ml.model_store_service import load_global_model, save_global_model, get_model_config
from services.ml.walk_forward_service import (
    RETRAIN_INTERVAL,
    build_date_blocks,
    predict,
    train
)
from services.ml.evaluation_service import evaluate
from services.simulator_service import DEFAULT_CONFIDENCE_THRESHOLD, validate_confidence_threshold
from services.data_updater import ensure_stock_data
from services.ml.model_config import ModelConfig
from services.ml.feature_schema import get_feature_columns
from services.ml.news_feature_service import NEWS_FEATURE_POLICY, normalize_training_news_range
from services.news_fetcher import ensure_news_data
from services.ml.finbert_service import FinbertAnalyzer
from services.ml.data_preparation_service import split_dataset_by_date, prepare_prediction_data
from services.ml.data_preparation_service import prepare_latest_prediction_data
from services.ml.tree_prediction_service import (predict_with_global_model,simulate_with_global_model,predict_latest_with_global_model)
import pandas as pd
from services.ml.sector_feature_service import extract_sector_benchmarks

tickers = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "AMD", "INTC", "ORCL",
    "JPM", "BAC", "GS", "V", "MA", "BRK-B",
    "JNJ", "PFE", "LLY", "UNH", "ABBV",
    "XOM", "CVX", "SHEL", "NEE",
    "KO", "PEP", "MCD", "SBUX", "WMT",
    "COST", "PG", "NKE", "MC.PA",
    "DIS", "NFLX", "CMCSA", "VZ", "T", "TMUS",
    "BA", "AIR.PA", "LMT", "CAT", "DE",
    "UNP", "FDX", "UPS", "O", "AMT"
]

train_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "INTC", "ORCL", "GS", "V",
    "MA", "BRK-B", "JNJ", "UNH", "ABBV",
    "SHEL", "NEE", "KO", "PEP", "MCD",
    "SBUX", "WMT", "COST", "PG", "NKE",
    "MC.PA", "DIS", "NFLX", "CMCSA", "TMUS",
    "BA", "AIR.PA", "LMT", "CAT", "DE",
    "UNP", "FDX", "UPS", "O", "AMT"
]

test_tickers = [
    "NVDA", "AMD", "JPM", "BAC", "PFE", "LLY", "XOM", "CVX", "VZ", "T"
]

def split_dataset_by_tickers(data, train_tickers, test_tickers):
    train_df = data[data["ticker"].isin(train_tickers)]
    test_df = data[data["ticker"].isin(test_tickers)]
    return train_df, test_df


def train_global_model(training_cutoff, model_dir=None, progress_callback=None,*, config=None, news_start=None, news_end=None, training_tickers=None, prepare_news = True, sector_assignments=None):
    validate_training_cutoff(training_cutoff)
    if config is None:
        config = ModelConfig(
            algorithm="random_forest",
            task="classification",
        )

    if config.algorithm != "random_forest" or config.scope != "global":
        raise ValueError("Expected a global Random Forest configuration")

    if "news" in config.feature_groups:
        if training_tickers is None:
            raise ValueError(
                "News features require an explicit list of training tickers")

        news_start, news_end = normalize_training_news_range(news_start,news_end,training_cutoff)

    feature_columns = get_feature_columns(config.feature_groups)


    selected_tickers = list(train_tickers if training_tickers is None else training_tickers)

    if not selected_tickers:
        raise ValueError("Training tickers cannot be empty")

    data = connect_data(selected_tickers, progress_callback=progress_callback, horizon=config.horizon, feature_columns=feature_columns, feature_groups=config.feature_groups, news_start=news_start, news_end=news_end, prepare_news = prepare_news, sector_assignments=sector_assignments)
    train_df = data[
        (data["date"] < training_cutoff) & (data["target_end_date"] <= training_cutoff)
    ]
    missing_tickers = sorted(set(selected_tickers) - set(train_df["ticker"]))
    if missing_tickers:
        raise ValueError(f"Brak próbek treningowych przed wybraną datą: {missing_tickers}.")

    if progress_callback:
        progress_callback(f"Uczenie foresta na {len(train_df)} próbkach. To może potrwać kilka minut.")

    model = train(train_df, feature_columns, config=config)
    metadata = {
        "training_cutoff": training_cutoff,
        "training_rows": int(len(train_df)),
        "training_start_date": str(train_df["date"].min()),
        "training_end_date": str(train_df["date"].max()),
        "last_target_end_date": str(train_df["target_end_date"].max()),
        "train_tickers": selected_tickers,
        "feature_columns": list(feature_columns),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "n_estimators": model.n_estimators,
        "random_state": model.random_state,
        "config": config.to_dict(),
    }
    if "news" in config.feature_groups:
        metadata["news_policy"] = NEWS_FEATURE_POLICY.copy()
        metadata["news_start"] = news_start
        metadata["news_end"] = news_end
    if "sector" in config.feature_groups:
        metadata["sector_benchmarks"] = extract_sector_benchmarks(train_df)
    if config.task == "classification":
        metadata["baseline_prediction"] = int(
            train_df["target"].mode().iloc[0])
        metadata["target_definition"] = (
            f"open(t+{config.horizon + 1}) > open(t+1)")
    else:
        metadata["baseline_return"] = float(
            train_df["target_return"].mean())
        metadata["target_definition"] = (
            f"open(t+{config.horizon + 1}) / open(t+1) - 1")

    if progress_callback:
        progress_callback("Zapisywanie wytrenowanego modelu.")
    return save_global_model(model, metadata, model_dir)


def predict_global_walk_forward(train_df, test_df, progress_callback=None):
    if test_df.empty:
        raise ValueError("Brak danych do symulacji w wybranym okresie.")
    date_blocks = build_date_blocks(test_df, RETRAIN_INTERVAL)
    prediction_blocks = []
    training_blocks = []
    for block_dates in date_blocks:
        training_date = block_dates[0]
        current_train_df = train_df[
            (train_df["date"] < training_date)
            & (train_df["target_end_date"] <= training_date)
        ]
        if len(current_train_df) < 100:
            raise ValueError("Za mało danych treningowych. Wybierz późniejszy początek symulacji.")
        current_test_df = test_df[test_df["date"].isin(block_dates)].copy()
        block = {
            "start_date": training_date,
            "end_date": block_dates[-1],
            "training_rows": int(len(current_train_df)),
            "test_rows": int(len(current_test_df)),
        }
        if progress_callback:
            progress_callback(block)

        model = train(current_train_df, FEATURE_COLUMNS)
        current_test_df = predict(model, current_test_df, FEATURE_COLUMNS)
        current_test_df["baseline_prediction"] = int(current_train_df["target"].mode().iloc[0])
        prediction_blocks.append(current_test_df)
        training_blocks.append(block)
    results_df = pd.concat(prediction_blocks, ignore_index=True).sort_values(["date", "ticker"])
    return results_df, training_blocks


def simulate_global_walk_forward(data, ticker, start_date, end_date, initial_cash, fee_rate,
                                 *, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    confidence_threshold = validate_confidence_threshold(confidence_threshold)
    validate_simulation_dates(start_date, end_date)
    ticker = ticker.upper()
    prepared = prepare_dataset(data)
    prepared["ticker"] = ticker
    _, test_df = split_dataset_by_date(prepared, start_date, end_date)
    if test_df.empty:
        raise ValueError("Brak danych do symulacji w wybranym okresie.")

    # Selected stocks outside train_tickers remain unseen throughout the simulation.
    train_df = connect_data(train_tickers)
    results_df, training_blocks = predict_global_walk_forward(train_df, test_df)
    result = build_backtest_result(
        results_df, start_date, end_date, initial_cash, fee_rate,
        training_blocks[0]["training_rows"],
        confidence_threshold=confidence_threshold,
    )
    result.update({
        "model_type": "global_walk_forward",
        "retrain_interval": RETRAIN_INTERVAL,
        "training_runs": len(training_blocks),
        "training_blocks": training_blocks,
        "train_tickers": list(train_tickers),
        "ticker_seen_during_training": ticker in train_tickers,
    })
    return result


def run_experiment(start_date, end_date):
    validate_simulation_dates(start_date, end_date)

    data = connect_data(tickers)
    train_df, test_df = split_dataset_by_tickers(data, train_tickers, test_tickers)
    train_period_df, seen_test_df = split_dataset_by_date(train_df, start_date, end_date)
    _, unseen_test_df = split_dataset_by_date(test_df, start_date, end_date)
    test_period_df = pd.concat([seen_test_df, unseen_test_df],ignore_index=True)

    if test_period_df.empty:
        raise ValueError("No test data in the selected period.")


    print("Train rows:", len(train_period_df))
    print("Seen test rows:", len(seen_test_df))
    print("Unseen test rows:", len(unseen_test_df))


    def print_block(block):
        print(block["start_date"])
        print(block["end_date"])
        print(block["training_rows"])
        print(block["test_rows"])

    results_df, _ = predict_global_walk_forward(train_df, test_period_df, print_block)
    seen_test_df, unseen_test_df = split_dataset_by_tickers(results_df, train_tickers, test_tickers)
    return seen_test_df, unseen_test_df

if __name__ == "__main__":
    seen_test_df, unseen_test_df = run_experiment("2025-01-01", "2025-12-31")
    seen_model_eval = evaluate(seen_test_df)
    unseen_model_eval = evaluate(unseen_test_df)

    print("Seen accuracy: {:.2%}".format(seen_model_eval["accuracy"]))
    print("Precision: {:.2%}".format(seen_model_eval["precision"]))
    print("Recall: {:.2%}".format(seen_model_eval["recall"]))
    print("F1: {:.2%}".format(seen_model_eval["f1"]))
    print("Seen confusion matrix:", seen_model_eval["confusion_matrix"])

    print("Unseen accuracy: {:.2%}".format(unseen_model_eval["accuracy"]))
    print("Unseen precision: {:.2%}".format(unseen_model_eval["precision"]))
    print("Unseen recall: {:.2%}".format(unseen_model_eval["recall"]))
    print("Unseen F1: {:.2%}".format(unseen_model_eval["f1"]))
    print("Unseen confusion matrix:", unseen_model_eval["confusion_matrix"])

    print(seen_test_df["target"].value_counts(normalize=True))
    print(unseen_test_df["target"].value_counts(normalize=True))
