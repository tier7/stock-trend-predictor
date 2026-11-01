import pandas as pd

from services.ml.data_preparation_service import connect_data, validate_simulation_dates, split_dataset_by_date, select_training_rows
from services.ml.dataset_service import FEATURE_COLUMNS, prepare_dataset
from services.ml.model_config import ModelConfig
from services.ml.model_service import DEFAULT_TRAINING_TICKERS, fit_model, predict_model
from services.ml.evaluation_service import evaluate
from services.ml.walk_forward_service import RETRAIN_INTERVAL, build_date_blocks
from services.simulator_service import DEFAULT_CONFIDENCE_THRESHOLD, validate_confidence_threshold, build_backtest_result


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

train_tickers = list(DEFAULT_TRAINING_TICKERS)

test_tickers = [
    "NVDA", "AMD", "JPM", "BAC", "PFE", "LLY", "XOM", "CVX", "VZ", "T"
]

def split_dataset_by_tickers(data, train_tickers, test_tickers):
    train_df = data[data["ticker"].isin(train_tickers)]
    test_df = data[data["ticker"].isin(test_tickers)]
    return train_df, test_df




def predict_global_walk_forward(train_df, test_df, progress_callback=None):
    if test_df.empty:
        raise ValueError("Brak danych do symulacji w wybranym okresie.")
    config = ModelConfig(algorithm="random_forest", task="classification")
    date_blocks = build_date_blocks(test_df, RETRAIN_INTERVAL)
    prediction_blocks = []
    training_blocks = []
    for block_dates in date_blocks:
        training_date = block_dates[0]
        current_train_df = select_training_rows(train_df, training_date)
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

        model, scaler, _ = fit_model(current_train_df, FEATURE_COLUMNS, config)
        current_test_df = predict_model(model, current_test_df, FEATURE_COLUMNS, config, scaler)
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

