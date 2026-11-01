import pandas as pd

from services.ml.data_preparation_service import connect_data, select_training_rows, split_dataset_by_date, validate_simulation_dates
from services.ml.dataset_service import FEATURE_COLUMNS, prepare_dataset
from services.ml.model_config import ModelConfig
from services.ml.model_service import DEFAULT_TRAINING_TICKERS, fit_model, predict_model
from services.simulator_service import DEFAULT_CONFIDENCE_THRESHOLD, validate_confidence_threshold, build_backtest_result

RETRAIN_INTERVAL = 20

def build_date_blocks(test_df, retrain_interval):
    if retrain_interval <= 0:
        raise ValueError("Retrain interval must be positive")
    dates = test_df["date"].drop_duplicates().sort_values().tolist()
    return [dates[position:position + retrain_interval] for position in range(0, len(dates), retrain_interval)]


def run_walk_forward(train_df, test_df, feature_columns, config, *, date_blocks=None,retrain_interval=RETRAIN_INTERVAL, progress_callback=None):
    if config.algorithm != "random_forest" or config.task != "classification" or config.horizon != 1:
        raise ValueError("Walk-forward supports Random Forest classification with horizon=1 only")
    if test_df.empty:
        raise ValueError("No data to simulate in choosen period.")
    if date_blocks is None:
        date_blocks = build_date_blocks(test_df, retrain_interval)
    if not date_blocks or any(not block_dates for block_dates in date_blocks):
        raise ValueError("Date blocks must contain non-empty blocks.")
    scheduled_dates = [date for block_dates in date_blocks for date in block_dates]
    if scheduled_dates != sorted(set(scheduled_dates)):
        raise ValueError("Date blocks must contain unique dates in chronological order.")
    if not test_df["date"].isin(scheduled_dates).all():
        raise ValueError("Date blocks do not cover all test dates.")

    prediction_blocks = []
    training_blocks = []
    for block_dates in date_blocks:
        current_test_df = test_df[test_df["date"].isin(block_dates)].copy()
        if current_test_df.empty:
            continue
        training_date = block_dates[0]
        current_train_df = select_training_rows(train_df, training_date)
        if config.scope == "global" and len(current_train_df) < 100:
            raise ValueError("Not enough train data. Choose later date.")
        block = {
            "start_date": training_date,
            "end_date": block_dates[-1],
            "training_rows": int(len(current_train_df)),
            "test_rows": int(len(current_test_df))}
        if progress_callback:
            progress_callback(block)
        model, scaler, _ = fit_model(current_train_df, feature_columns, config)
        current_test_df = predict_model(model, current_test_df, feature_columns, config, scaler)
        current_test_df["baseline_prediction"] = int(current_train_df["target"].mode().iloc[0])
        prediction_blocks.append(current_test_df)
        training_blocks.append(block)
    results = pd.concat(prediction_blocks, ignore_index=True)
    sort_columns = ["date", "ticker"] if "ticker" in results.columns else ["date"]
    return results.sort_values(sort_columns), training_blocks


def train_and_predict(data, start_date, end_date, initial_cash, fee_rate, date_blocks=None, *, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    confidence_threshold = validate_confidence_threshold(confidence_threshold)
    if start_date >= end_date:
        raise ValueError("End date must be later than start date.")
    prepared = prepare_dataset(data)
    train_period_df, test_df = split_dataset_by_date(prepared, start_date, end_date)
    if len(train_period_df) < 100:
        raise ValueError("Not enough training data before selected date. Choose a later date.")
    if test_df.empty:
        raise ValueError("No test data from selected date. Choose an earlier date.")
    config = ModelConfig(algorithm="random_forest", task="classification", scope="local")
    results, _ = run_walk_forward(
        prepared, test_df, FEATURE_COLUMNS, config,
        date_blocks=date_blocks, retrain_interval=RETRAIN_INTERVAL)
    return build_backtest_result(
        results, start_date, end_date, initial_cash, fee_rate, len(train_period_df),
        confidence_threshold=confidence_threshold)


def simulate_global_walk_forward(data, ticker, start_date, end_date, initial_cash, fee_rate, *,confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    confidence_threshold = validate_confidence_threshold(confidence_threshold)
    validate_simulation_dates(start_date, end_date)
    ticker = ticker.upper()
    prepared = prepare_dataset(data)
    prepared["ticker"] = ticker
    _, test_df = split_dataset_by_date(prepared, start_date, end_date)
    if test_df.empty:
        raise ValueError("Brak danych do symulacji w wybranym okresie.")
    train_tickers = list(DEFAULT_TRAINING_TICKERS)
    train_df = connect_data(train_tickers)
    config = ModelConfig(algorithm="random_forest", task="classification")
    results, training_blocks = run_walk_forward(train_df, test_df, FEATURE_COLUMNS, config, retrain_interval=RETRAIN_INTERVAL)
    result = build_backtest_result(results, start_date, end_date, initial_cash, fee_rate, training_blocks[0]["training_rows"],confidence_threshold=confidence_threshold)
    result.update({
        "model_type": "global_walk_forward",
        "retrain_interval": RETRAIN_INTERVAL,
        "training_runs": len(training_blocks),
        "training_blocks": training_blocks,
        "train_tickers": train_tickers,
        "ticker_seen_during_training": ticker in train_tickers})

    return result