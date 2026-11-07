from math import isfinite

import pandas as pd

from services.ml.data_preparation_service import connect_data, normalize_evaluation_tickers, select_training_rows, split_dataset_by_date, validate_simulation_dates
from services.ml.dataset_service import prepare_dataset
from services.ml.evaluation_service import predict_baseline
from services.ml.feature_schema import get_feature_columns
from services.ml.model_config import ModelConfig
from services.ml.model_service import DEFAULT_TRAINING_TICKERS, fit_prepared_model, predict_model, resolve_training_parameters
from services.ml.news_feature_service import normalize_training_news_range
from services.simulator_service import build_backtest_result
from services.strategy_service import StrategyConfig
from services.ml.baseline_service import (fit_feature_baseline,predict_feature_baseline)

RETRAIN_INTERVAL = 20

def build_date_blocks(test_df, retrain_interval):
    if type(retrain_interval) is not int or retrain_interval <= 0:
        raise ValueError("Retrain interval must be a positive integer")
    dates = test_df["date"].drop_duplicates().sort_values().tolist()
    return [dates[position:position + retrain_interval] for position in range(0, len(dates), retrain_interval)]


def _resolve_walk_forward_parameters(config, retrain_interval, validation_fraction, training_parameters):
    if not isinstance(config, ModelConfig):
        raise ValueError("Expected ModelConfig")
    if type(retrain_interval) is not int or retrain_interval < 1:
        raise ValueError("Retrain interval must be a positive integer")
    if isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)) or not isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be finite and between 0 and 1")
    if training_parameters is not None and not isinstance(training_parameters, dict):
        raise ValueError("Training parameters must be a dictionary")
    try:
        return resolve_training_parameters(config, **(training_parameters or {}))
    except TypeError as error:
        raise ValueError(f"Invalid training parameters: {error}") from error


def run_walk_forward(train_df, test_df, feature_columns, config, *, date_blocks=None,
    retrain_interval=RETRAIN_INTERVAL, validation_fraction=0.2,training_parameters=None, progress_callback=None):

    parameters = _resolve_walk_forward_parameters(config, retrain_interval, validation_fraction, training_parameters)
    if train_df.empty or test_df.empty:
        raise ValueError("Walk-forward requires non-empty training and test data")
    frames = []
    for data in (train_df, test_df):
        frame = data.copy()
        for column in ("date", "target_end_date"):
            frame[column] = pd.to_datetime(frame[column], errors="raise")
            if frame[column].isna().any():
                raise ValueError("Missing walk-forward dates")
            frame[column] = frame[column].dt.strftime("%Y-%m-%d")
        identifiers = ["ticker", "date"] if "ticker" in frame else ["date"]
        if frame.duplicated(identifiers).any():
            raise ValueError("Duplicate company sessions in walk-forward data")
        if config.scope == "local" and "ticker" in frame and frame["ticker"].nunique() > 1:
            raise ValueError("Local walk-forward requires one ticker")
        frames.append(frame.sort_values(identifiers).reset_index(drop=True))
    train_df, test_df = frames
    if config.scope == "local" and "ticker" in train_df and "ticker" in test_df:
        if set(train_df["ticker"]) != set(test_df["ticker"]):
            raise ValueError("Local training and test tickers must match")
    if date_blocks is None:
        date_blocks = build_date_blocks(test_df, retrain_interval)
    if not date_blocks or any(not block_dates for block_dates in date_blocks):
        raise ValueError("Date blocks must contain non-empty blocks.")
    normalized_blocks = []
    for block_dates in date_blocks:
        if isinstance(block_dates, (str, bytes)):
            raise ValueError("Each date block must be a collection of session dates")
        dates = pd.to_datetime(pd.Series(block_dates), errors="raise")
        if dates.isna().any():
            raise ValueError("Missing date in walk-forward blocks")
        normalized_blocks.append(dates.dt.strftime("%Y-%m-%d").tolist())
    date_blocks = normalized_blocks
    scheduled_dates = [date for block_dates in date_blocks for date in block_dates]

    if scheduled_dates != sorted(set(scheduled_dates)):
        raise ValueError("Date blocks must contain unique dates in chronological order.")
    if not test_df["date"].isin(scheduled_dates).all():
        raise ValueError("Date blocks do not cover all test dates.")

    prediction_blocks = []
    training_blocks = []
    for block_index, block_dates in enumerate(date_blocks, start=1):
        current_test_df = test_df[test_df["date"].isin(block_dates)].copy()
        if current_test_df.empty:
            continue
        training_date = block_dates[0]
        current_train_df = select_training_rows(train_df, training_date)

        if current_train_df.empty or (config.scope == "global" and len(current_train_df) < 100):
            raise ValueError(f"Not enough mature training history before {training_date}")

        validation_start = None
        selection_train_df = None
        validation_df = None

        if config.algorithm != "random_forest":
            training_dates = current_train_df["date"].drop_duplicates().sort_values().tolist()
            if len(training_dates) < 3:
                raise ValueError(f"Not enough dates for validation before {training_date}")
            position = max(1, min(len(training_dates) - 1, int(len(training_dates) * (1 - validation_fraction))))
            validation_start = training_dates[position]
            selection_train_df, validation_df = split_dataset_by_date(current_train_df, validation_start, training_date)
            if selection_train_df.empty or validation_df.empty:
                raise ValueError(f"Not enough mature data for selection and validation before {training_date}")
        training_data = {
            "final_train_df": current_train_df,
            "selection_train_df": selection_train_df,
            "validation_df": validation_df,
            "feature_columns": feature_columns
        }

        block = {
            "block": block_index,
            "algorithm": config.algorithm,
            "task": config.task,
            "horizon": config.horizon,
            "start_date": training_date,
            "end_date": block_dates[-1],
            "first_training_date": str(current_train_df["date"].min()),
            "last_training_date": str(current_train_df["date"].max()),
            "last_target_end_date": str(current_train_df["target_end_date"].max()),
            "training_rows": int(len(current_train_df)),
            "test_rows": int(len(current_test_df)),
            "validation_start": validation_start,
            "selection_training_rows": 0 if selection_train_df is None else int(len(selection_train_df)),
            "validation_rows": 0 if validation_df is None else int(len(validation_df)),
            "train_tickers": sorted(current_train_df["ticker"].unique().tolist()) if "ticker" in current_train_df else []
        }
        if progress_callback:
            progress_callback(block.copy())
        model, scaler, details = fit_prepared_model(
            training_data, config, parameters, validation_start=validation_start)
        current_test_df = predict_model(model, current_test_df, feature_columns, config, scaler)
        baseline = predict_baseline(current_train_df, current_test_df, config)
        linear_baseline_model = fit_feature_baseline(current_train_df, feature_columns, config)
        linear_baseline = predict_feature_baseline(linear_baseline_model,current_test_df,feature_columns,config)

        prediction_columns = (["prediction", "probability_up"]if config.task == "classification" else ["predicted_return"])

        for prefix, frame in (("baseline", baseline),("linear_baseline", linear_baseline)):
            for column in prediction_columns:
                current_test_df[f"{prefix}_{column}"] = (frame[column].to_numpy())


        current_test_df["walk_forward_block"] = block_index
        block["training_details"] = details
        prediction_blocks.append(current_test_df)
        training_blocks.append(block)
    results = pd.concat(prediction_blocks, ignore_index=True)
    sort_columns = ["date", "ticker"] if "ticker" in results.columns else ["date"]
    return results.sort_values(sort_columns).reset_index(drop=True), training_blocks


def train_and_predict(
    data, start_date, end_date, initial_cash=10000, *, strategy_config=None,
    date_blocks=None, config=None, retrain_interval=RETRAIN_INTERVAL,
    validation_fraction=0.2, training_parameters=None, ticker=None,
    news_start=None, prepare_news=True, sector_assignments=None, progress_callback=None):
    if config is None:
        config = ModelConfig(algorithm="random_forest", task="classification", scope="local")
    parameters = _resolve_walk_forward_parameters(config, retrain_interval, validation_fraction, training_parameters)
    if config.scope != "local":
        raise ValueError("Expected a local walk-forward configuration")
    if strategy_config is None:
        strategy_config = StrategyConfig()
    if not isinstance(strategy_config, StrategyConfig):
        raise ValueError("Expected StrategyConfig")

    validate_simulation_dates(start_date, end_date)
    if data.empty:
        raise ValueError("Empty stock data")
    feature_columns = get_feature_columns(config.feature_groups)
    if any(group in config.feature_groups for group in ("news", "market", "sector")):
        if not ticker:
            raise ValueError("Context features require ticker")
        ticker = ticker.strip().upper()
        news_end = None
        if "news" in config.feature_groups:
            news_start, news_end = normalize_training_news_range(news_start, end_date, end_date)
        prepared = connect_data(
            [ticker], horizon=config.horizon, feature_columns=feature_columns,
            feature_groups=config.feature_groups, news_start=news_start, news_end=news_end,
            prepare_news=prepare_news, sector_assignments=sector_assignments, stock_data={ticker: data})
    else:
        prepared = prepare_dataset(data, horizon=config.horizon, feature_columns=feature_columns, feature_groups=config.feature_groups)
        if ticker:
            prepared["ticker"] = ticker.strip().upper()
    train_period_df, test_df = split_dataset_by_date(prepared, start_date, end_date)
    if len(train_period_df) < 100:
        raise ValueError("Not enough training data before selected date. Choose a later date.")
    if test_df.empty:
        raise ValueError("No test data from selected date. Choose an earlier date.")
    results, training_blocks = run_walk_forward(
        prepared, test_df, feature_columns, config, date_blocks=date_blocks,
        retrain_interval=retrain_interval, validation_fraction=validation_fraction,
        training_parameters=parameters, progress_callback=progress_callback)
    result = build_backtest_result(results, data, config, strategy_config, start_date, end_date, initial_cash, training_blocks[0]["training_rows"])
    result.update({
        "model_type": "local_walk_forward",
        "model_config": config.to_dict(),
        "retrain_interval": retrain_interval,
        "validation_fraction": validation_fraction,
        "training_parameters": parameters,
        "training_runs": len(training_blocks),
        "training_blocks": training_blocks
    })
    return result



def simulate_global_walk_forward(
    data, ticker, start_date, end_date, initial_cash=10000, *, strategy_config=None,
    config=None, training_tickers=None, date_blocks=None, retrain_interval=RETRAIN_INTERVAL,
    validation_fraction=0.2, training_parameters=None, news_start=None,
    prepare_news=True, sector_assignments=None, progress_callback=None):

    if config is None:
        config = ModelConfig(algorithm="random_forest", task="classification")
    parameters = _resolve_walk_forward_parameters(config, retrain_interval, validation_fraction, training_parameters)
    if config.scope != "global":
        raise ValueError("Expected a global walk-forward configuration")
    if strategy_config is None:
        strategy_config = StrategyConfig()
    if not isinstance(strategy_config, StrategyConfig):
        raise ValueError("Expected StrategyConfig")

    validate_simulation_dates(start_date, end_date)
    if data.empty:
        raise ValueError("Empty stock data")
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty")
    if not pd.to_datetime(data["date"], errors="raise").between(pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="left").any():
        raise ValueError("No stock sessions in the selected period")
    train_tickers, _ = normalize_evaluation_tickers(DEFAULT_TRAINING_TICKERS if training_tickers is None else training_tickers)
    tickers = train_tickers + ([] if ticker in train_tickers else [ticker])
    feature_columns = get_feature_columns(config.feature_groups)
    news_end = None
    if "news" in config.feature_groups:
        news_start, news_end = normalize_training_news_range(news_start, end_date, end_date)
    prepared = connect_data(
        tickers, horizon=config.horizon, feature_columns=feature_columns,
        feature_groups=config.feature_groups, news_start=news_start, news_end=news_end,
        prepare_news=prepare_news, sector_assignments=sector_assignments, stock_data={ticker: data})
    train_df = prepared.loc[prepared["ticker"].isin(train_tickers)].copy()
    _, test_df = split_dataset_by_date(prepared.loc[prepared["ticker"] == ticker], start_date, end_date)
    if test_df.empty:
        raise ValueError("Brak danych do symulacji w wybranym okresie.")
    results, training_blocks = run_walk_forward(
        train_df, test_df, feature_columns, config, date_blocks=date_blocks,
        retrain_interval=retrain_interval, validation_fraction=validation_fraction,
        training_parameters=parameters, progress_callback=progress_callback)
    result = build_backtest_result(results, data, config, strategy_config, start_date, end_date,  initial_cash, training_blocks[0]["training_rows"])
    result.update({
        "model_type": "global_walk_forward",
        "model_config": config.to_dict(),
        "retrain_interval": retrain_interval,
        "validation_fraction": validation_fraction,
        "training_parameters": parameters,
        "training_runs": len(training_blocks),
        "training_blocks": training_blocks,
        "train_tickers": train_tickers,
        "ticker_seen_during_training": ticker in train_tickers})

    return result
