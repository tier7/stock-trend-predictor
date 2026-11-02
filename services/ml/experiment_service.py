
from services.db import get_stock_data
from services.ml.walk_forward_service import RETRAIN_INTERVAL, run_walk_forward, train_and_predict
from services.ml.data_preparation_service import connect_data, split_dataset_by_date, validate_simulation_dates, prepare_evaluation_data
from services.ml.dataset_service import FEATURE_COLUMNS
from services.ml.model_config import ModelConfig
from services.ml.model_store_service import get_active_global_model, build_training_metadata, save_global_model, save_torch_model
import pandas as pd
from services.ml.model_service import DEFAULT_TRAINING_TICKERS, simulate_model, fit_prepared_model, predict_model, resolve_training_parameters
from services.ml.evaluation_service import evaluate, predict_baseline
from services.ml.calibration_service import fit_calibrator, apply_calibrator, serialize_calibrator


def build_result_row(ticker, model_type, result):
    return {
        "ticker": ticker,
        "model_type": model_type,
        "start_date": result["effective_test_start_date"],
        "end_date": result["effective_test_end_date"],
        "confidence_threshold": result["model_summary"]["confidence_threshold"],
        "accuracy": result["accuracy"],
        "baseline_accuracy": result["baseline_accuracy"],
        "model_return": result["model_summary"]["total_return"],
        "model_drawdown": result["model_summary"]["max_drawdown"],
        "buy_hold_return": result["buyandhold_summary"]["total_return"],
        "buy_hold_drawdown": result["buyandhold_summary"]["max_drawdown"],
        "excess_return": (result["model_summary"]["total_return"]- result["buyandhold_summary"]["total_return"]),
        "model_exposure": result["model_summary"]["market_exposure"],
        "transactions": result["model_summary"]["total_transactions"]}

def run_simulation(ticker, model_type, start_date, end_date, global_model_id=None, initial_cash=10000, fee_rate=0.001, confidence_threshold=0.6, *, torch_model_id=None):
    data = get_stock_data(ticker,years=None)
    if data.empty:
        raise ValueError("No data")

    if model_type in ("global", "mlp"):
        if model_type == "global":
            if global_model_id is None:
                global_model_id = get_active_global_model()["model_id"]
            model_id = global_model_id
        else:
            if torch_model_id is None:
                raise ValueError("torch_model_id is required for MLP")
            model_id = torch_model_id
        result = simulate_model(
            data, model_id, start_date, end_date, initial_cash, fee_rate,
            confidence_threshold=confidence_threshold, ticker=ticker
        )
    elif model_type == "local":
        result = train_and_predict(
            data, start_date, end_date, initial_cash, fee_rate,
            confidence_threshold=confidence_threshold
        )
    else:
        raise ValueError("model_type must be either 'global', 'local' or 'mlp'")

    row = build_result_row(ticker, model_type, result)
    row["model_id"] = result.get("model_metadata", {}).get("model_id")
    row["initial_cash"] = initial_cash
    row["fee_rate"] = fee_rate
    return row

def run_experiments(tickers,model_types,periods,global_model_id=None,initial_cash=10000,fee_rate=0.001,confidence_threshold=0.6,*,torch_model_id=None,):
    rows = []
    if "global" in model_types and global_model_id is None:
        metadata = get_active_global_model()
        global_model_id = metadata["model_id"]

    for period in periods:
        for ticker in tickers:
            for model_type in model_types:
                row = run_simulation(
                    ticker,
                    model_type,
                    period["start_date"],
                    period["end_date"],
                    global_model_id=global_model_id,
                    initial_cash=initial_cash,
                    fee_rate=fee_rate,
                    confidence_threshold=confidence_threshold,
                    torch_model_id=torch_model_id,
                )
                row["period"] = period["name"]
                rows.append(row)

    return rows

def summarize_results(report):
    summary = []

    for model_type, group in report.groupby("model_type"):
        row = {
            "model_type": model_type,
            "experiments": len(group),
            "wins_vs_buy_hold": len(group[(group["excess_return"] > 0)]),
            "profitable_runs": len(group[(group["model_return"] > 0)]),
            "mean_excess_return": group["excess_return"].mean(),
            "mean_drawdown": group["model_drawdown"].mean(),
        }
        summary.append(row)

    return pd.DataFrame(summary)

UNSEEN_TICKERS = ("NVDA", "AMD", "JPM", "BAC", "PFE", "LLY", "XOM", "CVX", "VZ", "T")


def run_experiment(start_date, end_date):
    validate_simulation_dates(start_date, end_date)
    train_tickers = list(DEFAULT_TRAINING_TICKERS)
    data = connect_data(train_tickers + list(UNSEEN_TICKERS))
    train_df = data[data["ticker"].isin(train_tickers)]
    unseen_df = data[data["ticker"].isin(UNSEEN_TICKERS)]
    train_period_df, seen_test_df = split_dataset_by_date(train_df, start_date, end_date)
    _, unseen_test_df = split_dataset_by_date(unseen_df, start_date, end_date)
    test_df = pd.concat([seen_test_df, unseen_test_df], ignore_index=True)
    if test_df.empty:
        raise ValueError("No test data in the selected period.")
    print("Train rows:", len(train_period_df))
    print("Seen test rows:", len(seen_test_df))
    print("Unseen test rows:", len(unseen_test_df))

    def print_block(block):
        for key in ("start_date", "end_date", "training_rows", "test_rows"):
            print(block[key])

    config = ModelConfig(algorithm="random_forest", task="classification")
    results, _ = run_walk_forward(
        train_df, test_df, FEATURE_COLUMNS, config,
        retrain_interval=RETRAIN_INTERVAL, progress_callback=print_block)
    seen = results[results["ticker"].isin(train_tickers)]
    unseen = results[results["ticker"].isin(UNSEEN_TICKERS)]
    return seen, unseen

def evaluate_prepared_model(evaluation_data, config, parameters, *, validation_start=None, calibrate=False, progress_callback=None):
    if calibrate and config.task != "classification":
        raise ValueError("Calibration task not supported.")

    model, scaler, training_details = fit_prepared_model(evaluation_data, config, parameters, validation_start=validation_start, progress_callback=progress_callback)
    feature_columns = evaluation_data["feature_columns"]
    test_predictions = predict_model(
        model, evaluation_data["test_df"], feature_columns, config, scaler)
    baseline_predictions = predict_baseline(
        evaluation_data["final_train_df"], evaluation_data["test_df"], config)

    predictions = {"model": test_predictions, "baseline": baseline_predictions}
    calibrator = None

    if calibrate:
        calibration_predictions = predict_model(
            model, evaluation_data["calibration_df"], feature_columns, config, scaler)
        calibrator = fit_calibrator(calibration_predictions, config)
        predictions["calibrated_model"] = apply_calibrator(test_predictions, calibrator, config)

    metrics = {name: evaluate(frame, config) for name, frame in predictions.items()}

    return {
        "model": model,
        "scaler": scaler,
        "calibrator": calibrator,
        "training_details": training_details,
        "predictions": predictions,
        "metrics": metrics
    }


def run_model_evaluation(tickers, validation_start, calibration_start, test_start, test_end, config, *, calibrate=False, n_estimators=None, epochs=None, learning_rate=None,
    batch_size=None, patience=None, news_start=None, prepare_news=True, sector_assignments=None, progress_callback=None, save_model=False, model_dir=None):

    if calibrate and config.task != "classification":
        raise ValueError("Calibration task not supported.")

    parameters = resolve_training_parameters(config, n_estimators=n_estimators, epochs=epochs, learning_rate=learning_rate,batch_size=batch_size, patience=patience)
    evaluation_data = prepare_evaluation_data(tickers, validation_start, calibration_start, test_start, test_end, config,news_start=news_start, prepare_news=prepare_news,sector_assignments=sector_assignments, progress_callback=progress_callback)

    result = evaluate_prepared_model(evaluation_data, config, parameters, validation_start=validation_start, calibrate=calibrate, progress_callback=progress_callback)
    if save_model:
        training_news_end = evaluation_data["news_end"]
        if training_news_end is not None:
            training_news_end = min(training_news_end, pd.Timestamp(calibration_start, tz="UTC").isoformat())
        metadata = build_training_metadata(
            evaluation_data["final_train_df"], evaluation_data["feature_columns"], config, calibration_start,
            news_start=evaluation_data["news_start"], news_end=training_news_end)
        metadata.update(result["training_details"])
        metadata["evaluation_periods"] = {"validation_start": validation_start, "calibration_start": calibration_start, "test_start": test_start, "test_end": test_end}
        if calibrate:
            metadata["calibration"] = serialize_calibrator(result["calibrator"], evaluation_data["calibration_df"], config, calibration_start, test_start)
        if config.algorithm == "mlp":
            result["metadata"] = save_torch_model(result["model"], result["scaler"], metadata, model_dir)
        else:
            result["metadata"] = save_global_model(result["model"], metadata, model_dir)
    return result

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
