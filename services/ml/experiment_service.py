
from services.db import get_stock_data
from services.strategy_service import DEFAULT_CONFIDENCE_THRESHOLD, StrategyConfig
from services.ml.walk_forward_service import RETRAIN_INTERVAL, run_walk_forward, train_and_predict
from services.ml.data_preparation_service import connect_data, split_dataset_by_date, validate_simulation_dates, prepare_evaluation_data
from services.ml.dataset_service import FEATURE_COLUMNS
from services.ml.model_config import ModelConfig
from services.ml.evaluation_universe import UNSEEN_TICKERS
from services.ml.model_store_service import get_active_global_model, build_training_metadata, save_global_model, save_torch_model, save_hybrid_model
import pandas as pd
from services.ml.model_service import DEFAULT_TRAINING_TICKERS, simulate_model, fit_prepared_model, predict_model, resolve_training_parameters, blend_predictions
from services.ml.model_service import resolve_hybrid_parameters, apply_stacking_model
from services.ml.stacking_service import fit_stacking_model
from services.ml.evaluation_service import evaluate, predict_baseline
from services.ml.calibration_service import fit_calibrator, apply_calibrator, serialize_calibrator, restore_calibrator
from uuid import uuid4
import json
from pathlib import Path
import platform
from importlib.metadata import version
import joblib
from hashlib import sha256
from dataclasses import replace
from copy import deepcopy

def build_result_row(ticker, model_type, result):
    return {
        "ticker": ticker,
        "model_type": model_type,
        "start_date": result["effective_test_start_date"],
        "end_date": result["effective_test_end_date"],
        "confidence_threshold": result["model_summary"]["confidence_threshold"],
        "accuracy": result.get("accuracy"),
        "baseline_accuracy": result.get("baseline_accuracy"),
        "task": result["task"],
        "horizon": result["horizon"],
        "mae": result.get("mae"),
        "rmse": result.get("rmse"),
        "model_return": result["model_summary"]["total_return"],
        "model_drawdown": result["model_summary"]["max_drawdown"],
        "buy_hold_return": result["buyandhold_summary"]["total_return"],
        "buy_hold_drawdown": result["buyandhold_summary"]["max_drawdown"],
        "excess_return": (result["model_summary"]["total_return"]- result["buyandhold_summary"]["total_return"]),
        "model_exposure": result["model_summary"]["market_exposure"],
        "transactions": result["model_summary"]["total_transactions"]}

def run_simulation(ticker, model_type, start_date, end_date, global_model_id=None, initial_cash=10000, fee_rate=0.001, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, *, torch_model_id=None):
    strategy_config = StrategyConfig(confidence_threshold=confidence_threshold, fee_rate=fee_rate)
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

        result = simulate_model(data,model_id,start_date,end_date,initial_cash=initial_cash,strategy_config=strategy_config,ticker=ticker)
    elif model_type == "local":
        result = train_and_predict(data, start_date, end_date, initial_cash=initial_cash, strategy_config=strategy_config)
    else:
        raise ValueError("model_type must be either 'global', 'local' or 'mlp'")

    row = build_result_row(ticker, model_type, result)
    row["model_id"] = result.get("model_metadata", {}).get("model_id")
    row["initial_cash"] = initial_cash
    row["fee_rate"] = fee_rate
    return row

def run_experiments(tickers,model_types,periods,global_model_id=None,initial_cash=10000,fee_rate=0.001,confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,*,torch_model_id=None):
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
    cohort_metrics = {"seen": {}, "unseen": {}}
    training_tickers = evaluation_data["final_train_df"]["ticker"].unique()
    for name, frame in predictions.items():
        seen_mask = frame["ticker"].isin(training_tickers)
        cohorts = {"seen": frame.loc[seen_mask], "unseen": frame.loc[~seen_mask]}
        for cohort_name, cohort_frame in cohorts.items():
            if not cohort_frame.empty:
                cohort_metrics[cohort_name][name] = evaluate(cohort_frame, config)

    return {
        "model": model,
        "scaler": scaler,
        "calibrator": calibrator,
        "training_details": training_details,
        "predictions": predictions,
        "metrics": metrics,
        "cohort_metrics": cohort_metrics
    }



def build_evaluation_metadata(
    evaluation_data, config, parameters, validation_start, calibration_start, test_start, test_end, *,
    train_start=None, calibrate=False, method="single", stacking_splits=None):
    if method not in ("single", "blend", "stacking"):
        raise ValueError("Unsupported evaluation method")
    training_news_end = evaluation_data["news_end"]
    if training_news_end is not None:
        training_news_end = min(training_news_end, pd.Timestamp(calibration_start, tz="UTC").isoformat())
    metadata = build_training_metadata(
        evaluation_data["final_train_df"], evaluation_data["feature_columns"], config, calibration_start,
        news_start=evaluation_data["news_start"], news_end=training_news_end)
    project_dir = Path(__file__).resolve().parents[2]
    source_files = sorted(project_dir.glob("*.py")) + sorted((project_dir / "services").rglob("*.py"))
    metadata["code_hashes"] = {path.relative_to(project_dir).as_posix(): sha256(path.read_bytes()).hexdigest() for path in source_files}
    metadata["environment"] = {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit-learn": version("scikit-learn"),
        "torch": version("torch"),
        "lightgbm": version("lightgbm"),
        "joblib": version("joblib")}
    metadata["evaluation_id"] = f"evaluation_{uuid4().hex}"
    metadata["evaluation_method"] = method
    metadata["evaluation_parameters"] = deepcopy(parameters)
    metadata["evaluation_periods"] = {
        "train_start": train_start,
        "validation_start": validation_start,
        "calibration_start": calibration_start,
        "test_start": test_start,
        "test_end": test_end}
    metadata["evaluation_cohorts"] = {
        "training_tickers": list(evaluation_data["training_tickers"]),
        "unseen_tickers": list(evaluation_data["unseen_tickers"])}
    metadata["evaluation_news_range"] = {"start": evaluation_data["news_start"], "end": evaluation_data["news_end"]}
    metadata["calibration_enabled"] = calibrate
    dataset_names = ("selection_train_df", "validation_df", "final_train_df", "calibration_df", "seen_test_df", "unseen_test_df", "test_df")
    metadata["dataset_rows"] = {name: int(len(evaluation_data[name])) for name in dataset_names}
    if method != "single":
        metadata["component_algorithms"] = list(parameters)
    if method == "stacking":
        metadata["stacking_splits"] = stacking_splits
    return metadata


def save_evaluation_results(result, report_dir, evaluation_data):
    metadata = result["evaluation_metadata"]
    evaluation_id = metadata["evaluation_id"]
    payload = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
    metric_groups = {"all": result["metrics"], "seen": result["cohort_metrics"]["seen"], "unseen": result["cohort_metrics"]["unseen"]}
    metric_rows = []

    for group_name, variants in metric_groups.items():
        for variant_name, metrics in variants.items():
            row = {"evaluation_id": evaluation_id, "group": group_name, "variant": variant_name, "rows": int(len(metrics["predictions"]))}

            for metric_name, value in metrics.items():
                if metric_name in ("predictions", "probability_up"):
                    continue

                if metric_name == "confusion_matrix":
                    row.update({name: int(count) for name, count in value.items()})
                else:
                    row[metric_name] = float(value)

            metric_rows.append(row)

    columns = ["ticker", "date", "next_date", "target_end_date", "target_return"]
    if metadata["config"]["task"] == "classification":
        columns.extend(["target", "probability_up", "prediction"])
    else:
        columns.append("predicted_return")

    prediction_frames = []
    training_tickers = metadata["evaluation_cohorts"]["training_tickers"]

    for variant_name, frame in result["predictions"].items():
        selected_columns = columns + (["sector_benchmark"] if "sector_benchmark" in frame.columns else [])
        predictions = frame[selected_columns].copy()
        groups = predictions["ticker"].isin(training_tickers).map({True: "seen", False: "unseen"})
        predictions.insert(0, "group", groups)
        predictions.insert(0, "variant", variant_name)
        predictions.insert(0, "evaluation_id", evaluation_id)
        prediction_frames.append(predictions)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    directory = Path(report_dir) / evaluation_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "metadata.json").write_text(payload + "\n", encoding="utf-8")
    metrics_df.to_csv(directory / "metrics.csv", index=False, encoding="utf-8")
    predictions_df.to_csv(directory / "predictions.csv", index=False, encoding="utf-8")


    snapshot = {name: value for name, value in evaluation_data.items() if name not in ("seen_test_df", "unseen_test_df")}
    joblib.dump(snapshot, directory / "datasets.joblib", compress=3)
    stacking_training_data = result.get("stacking_training_data")
    if stacking_training_data is not None:
        stacking_predictions = stacking_training_data.copy()
        stacking_predictions.insert(0, "evaluation_id", evaluation_id)
        stacking_predictions.to_csv(directory / "stacking_training_predictions.csv", index=False, encoding="utf-8")

    return str(directory)


def run_model_evaluation(
    tickers, validation_start, calibration_start, test_start, test_end, config, *,
    calibrate=False, n_estimators=None, epochs=None, learning_rate=None,
    batch_size=None, patience=None, news_start=None, prepare_news=True,
    sector_assignments=None, progress_callback=None, save_model=False,
    model_dir=None, unseen_tickers=None, report_dir=None, train_start=None):
    if calibrate and config.task != "classification":
        raise ValueError("Calibration task not supported.")
    parameters = resolve_training_parameters(
        config, n_estimators=n_estimators, epochs=epochs, learning_rate=learning_rate,
        batch_size=batch_size, patience=patience)

    evaluation_data = prepare_evaluation_data(tickers, validation_start, calibration_start, test_start, test_end, config,news_start=news_start, prepare_news=prepare_news, sector_assignments=sector_assignments,progress_callback=progress_callback, unseen_tickers=unseen_tickers, train_start=train_start)

    metadata = build_evaluation_metadata(evaluation_data, config, parameters, validation_start, calibration_start, test_start, test_end,train_start=train_start, calibrate=calibrate)

    result = evaluate_prepared_model(evaluation_data, config, parameters, validation_start=validation_start,calibrate=calibrate, progress_callback=progress_callback)
    metadata.update(result["training_details"])
    if calibrate:
        metadata["calibration"] = serialize_calibrator(result["calibrator"], evaluation_data["calibration_df"], config, calibration_start, test_start)
    if save_model:
        if config.algorithm == "mlp":
            result["metadata"] = save_torch_model(result["model"], result["scaler"], metadata, model_dir)
        else:
            result["metadata"] = save_global_model(result["model"], metadata, model_dir)
    result["evaluation_metadata"] = result.get("metadata", metadata)
    if report_dir is not None:
        result["report_dir"] = save_evaluation_results(result, report_dir, evaluation_data)
    return result


def repeat_model_evaluation(report_dir, progress_callback=None):
    directory = Path(report_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    method = metadata.get("evaluation_method", "single")
    if method not in ("single", "blend", "stacking"):
        raise ValueError("Unsupported evaluation method")
    config = ModelConfig.from_dict(metadata["config"])
    calibrate = metadata.get("calibration_enabled", False)
    if calibrate:
        restore_calibrator(metadata.get("calibration"), config)
    if method == "single":
        parameters = resolve_training_parameters(config, **metadata["evaluation_parameters"])
    else:
        algorithms = metadata["component_algorithms"]
        options = metadata["evaluation_parameters"]

        if not isinstance(algorithms, list) or any(not isinstance(name, str) for name in algorithms):
            raise ValueError("Invalid hybrid component order")
        if not isinstance(options, dict) or len(set(algorithms)) != len(algorithms) or set(algorithms) != set(options):
            raise ValueError("Hybrid components do not match the saved parameters")
        stacking_splits = metadata.get("stacking_splits") if method == "stacking" else 5
        parameters = resolve_hybrid_parameters(config, {name: options[name] for name in algorithms}, method, stacking_splits)
    evaluation_data = joblib.load(directory / "datasets.joblib")
    validation_start = metadata["evaluation_periods"]["validation_start"]
    if method == "single":
        result = evaluate_prepared_model(evaluation_data, config, parameters, validation_start=validation_start, calibrate=calibrate, progress_callback=progress_callback)
    else:
        result = evaluate_hybrid_model(
            evaluation_data, config, parameters, validation_start=validation_start,
            progress_callback=progress_callback, method=method, stacking_splits=stacking_splits,
            blend_weights=metadata.get("blend_weights"), calibrate=calibrate)

    result["source_metadata"] = metadata
    return result


def evaluate_hybrid_model(evaluation_data, config, model_parameters, *, validation_start=None, progress_callback=None, method="blend", stacking_splits=5, blend_weights=None, calibrate=False):

    if calibrate and config.task != "classification":
        raise ValueError("Calibration requires classification")

    parameters = resolve_hybrid_parameters(config, model_parameters, method, stacking_splits)


    if blend_weights is not None:
        if method != "blend":
            raise ValueError("Blend weights require method blend")
        if not isinstance(blend_weights, dict) or set(blend_weights) != set(parameters):
            raise ValueError("Weights must match selected algorithms")

        blend_weights = blend_weights.copy()

    elif method == "blend":
        blend_weights = {algorithm:1 for algorithm in parameters}


    stacking_model = None
    stacking_training_data = None
    stacking_training_details = {}
    if method == "stacking":
        stacking_model, stacking_training_data, stacking_training_details = fit_stacking_model(evaluation_data, config, parameters, n_splits=stacking_splits, progress_callback=progress_callback)

    model_results = {}
    for algorithm, options in parameters.items():
        model_config = replace(config, algorithm=algorithm)
        model_results[algorithm] = evaluate_prepared_model(evaluation_data, model_config, options, validation_start=validation_start,progress_callback=progress_callback)

    predictions = {name: result["predictions"]["model"] for name, result in model_results.items()}
    metrics = {name: result["metrics"]["model"] for name, result in model_results.items()}

    if method == "blend":
        weights = []
        for algorithm in predictions:
            weights.append(blend_weights[algorithm])
        combined = blend_predictions(predictions.values(), config.task, weights=weights)
    else:
        combined = apply_stacking_model(predictions, stacking_model, config)


    calibrator = None
    if calibrate:
        hybrid_model = {"method": method,"model_results": model_results,"stacking_model": stacking_model,"blend_weights": blend_weights}
        calibration_predictions = predict_model(hybrid_model, evaluation_data["calibration_df"], evaluation_data["feature_columns"], config)
        calibrator = fit_calibrator(calibration_predictions, config)

    predictions[method] = combined
    metrics[method] = evaluate(combined, config)
    if calibrate:
        calibrated_name = f"calibrated_{method}"
        predictions[calibrated_name] = apply_calibrator(combined, calibrator, config)
        metrics[calibrated_name] = evaluate(predictions[calibrated_name], config)

    first_result = next(iter(model_results.values()))
    predictions["baseline"] = first_result["predictions"]["baseline"]
    metrics["baseline"] = first_result["metrics"]["baseline"]
    training_tickers = evaluation_data["final_train_df"]["ticker"].unique()
    seen_mask = combined["ticker"].isin(training_tickers)
    groups = {"seen": combined.loc[seen_mask], "unseen": combined.loc[~seen_mask]}
    group_metrics = {"seen": {}, "unseen": {}}
    for group_name, frame in groups.items():
        if not frame.empty:
            for algorithm, result in model_results.items():
                group_metrics[group_name][algorithm] = result["cohort_metrics"][group_name]["model"]
            group_metrics[group_name]["baseline"] = first_result["cohort_metrics"][group_name]["baseline"]
            group_metrics[group_name][method] = evaluate(frame, config)
            if calibrate:
                group_mask = seen_mask if group_name == "seen" else ~seen_mask
                calibrated_frame = predictions[calibrated_name].loc[group_mask]
                group_metrics[group_name][calibrated_name] = evaluate(calibrated_frame, config)
    return {
        "method": method,
        "model_results": model_results,
        "stacking_model": stacking_model,
        "stacking_training_data": stacking_training_data,
        "stacking_training_details": stacking_training_details,
        "predictions": predictions,
        "metrics": metrics,
        "cohort_metrics": group_metrics,
        "blend_weights": blend_weights,
        "calibrator": calibrator
    }


def run_hybrid_evaluation(tickers, validation_start, calibration_start, test_start, test_end, config, model_parameters, *, train_start=None, news_start=None, prepare_news=True, sector_assignments=None,
    unseen_tickers=None, progress_callback=None, method="blend", stacking_splits=5, report_dir=None, save_model=False, model_dir=None, blend_weights=None, calibrate=False):

    if calibrate and config.task != "classification":
        raise ValueError("Calibration requires classification")

    parameters = resolve_hybrid_parameters(config, model_parameters, method, stacking_splits)
    evaluation_data = prepare_evaluation_data(tickers, validation_start, calibration_start, test_start, test_end, config,train_start=train_start, news_start=news_start, prepare_news=prepare_news,sector_assignments=sector_assignments, unseen_tickers=unseen_tickers, progress_callback=progress_callback)

    metadata = build_evaluation_metadata(evaluation_data, config, parameters, validation_start, calibration_start, test_start, test_end,train_start=train_start, calibrate=calibrate, method=method, stacking_splits=stacking_splits)
    result = evaluate_hybrid_model(evaluation_data, config, parameters, validation_start=validation_start,progress_callback=progress_callback, method=method, stacking_splits=stacking_splits,blend_weights=blend_weights, calibrate=calibrate)

    metadata["component_training_details"] = {name: component["training_details"] for name, component in result["model_results"].items()}
    if method == "stacking":
        metadata["stacking_training_details"] = deepcopy(result["stacking_training_details"])
    if method == "blend":
        metadata["blend_weights"] = result["blend_weights"].copy()
    if calibrate:
        metadata["calibration"] = serialize_calibrator(result["calibrator"], evaluation_data["calibration_df"], config, calibration_start, test_start)
    if save_model:
        result["metadata"] = save_hybrid_model(result, metadata, model_dir)

    result["evaluation_metadata"] = result.get("metadata", metadata)
    if report_dir is not None:
        result["report_dir"] = save_evaluation_results(result, report_dir, evaluation_data)
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
