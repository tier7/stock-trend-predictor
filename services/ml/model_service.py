from math import isfinite

import pandas as pd
import sklearn
import lightgbm

from services.ml.model_config import ModelConfig
from services.ml.data_preparation_service import prepare_training_data, prepare_prediction_data, prepare_latest_prediction_data, validate_simulation_dates
from services.ml.tree_model_service import fit as fit_tree, predict as predict_tree
from services.ml.mlp_service import fit as fit_mlp, predict as predict_mlp
from services.ml.model_store_service import (build_training_metadata, save_global_model, save_torch_model, load_model, get_model_config)
from services.simulator_service import DEFAULT_CONFIDENCE_THRESHOLD, validate_confidence_threshold, build_backtest_result
from services.ml.evaluation_service import evaluate, calculate_baseline_loss
from services.ml.calibration_service import apply_saved_calibration

DEFAULT_TRAINING_TICKERS = (
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'TSLA', 'INTC', 'ORCL', 'GS', 'V',
    'MA', 'BRK-B', 'JNJ', 'UNH', 'ABBV',
    'SHEL', 'NEE', 'KO', 'PEP', 'MCD',
    'SBUX', 'WMT', 'COST', 'PG', 'NKE',
    'MC.PA', 'DIS', 'NFLX', 'CMCSA', 'TMUS',
    'BA', 'AIR.PA', 'LMT', 'CAT', 'DE',
    'UNP', 'FDX', 'UPS', 'O', 'AMT'
)


def resolve_training_parameters(config, *, n_estimators=None, epochs=None, learning_rate=None, batch_size=None, patience=None):
    if config.algorithm == "mlp":
        if n_estimators is not None:
            raise ValueError("n_estimators concerns tree models only")
        parameters = {
            "epochs": 50 if epochs is None else epochs,
            "learning_rate": 0.001 if learning_rate is None else learning_rate,
            "batch_size": 256 if batch_size is None else batch_size,
            "patience": 5 if patience is None else patience}
        for name in ("epochs", "batch_size", "patience"):
            if type(parameters[name]) is not int or parameters[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isfinite(parameters["learning_rate"]) or parameters["learning_rate"] <= 0:
            raise ValueError("learning_rate must be finite and positive")
        return parameters

    if config.algorithm not in ("random_forest", "lightgbm"):
        raise ValueError("Unsupported algorithm")
    if any(value is not None for value in (epochs, learning_rate, batch_size)):
        raise ValueError("epochs, learning_rate and batch_size concern MLP only")
    estimators = (100 if config.algorithm == "random_forest" else 500) if n_estimators is None else n_estimators
    if type(estimators) is not int or estimators < 1:
        raise ValueError("n_estimators must be a positive integer")
    parameters = {"n_estimators": estimators}
    if config.algorithm == "lightgbm":
        parameters["patience"] = 20 if patience is None else patience
        if type(parameters["patience"]) is not int or parameters["patience"] < 1:
            raise ValueError("patience must be a positive integer")
    elif patience is not None:
        raise ValueError("patience concerns MLP and LightGBM only")
    return parameters


def fit_model(train_df, feature_columns, config, *, validation_df=None, n_estimators=None, epochs=None, learning_rate=None, batch_size=None, patience=None, progress_callback=None):
    parameters = resolve_training_parameters(
        config, n_estimators=n_estimators, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, patience=patience)

    if config.algorithm == "mlp":
        return fit_mlp(train_df, feature_columns, config, validation_df=validation_df, progress_callback=progress_callback, **parameters)

    model = fit_tree(train_df, feature_columns, config, validation_df=validation_df, **parameters)
    details = {"sklearn_version": sklearn.__version__}
    if config.algorithm == "random_forest":
        details["n_estimators"] = model.n_estimators
        details["random_state"] = model.random_state
    else:
        details.update({
            "lightgbm_version": lightgbm.__version__,
            "trained_iterations": int(model.n_estimators_),
            "training_parameters": {
                "max_estimators": parameters["n_estimators"],
                "patience": parameters["patience"],
                "model_parameters": model.get_params()}})
        if validation_df is not None:
            details["best_iteration"] = int(model.best_iteration_)
            details["selection_scores"] = model.best_score_
    return model, None, details


def predict_model(model, data, feature_columns, config, scaler=None):
    if config.algorithm == "mlp":
        if scaler is None:
            raise ValueError("MLP prediction requires a scaler")
        return predict_mlp(model, data, feature_columns, scaler, config)
    if scaler is not None:
        raise ValueError("Tree prediction does not use a scaler")
    return predict_tree(model, data, feature_columns, config=config)


def fit_prepared_model(training_data, config, parameters, *, validation_start=None, progress_callback=None):
    final_train_df = training_data["final_train_df"]
    feature_columns = training_data["feature_columns"]
    metadata = {}

    report = progress_callback or print
    if config.algorithm == "random_forest":
        report(f"Train random forest on {len(final_train_df)} samples.")
        model, scaler, details = fit_model(final_train_df, feature_columns, config, **parameters)
        metadata.update(details)
    else:
        train_df = training_data["selection_train_df"]
        validation_df = training_data["validation_df"]
        selection_model, selection_scaler, selection_details = fit_model(
            train_df, feature_columns, config, validation_df=validation_df,
            progress_callback=progress_callback, **parameters)

        metadata["validation_start"] = validation_start
        metadata["validation_rows"] = int(len(validation_df))
        metadata["baseline_validation_loss"] = float(calculate_baseline_loss(train_df, validation_df, config))
        final_parameters = dict(parameters)
        if config.algorithm == "mlp":
            selected_steps = selection_details["best_epoch"]
            if selected_steps is None or selected_steps < 1:
                raise ValueError("No valid best epoch selected")
            final_parameters["epochs"] = selected_steps
            validation_results = predict_model(
                selection_model, validation_df, feature_columns, config, selection_scaler)
            evaluation = evaluate(validation_results, config)
            report(f"Baseline loss: {metadata['baseline_validation_loss']:.6f}")
            report(f"Best epoch: {selected_steps}")
            if config.task == "classification":
                report(f"Validation accuracy: {evaluation['accuracy']:.2%}")
                report(f"Validation log loss: {evaluation['log_loss']:.6f}")
            else:
                report(f"Validation MAE: {evaluation['mae']:.6f}")
                report(f"Validation RMSE: {evaluation['rmse']:.6f}")
        else:
            selected_steps = selection_details["best_iteration"]
            if selected_steps < 1:
                raise ValueError("No valid best iteration selected")
            final_parameters["n_estimators"] = selected_steps

        model, scaler, details = fit_model(
            final_train_df, feature_columns, config,
            progress_callback=progress_callback, **final_parameters
        )
        if config.algorithm == "mlp":
            metadata.update(
                {key: value for key, value in selection_details.items() if key not in ("history", "best_epoch")})
            metadata["epochs"] = selected_steps
            metadata["selection_history"] = selection_details["history"]
            metadata["final_history"] = details["history"]
            metadata["best_validation_loss"] = float(
                min(row["validation_loss"] for row in selection_details["history"]))
        else:
            metadata.update(details)
            metadata["best_iteration"] = selected_steps
            metadata["selection_scores"] = selection_details["selection_scores"]
            metadata["training_parameters"]["max_estimators"] = parameters["n_estimators"]


    return model, scaler, metadata


def train_model(
    training_cutoff, model_dir=None, progress_callback=None, *, config=None,
    training_tickers=None, validation_start=None, n_estimators=None, epochs=None,
    learning_rate=None, batch_size=None, patience=None, news_start=None, news_end=None,
    prepare_news=True, sector_assignments=None):

    if config is None:
        config = ModelConfig(algorithm="random_forest", task="classification")
    if config.scope != "global":
        raise ValueError("Expected a global model configuration")
    parameters = resolve_training_parameters(config, n_estimators=n_estimators, epochs=epochs,learning_rate=learning_rate, batch_size=batch_size, patience=patience)
    if config.algorithm == "random_forest":
        if validation_start is not None:
            raise ValueError("Random Forest training does not use validation_start")
        if "news" in config.feature_groups and training_tickers is None:
            raise ValueError("News features require an explicit list of training tickers")
        tickers = list(DEFAULT_TRAINING_TICKERS if training_tickers is None else training_tickers)
    else:
        if training_tickers is None or validation_start is None:
            raise ValueError("MLP and LightGBM require training tickers and validation_start")
        tickers = list(training_tickers)

    training_data = prepare_training_data(
        tickers, validation_start, training_cutoff, config,
        news_start=news_start, news_end=news_end, prepare_news=prepare_news,
        sector_assignments=sector_assignments, progress_callback=progress_callback)

    final_train_df = training_data["final_train_df"]
    feature_columns = training_data["feature_columns"]
    metadata = build_training_metadata(
        final_train_df, feature_columns, config, training_cutoff,
        news_start=training_data["news_start"], news_end=training_data["news_end"])

    model, scaler, training_details = fit_prepared_model(training_data, config, parameters, validation_start=validation_start, progress_callback=progress_callback)
    metadata.update(training_details)

    report = progress_callback or print
    report("Saving trained model")

    if config.algorithm == "mlp":
        return save_torch_model(model, scaler, metadata, model_dir)

    return save_global_model(model, metadata, model_dir)



def _predict_with_loaded_model(data, model, scaler, metadata, config, start_date, end_date, *, ticker=None, prepare_news=True, progress_callback=None, sector_assignments=None):
    feature_columns = metadata["feature_columns"]
    prediction_cutoff = metadata.get("calibration", {}).get("end_date", metadata["training_cutoff"])
    test_df = prepare_prediction_data(
        data, start_date, end_date, config, feature_columns, prediction_cutoff,
        ticker=ticker, prepare_news=prepare_news, progress_callback=progress_callback,
        sector_assignments=sector_assignments, saved_benchmarks=metadata.get("sector_benchmarks"))
    results = predict_model(model, test_df, feature_columns, config, scaler)
    results = apply_saved_calibration(results, metadata, config)
    if config.task == "classification":
        results["baseline_prediction"] = metadata["baseline_prediction"]
    return results


def predict_history(data, model_id, start_date, end_date, model_dir=None, *, ticker=None, prepare_news=True, progress_callback=None, sector_assignments=None):
    validate_simulation_dates(start_date, end_date)
    if data.empty:
        raise ValueError("Empty stock data")
    model, scaler, metadata = load_model(model_id, model_dir)
    config = get_model_config(metadata)
    results = _predict_with_loaded_model(
        data, model, scaler, metadata, config, start_date, end_date,
        ticker=ticker, prepare_news=prepare_news, progress_callback=progress_callback,
        sector_assignments=sector_assignments
    )
    return results, metadata


def simulate_model(data, model_id, start_date, end_date, initial_cash, fee_rate, model_dir=None, *,ticker=None, prepare_news=True, progress_callback=None, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, sector_assignments=None):
    confidence_threshold = validate_confidence_threshold(confidence_threshold)
    validate_simulation_dates(start_date, end_date)
    if data.empty:
        raise ValueError("Empty stock data")
    model, scaler, metadata = load_model(model_id, model_dir)
    config = get_model_config(metadata)
    if config.task != "classification" or config.horizon != 1:
        raise ValueError("Obecny symulator obsługuje klasyfikację z horyzontem 1 sesji.")
    results = _predict_with_loaded_model(
        data, model, scaler, metadata, config, start_date, end_date,
        ticker=ticker, prepare_news=prepare_news, progress_callback=progress_callback,
        sector_assignments=sector_assignments)
    result = build_backtest_result(
        results, start_date, end_date, initial_cash, fee_rate, metadata["training_rows"],
        confidence_threshold=confidence_threshold)
    result["model_type"] = "mlp" if config.algorithm == "mlp" else "global"
    result["model_metadata"] = metadata
    return result


def predict_latest(data, model_id, model_dir=None, *, ticker=None, as_of=None,prepare_news=True, progress_callback=None, sector_assignments=None):
    model, scaler, metadata = load_model(model_id, model_dir)
    config = get_model_config(metadata)
    feature_columns = metadata["feature_columns"]
    prediction_cutoff = metadata.get("calibration", {}).get("end_date", metadata["training_cutoff"])
    latest, prediction_window = prepare_latest_prediction_data(
        data, config, feature_columns, prediction_cutoff, ticker=ticker,
        as_of=as_of, prepare_news=prepare_news, progress_callback=progress_callback,
        sector_assignments=sector_assignments, saved_benchmarks=metadata.get("sector_benchmarks")
    )
    predictions = predict_model(model, latest, feature_columns, config, scaler)
    forecast = apply_saved_calibration(predictions, metadata, config).iloc[0]
    result = {
        "date": str(latest["date"].iloc[0]),
        "task": config.task,
        "horizon": config.horizon,
        "model_metadata": metadata}
    if "sector" in config.feature_groups:
        result["sector_benchmark"] = latest["sector_benchmark"].iloc[0]
    if prediction_window:
        for key in ("execution_date", "decision_at", "as_of", "market_open"):
            result[key] = prediction_window[key]
    if config.task == "classification":
        result["prediction"] = int(forecast["prediction"])
        result["probability_up"] = float(forecast["probability_up"])
    else:
        result["predicted_return"] = float(forecast["predicted_return"])
    if prediction_window and as_of is None:
        completed_at = pd.Timestamp.now(tz="UTC")
        market_open = pd.to_datetime(prediction_window["market_open"], utc=True)
        if completed_at >= market_open:
            raise ValueError(
                "The market opened while preparing the forecast; "
                "the upcoming-open decision is no longer available")
    return result
