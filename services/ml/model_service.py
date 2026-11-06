from dataclasses import replace
from math import isfinite

import pandas as pd
import sklearn
import lightgbm

from services.ml.model_config import ModelConfig
from services.ml.evaluation_universe import SEEN_TICKERS as DEFAULT_TRAINING_TICKERS
from services.ml.data_preparation_service import prepare_training_data, prepare_prediction_data, prepare_latest_prediction_data, validate_simulation_dates
from services.ml.tree_model_service import fit as fit_tree, predict as predict_tree
from services.ml.mlp_service import fit as fit_mlp, predict as predict_mlp
from services.ml.model_store_service import (build_training_metadata, save_global_model, save_torch_model, load_model, get_model_config)
from services.simulator_service import build_backtest_result
from services.strategy_service import StrategyConfig
from services.ml.evaluation_service import evaluate, calculate_baseline_loss
from services.ml.calibration_service import apply_saved_calibration



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
    if isinstance(model, dict) and "model_results" in model:
        if scaler is not None or model.get("method") not in ("blend", "stacking"):
            raise ValueError("Invalid hybrid model or shared scaler")

        predictions = {}
        for algorithm, component in model["model_results"].items():
            component_config = replace(config, algorithm=algorithm)
            predictions[algorithm] = predict_model(component["model"], data, feature_columns, component_config, component["scaler"])
        if model["method"] == "blend":
            weights=None
            blend_weights=model.get("blend_weights")

            if blend_weights is not None:
                weights = []
                for algorithm in predictions:
                    weights.append(blend_weights[algorithm])
            return blend_predictions(predictions.values(), config.task, identity_columns=["date"], weights=weights)
        return apply_stacking_model(predictions, model["stacking_model"], config, identity_columns=["date"])

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


def simulate_model(data, model_id, start_date, end_date, initial_cash=10000, *,strategy_config=None, model_dir=None, ticker=None,prepare_news=True, progress_callback=None, sector_assignments=None):
    validate_simulation_dates(start_date, end_date)
    if data.empty:
        raise ValueError("Empty stock data")
    if strategy_config is None:
        strategy_config = StrategyConfig()
    if not isinstance(strategy_config, StrategyConfig):
        raise ValueError("Expected StrategyConfig")
    initial_cash = float(initial_cash)
    if not isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("Initial cash must be finite and positive")

    model, scaler, metadata = load_model(model_id, model_dir)
    model_config = get_model_config(metadata)

    results = _predict_with_loaded_model(data, model, scaler, metadata, model_config,
    start_date, end_date, ticker=ticker, prepare_news=prepare_news, progress_callback=progress_callback, sector_assignments=sector_assignments)

    result = build_backtest_result(results, data, model_config, strategy_config, start_date, end_date, initial_cash, metadata["training_rows"])

    if metadata.get("model_id", "").startswith("hybrid_"):
        result["model_type"] = metadata["evaluation_method"]
    else:
        result["model_type"] = model_config.algorithm

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

def resolve_hybrid_parameters(config, model_parameters, method="blend", stacking_splits=5):
    if method not in ("blend", "stacking"):
        raise ValueError("Hybrid method must be blend or stacking")
    if config.scope != "global":
        raise ValueError("Expected a global model configuration")
    if not isinstance(model_parameters, dict) or len(model_parameters) < 2:
        raise ValueError("Must provide at least two algorithms")
    if method == "stacking" and (type(stacking_splits) is not int or stacking_splits < 2):
        raise ValueError("stacking_splits must be an integer of at least 2")
    parameters = {}
    for algorithm, options in model_parameters.items():
        if not isinstance(options, dict):
            raise ValueError("Each algorithm requires a dictionary of training parameters")
        parameters[algorithm] = resolve_training_parameters(replace(config, algorithm=algorithm), **options)
    return parameters


def prepare_ensemble_features(predictions, task, *, identity_columns=None):
    if task not in ("classification", "regression"):
        raise ValueError("Unsupported task")
    predictions = list(predictions)
    if len(predictions) < 2:
        raise ValueError("Combining requires at least two prediction dataframes")
    if any(frame.empty for frame in predictions):
        raise ValueError("Prediction dataframes cannot be empty")
    result = predictions[0].copy()
    identity_columns = list(["ticker", "date", "next_date", "target_end_date"] if identity_columns is None else identity_columns)
    if "date" not in identity_columns:
        raise ValueError("Prediction identity must include date")
    for column in ("ticker", "next_date", "target_end_date", "decision_at", "execution_date"):
        if column not in identity_columns and any(column in frame.columns for frame in predictions):
            identity_columns.append(column)
    features = pd.DataFrame(index=result.index)
    prediction_column = "probability_up" if task == "classification" else "predicted_return"

    for index, frame in enumerate(predictions):
        if not set(identity_columns).issubset(frame.columns):
            raise ValueError("Missing prediction identity columns")
        if frame[identity_columns].isna().any(axis=None):
            raise ValueError("Prediction identity columns cannot contain empty values")
        if frame.duplicated(["ticker", "date"] if "ticker" in frame.columns else ["date"]).any():
            raise ValueError("Duplicated predictions for the same ticker and date")
        if not result[identity_columns].equals(frame[identity_columns]):
            raise ValueError("Predictions must describe the same rows in the same order")
        if prediction_column not in frame.columns:
            raise ValueError(f"Missing prediction column: {prediction_column}")
        values = frame[prediction_column]
        if values.isna().any() or not values.map(isfinite).all():
            raise ValueError("Predictions must contain finite numbers")
        if task == "classification" and not values.between(0, 1).all():
            raise ValueError("Probabilities must be between 0 and 1")
        features[index] = values.to_numpy(dtype=float)
    return result, features


def blend_predictions(predictions, task, *, identity_columns=None, weights=None):
    result, features = prepare_ensemble_features(predictions, task, identity_columns=identity_columns)

    if weights is None:
        weights = [1] * len(features.columns)
    else:
        weights = list(weights)

    if len(weights) != len(features.columns):
        raise ValueError("Number of weights cannot be different than number of models")

    sum_weights = sum(weights)
    for weight in weights:
        if weight < 0 or not isfinite(weight):
            raise ValueError("Invalid weights")

    if not isfinite(sum_weights) or sum_weights <= 0:
        raise ValueError("Invalid weights")

    prediction_column = "probability_up" if task == "classification" else "predicted_return"
    result[prediction_column] = 0.0
    for column in features:
        result[prediction_column] += features[column] * weights[column]
    result[prediction_column] = result[prediction_column] / sum_weights
    if task == "classification":
        result["prediction"] = (result["probability_up"] >= 0.5).astype(int)
    return result


def apply_stacking_model(predictions, model, config, *, identity_columns=None):
    if (config.task == "classification") != hasattr(model, "predict_proba"):
        raise ValueError("Stacking model task does not match the configuration")
    algorithms = list(model.feature_names_in_)

    if set(predictions) != set(algorithms):
        raise ValueError("Stacking requires the same component algorithms used for training")

    result, features = prepare_ensemble_features([predictions[algorithm] for algorithm in algorithms], config.task, identity_columns=identity_columns)
    features.columns = algorithms
    if config.task == "classification":
        if list(model.classes_) != [0, 1]:
            raise ValueError("Invalid stacking classes")
        result["probability_up"] = model.predict_proba(features)[:, 1]
        result["prediction"] = result["probability_up"].ge(0.5).astype(int)
    else:
        result["predicted_return"] = model.predict(features)

    return result
