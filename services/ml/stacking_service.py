from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from services.ml.data_preparation_service import select_training_rows
from services.ml.model_service import fit_prepared_model, predict_model, prepare_ensemble_features, resolve_hybrid_parameters


def fit_stacking_model(evaluation_data, config, model_parameters, *, n_splits=5, progress_callback=None):
    parameters = resolve_hybrid_parameters(config, model_parameters, "stacking", n_splits)
    data = evaluation_data["final_train_df"].copy()
    if data.empty:
        raise ValueError("Stacking training data cannot be empty")
    required = ["ticker", "date", "next_date", "target_end_date", config.target_column]
    if not set(required).issubset(data.columns) or data[required].isna().any(axis=None):
        raise ValueError("Stacking requires complete forecast identifiers and targets")
    if data.duplicated(["ticker", "date"]).any():
        raise ValueError("Stacking training data contains duplicate company sessions")
    if set(data["ticker"]) & set(evaluation_data.get("unseen_tickers") or []):
        raise ValueError("Unseen tickers cannot be used to train stacking")

    for name in ("calibration_df", "test_df"):
        reserved = evaluation_data.get(name)
        if reserved is not None and not reserved.empty:
            cutoff = reserved["date"].min()
            if data["date"].ge(cutoff).any() or data["target_end_date"].gt(cutoff).any():
                raise ValueError(f"Stacking training overlaps {name}")

    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = data["date"].drop_duplicates().tolist()
    block_size = len(dates) // (n_splits + 1)
    if block_size < 3:
        raise ValueError("Not enough training dates for stacking; reduce stacking_splits or provide more history")
    feature_columns = evaluation_data["feature_columns"]
    algorithms = list(parameters)
    blocks = []
    block_details = []
    report = progress_callback or print
    column = "probability_up" if config.task == "classification" else "predicted_return"

    for block_index in range(n_splits):
        start_position = block_size * (block_index + 1)
        end_position = block_size * (block_index + 2) if block_index < n_splits - 1 else len(dates)
        prediction_dates = dates[start_position:end_position]
        cutoff = prediction_dates[0]
        train_df = select_training_rows(data, cutoff)
        prediction_df = data.loc[data["date"].isin(prediction_dates)].copy()
        training_dates = train_df["date"].drop_duplicates().sort_values().tolist()
        if len(training_dates) < 3:
            raise ValueError(f"Not enough mature training history before stacking block {block_index + 1}")
        validation_start = training_dates[len(training_dates) * 4 // 5]
        selection_train_df = select_training_rows(train_df, validation_start)
        validation_df = train_df.loc[train_df["date"].ge(validation_start)].copy()
        if selection_train_df.empty or validation_df.empty:
            raise ValueError(f"Not enough past data for validation in stacking block {block_index + 1}")

        fold_data = {
            "selection_train_df": selection_train_df,
            "validation_df": validation_df,
            "final_train_df": train_df,
            "feature_columns": feature_columns}

        predictions = []
        training_details = {}
        report(f"Stacking block {block_index + 1}/{n_splits}: {prediction_dates[0]} to {prediction_dates[-1]}.")
        for algorithm, options in parameters.items():
            model_config = replace(config, algorithm=algorithm)
            model, scaler, details = fit_prepared_model(
                fold_data, model_config, options,
                validation_start=validation_start, progress_callback=progress_callback)
            predictions.append(predict_model(model, prediction_df, feature_columns, model_config, scaler))
            training_details[algorithm] = details
        reference, features = prepare_ensemble_features(predictions, config.task)
        features.columns = algorithms
        block = reference[required].copy()
        for algorithm in algorithms:
            block[algorithm] = features[algorithm].to_numpy()
        block["stacking_block"] = block_index + 1
        blocks.append(block)
        block_details.append({
            "block": block_index + 1,
            "prediction_start": str(prediction_dates[0]),
            "prediction_end": str(prediction_dates[-1]),
            "prediction_rows": int(len(prediction_df)),
            "training_rows": int(len(train_df)),
            "first_training_date": str(train_df["date"].min()),
            "last_training_date": str(train_df["date"].max()),
            "last_training_target_end_date": str(train_df["target_end_date"].max()),
            "validation_start": str(validation_start),
            "selection_training_rows": int(len(selection_train_df)),
            "selection_last_target_end_date": str(selection_train_df["target_end_date"].max()),
            "validation_rows": int(len(validation_df)),
            "validation_last_target_end_date": str(validation_df["target_end_date"].max()),
            "models": training_details})

    training_data = pd.concat(blocks, ignore_index=True)
    features = training_data[algorithms]
    targets = training_data[config.target_column]
    if not np.isfinite(targets.to_numpy(dtype=float)).all():
        raise ValueError("Stacking targets must contain finite numbers")
    if config.task == "classification":
        if not targets.isin([0, 1]).all() or targets.nunique() != 2:
            raise ValueError("Stacking targets must contain both classes 0 and 1")
        final_model = LogisticRegression(max_iter=1000, random_state=config.random_state)
        model_name = "logistic_regression"
    else:
        final_model = Ridge(alpha=1.0)
        model_name = "ridge"
    model = make_pipeline(StandardScaler(), final_model)
    model.fit(features, targets)
    details = {
        "method": "chronological_stacking",
        "meta_model": model_name,
        "meta_parameters": final_model.get_params(),
        "meta_scaler": "standard_scaler",
        "component_algorithms": algorithms,
        "component_prediction_column": column,
        "config": config.to_dict(),
        "n_splits": n_splits,
        "local_validation_fraction": 0.2,
        "training_rows": int(len(training_data)),
        "warmup_rows": int(len(data) - len(training_data)),
        "first_prediction_date": str(training_data["date"].min()),
        "last_prediction_date": str(training_data["date"].max()),
        "last_target_end_date": str(training_data["target_end_date"].max()),
        "tickers": sorted(training_data["ticker"].unique().tolist()),
        "blocks": block_details}

    return model, training_data, details
