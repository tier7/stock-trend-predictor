from lightgbm import early_stopping
from services.ml.dataset_service import prepare_xy
from services.ml.tree_model_service import create_tree_model
from services.ml.data_preparation_service import prepare_training_data
from services.ml.feature_schema import get_feature_columns
from services.ml.news_feature_service import normalize_training_news_range, NEWS_FEATURE_POLICY
from datetime import datetime, timezone
import lightgbm
import sklearn
from services.ml.model_store_service import save_global_model
from services.ml.evaluation_service import calculate_baseline_loss
from services.ml.sector_feature_service import extract_sector_benchmarks


def train_model(train_df, feature_columns, config, *, validation_df=None, n_estimators=500, patience=20):
    if not config.algorithm == "lightgbm":
        raise ValueError("Incompatible algorithm")

    if type(patience) is not int or patience < 1:
        raise ValueError("Patience must be a positive integer")

    x_train, y_train = prepare_xy(train_df, feature_columns, config)
    model = create_tree_model(config, n_estimators=n_estimators)

    if validation_df is None:
        model.fit(x_train, y_train)
        return model

    x_validation, y_validation = prepare_xy(validation_df, feature_columns, config, require_both_classes=False)
    metric = ("binary_logloss" if config.task == "classification" else "l2")
    model.fit(x_train, y_train, eval_X = x_validation, eval_y = y_validation, eval_metric=metric, callbacks=[early_stopping(stopping_rounds=patience, first_metric_only=True)])

    return model

def train_lightgbm_model(tickers, validation_start, training_cutoff, *, config, n_estimators=500, patience=20, news_start=None, news_end=None, prepare_news=True, model_dir=None, sector_assignments=None):
    if not config.algorithm == "lightgbm":
        raise ValueError("Incompatible algorithm")
    if not tickers:
        raise ValueError("No tickers provided")

    feature_columns = get_feature_columns(config.feature_groups)
    if "news" in config.feature_groups:
        news_start, news_end = normalize_training_news_range(news_start, news_end, training_cutoff)
    data, train_df, validation_df = prepare_training_data(tickers, validation_start, training_cutoff, config, news_start=news_start, news_end=news_end, prepare_news=prepare_news, sector_assignments=sector_assignments)

    final_train_df = data[(data["date"] < training_cutoff) & (data["target_end_date"] <= training_cutoff)].copy()
    missing_tickers = sorted(set(tickers) - set(final_train_df["ticker"]))
    if missing_tickers:
        raise ValueError(
            f"Brak próbek treningowych przed wybraną datą: {missing_tickers}.")

    selection_model = train_model(train_df, feature_columns, config, validation_df = validation_df, n_estimators=n_estimators, patience=patience)
    best_iteration = int(selection_model.best_iteration_)

    if best_iteration < 1:
        raise ValueError("No valid best iteration selected")

    final_model = train_model(final_train_df, feature_columns, config, validation_df=None, n_estimators=best_iteration, patience=patience)

    metadata = {
        "config": config.to_dict(),
        "feature_columns": list(feature_columns),
        "training_cutoff": training_cutoff,
        "train_tickers": list(tickers),
        "training_rows": len(final_train_df),
        "training_start_date": str(final_train_df["date"].min()),
        "training_end_date": str(final_train_df["date"].max()),
        "last_target_end_date": str(final_train_df["target_end_date"].max()),
        "validation_start": validation_start,
        "validation_rows": len(validation_df),
        "best_iteration": best_iteration,
        "trained_iterations": int(final_model.n_estimators_),
        "training_parameters": {
            "max_estimators": n_estimators,
            "patience": patience,
            "model_parameters": final_model.get_params()},
        "selection_scores": selection_model.best_score_,
        "baseline_validation_loss": float(
            calculate_baseline_loss(train_df, validation_df, config)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lightgbm_version": lightgbm.__version__,
        "sklearn_version": sklearn.__version__,
    }

    if "news" in config.feature_groups:
        metadata["news_policy"] = NEWS_FEATURE_POLICY.copy()
        metadata["news_start"] = news_start
        metadata["news_end"] = news_end
    if "sector" in config.feature_groups:
        metadata["sector_benchmarks"] = extract_sector_benchmarks(final_train_df)
    if config.task == "classification":
        metadata["baseline_prediction"] = int(final_train_df["target"].mode().iloc[0])
        metadata["target_definition"] = (f"open(t+{config.horizon + 1}) > open(t+1)")
    else:
        metadata["baseline_return"] = float(final_train_df["target_return"].mean())
        metadata["target_definition"] = (f"open(t+{config.horizon + 1}) / open(t+1) - 1")

    return save_global_model(final_model, metadata, model_dir)

