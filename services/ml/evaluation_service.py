from services.ml.model_config import ModelConfig
from sklearn.metrics import (
    accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,
    mean_absolute_error,root_mean_squared_error,log_loss, mean_squared_error, roc_auc_score)
import numpy as np

CORRELATION_POLICY = {
    "ic_method": "pearson",
    "rank_ic_method": "spearman",
    "minimum_observations": 3,
    "time_series_group": "ticker",
    "cross_sectional_group": "date",
    "aggregation": "equal_weight_valid_groups",
    "undefined_value": None}

SCALAR_METRIC_COLUMNS = [
    "accuracy", "precision", "recall", "f1", "log_loss", "brier_score",
    "roc_auc", "mae", "rmse"] + [f"{scope}_{name}" for scope in ("time_series", "cross_sectional")
    for name in ("ic_mean", "rank_ic_mean", "ic_groups", "rank_ic_groups", "total_groups")]

def calculate_correlation(actual, predicted, method="pearson"):
    if method not in ("pearson", "spearman"):
        raise ValueError("Unsupported correlation method")

    actual = actual.reset_index(drop=True)
    predicted = predicted.reset_index(drop=True)

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lengths must match")

    if (not np.isfinite(actual.to_numpy(dtype=float)).all() or not np.isfinite(predicted.to_numpy(dtype=float)).all()):
        raise ValueError("Correlation inputs must be finite")

    if len(actual) < CORRELATION_POLICY["minimum_observations"]:
        return None
    if actual.nunique() < 2 or predicted.nunique() < 2:
        return None

    correlation = actual.corr(predicted, method=method)
    return float(correlation) if np.isfinite(correlation) else None

def summarize_correlations(results_df, group_column):
    if group_column not in ("ticker", "date"):
        raise ValueError("Unsupported correlation grouping")
    if group_column not in results_df.columns:
        raise ValueError(f"Missing grouping column: {group_column}")
    if results_df[group_column].isna().any():
        raise ValueError("Grouping values cannot be missing")

    ic_values = []
    rank_ic_values = []
    groups = results_df.groupby(group_column)

    for _, group in groups:
        actual = group["target_return"]
        predicted = group["predicted_return"]

        ic = calculate_correlation(actual, predicted)
        rank_ic = calculate_correlation(actual, predicted, method="spearman")

        if ic is not None:
            ic_values.append(ic)

        if rank_ic is not None:
            rank_ic_values.append(rank_ic)

    return {
        "ic_mean": float(np.mean(ic_values)) if ic_values else None,
        "rank_ic_mean": (float(np.mean(rank_ic_values)) if rank_ic_values else None),
        "ic_groups": len(ic_values),
        "rank_ic_groups": len(rank_ic_values),
        "total_groups": int(groups.ngroups)}

def evaluate(results_df, config=None):
    if config is None:
        config = ModelConfig(
            algorithm="random_forest",
            task="classification",
        )

    if results_df.empty:
        raise ValueError("Empty test dataframe")

    y_test = results_df[config.target_column]


    if config.task == "classification":
        predictions = results_df["prediction"]
        acc = accuracy_score(y_test, predictions)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        loss = log_loss(y_test, results_df["probability_up"], labels=[0, 1])
        brier_score = ((results_df["probability_up"] - y_test) ** 2).mean()
        auc = None
        if y_test.nunique() == 2:
            auc = float(roc_auc_score(y_test, results_df["probability_up"]))
        return {
            "predictions": predictions,
            "probability_up": results_df["probability_up"],
            "accuracy": acc,
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "log_loss": loss,
            "brier_score": float(brier_score),
            "roc_auc": auc
        }
    elif config.task == "regression":
        predictions = results_df["predicted_return"]

        metrics = {"predictions": predictions, "mae": mean_absolute_error(y_test, predictions),"rmse": root_mean_squared_error(y_test, predictions)}
        correlation_data = results_df.copy()

        if "ticker" not in correlation_data.columns:
            correlation_data["ticker"] = "single_ticker"

        if "date" in correlation_data.columns:
            if correlation_data.duplicated(["ticker", "date"]).any():
                raise ValueError("Duplicate predictions for ticker and date")

        for prefix, group_column in (("time_series", "ticker"),("cross_sectional", "date")):
            if group_column in correlation_data.columns:
                summary = summarize_correlations(correlation_data, group_column)
            else:
                summary = {
                    "ic_mean": None,
                    "rank_ic_mean": None,
                    "ic_groups": 0,
                    "rank_ic_groups": 0,
                    "total_groups": 0}

            for name, value in summary.items():
                metrics[f"{prefix}_{name}"] = value

        return metrics

def predict_baseline(train_df, evaluation_df, config):
    if train_df.empty or evaluation_df.empty:
        raise ValueError("Empty train or evaluation dataframe")
    evaluation_df = evaluation_df.copy()
    col_mean = train_df[config.target_column].mean()

    if config.task == "classification":
        evaluation_df["probability_up"] = col_mean
        evaluation_df["prediction"] = int(train_df[config.target_column].mode().iloc[0])


    if config.task == "regression":
        evaluation_df["predicted_return"] = col_mean

    return evaluation_df


def calculate_baseline_loss(train_df, validation_df, config):

    predictions_df = predict_baseline(train_df, validation_df, config)
    y_true = validation_df[config.target_column]

    if config.task == "classification":
        loss = log_loss(y_true, predictions_df["probability_up"], labels=[0,1])
    else:
        loss = mean_squared_error(y_true, predictions_df["predicted_return"])
    return loss
