from services.ml.model_config import ModelConfig
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    root_mean_squared_error,
    log_loss, mean_squared_error
)

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
            "log_loss": loss
        }
    elif config.task == "regression":
        predictions = results_df["predicted_return"]
        mae = mean_absolute_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)

        return{
            "predictions": predictions,
            "mae": mae,
            "rmse": rmse
        }

def calculate_baseline_loss(train_df, validation_df, config):
    if train_df.empty:
        raise ValueError("Empty train dataframe")
    if validation_df.empty:
        raise ValueError("Empty validation dataframe")

    baseline = train_df[config.target_column].mean()
    predictions = [baseline] * len(validation_df)
    y_true = validation_df[config.target_column]

    if config.task == "classification":
        loss = log_loss(y_true, predictions, labels=[0,1])
    else:
        loss = mean_squared_error(y_true, predictions)
    return loss