from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from services.ml.dataset_service import prepare_xy
from services.ml.model_config import ModelConfig
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    root_mean_squared_error
)


RETRAIN_INTERVAL = 20


def split_dataset_by_date(data, start_date, end_date):
    train_df = data[
        (data["date"] < start_date) & (data["target_end_date"] <= start_date)
    ]
    test_df = data[
        (data["date"] >= start_date) & (data["target_end_date"] <= end_date)
    ]
    return train_df, test_df


def build_date_blocks(test_df, retrain_interval):
    if retrain_interval <= 0:
        raise ValueError("Retrain interval must be positive")

    dates = test_df["date"].drop_duplicates().sort_values().tolist()
    blocks = []
    for position in range(0, len(dates), retrain_interval):
        block = dates[position:position + retrain_interval]
        blocks.append(block)

    return blocks


def train(train_df, feature_columns, config= None):
    if config is None:
        config = ModelConfig(
            algorithm="random_forest",
            task="classification",
        )

    if config.algorithm != "random_forest":
        raise ValueError("This function supports only random_forest")

    x_train, y_train = prepare_xy(train_df, feature_columns, config)

    if config.task == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=config.random_state)
    elif config.task == "regression":
        model = RandomForestRegressor(n_estimators=100, random_state=config.random_state)

    model.fit(x_train, y_train)
    return model

def predict(model, data, feature_columns, config=None):
    if config is None:
        config = ModelConfig(
            algorithm="random_forest",
            task="classification",
        )

    x = data[feature_columns]
    result = data.copy()

    if config.task == "classification":
        if not list(model.classes_) == [0, 1]:
            raise ValueError("Invalid model classes")

        result["prediction"] = model.predict(x)
        result["probability_up"] = model.predict_proba(x)[:,1]

    elif config.task == "regression":
        result["predicted_return"] = model.predict(x)

    return result

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
