from math import isfinite

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from services.ml.dataset_service import prepare_xy

FEATURE_BASELINE_THRESHOLD = 0.5


def get_feature_baseline_spec(config, parameters=None):
    values = (
        {"C": 1.0, "max_iter": 1000, "random_state": config.random_state}
        if config.task == "classification"
        else {"alpha": 1.0})
    if parameters is not None:
        if not isinstance(parameters, dict) or set(parameters) - set(values):
            raise ValueError("Invalid feature baseline parameters")
        values.update(parameters)

    strength = "C" if config.task == "classification" else "alpha"
    value = values[strength]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError("Baseline regularization must be finite")
    if value < 0 or (strength == "C" and value == 0):
        raise ValueError("Invalid baseline regularization")
    values[strength] = float(value)
    if config.task == "classification":
        if type(values["max_iter"]) is not int or values["max_iter"] < 1:
            raise ValueError("Baseline max_iter must be a positive integer")
        if type(values["random_state"]) is not int or not 0 <= values["random_state"] < 2 ** 32:
            raise ValueError("Invalid baseline random_state")

    return {"algorithm": "logistic_regression" if config.task == "classification" else "ridge",
        "parameters": values,"scaler": "standard_scaler",
        "prediction_threshold": FEATURE_BASELINE_THRESHOLD if config.task == "classification" else None
    }


def fit_feature_baseline(train_df, feature_columns, config, *, parameters=None):
    x_train, y_train = prepare_xy(train_df, feature_columns, config)
    spec = get_feature_baseline_spec(config, parameters)
    estimator_type = LogisticRegression if config.task == "classification" else Ridge
    model = make_pipeline(StandardScaler(), estimator_type(**spec["parameters"]))
    model.fit(x_train, y_train)

    return model

def predict_feature_baseline(model, data, feature_columns, config):
    if data.empty:
        raise ValueError("Prediction data cannot be empty")

    x = data[feature_columns]
    result = data.copy()

    if config.task == "classification":
        if list(model.classes_) != [0, 1]:
            raise ValueError("Expected model classes [0, 1]")

        result["probability_up"] = model.predict_proba(x)[:, 1]
        result["prediction"] = (result["probability_up"] >= FEATURE_BASELINE_THRESHOLD).astype(int)
    else:
        result["predicted_return"] = model.predict(x)

    return result
