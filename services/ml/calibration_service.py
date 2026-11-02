import numpy as np
from datetime import date
from sklearn.linear_model import LogisticRegression

CALIBRATION_EPSILON = 1e-6

def prepare_calibration_features(predictions_df):
    if predictions_df.empty:
        raise ValueError("Predictions DataFrame is empty")
    probabilities = predictions_df["probability_up"].astype(float)

    if ((probabilities > 1) | (probabilities < 0) | probabilities.isna()).any(axis=None):
        raise ValueError("Probabilities need to be between 0 and 1 values (and cannot be null)")

    probabilities = probabilities.clip(lower=CALIBRATION_EPSILON, upper=1 - CALIBRATION_EPSILON)
    logit = np.log(probabilities/(1-probabilities))

    return logit.to_frame(name="logit_probability")

def fit_calibrator(calibration_predictions, config):
    if config.task != "classification":
        raise ValueError("Expected task classification")

    features = prepare_calibration_features(calibration_predictions)
    targets = calibration_predictions[config.target_column]

    if not targets.isin([0, 1]).all() or targets.nunique() != 2:
        raise ValueError("Targets must contain both classes: 0 and 1")

    calibrator = LogisticRegression(C=float("inf"), solver="lbfgs", max_iter=1000)
    calibrator.fit(features, targets)
    return calibrator

def apply_calibrator(calibration_predictions, calibrator, config):
    if config.task != "classification":
        raise ValueError("Expected task classification")
    features = prepare_calibration_features(calibration_predictions)
    result = calibration_predictions.copy()
    result["probability_up"] = calibrator.predict_proba(features)[:, 1]

    result["prediction"] = (result["probability_up"] >= 0.5).astype(int)

    return result


def serialize_calibrator(calibrator, calibration_df, config, start_date, end_date):

    if calibration_df.empty:
        raise ValueError("Calibration dataframe cannot be empty")

    calibration = {
        "method": "sigmoid_logit",
        "epsilon": CALIBRATION_EPSILON,
        "feature_column": "logit_probability",
        "classes": calibrator.classes_.tolist(),
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
        "iterations": int(calibrator.n_iter_[0]),
        "config": config.to_dict(),
        "start_date": start_date,
        "end_date": end_date,
        "rows": int(len(calibration_df)),
        "first_prediction_date": str(calibration_df["date"].min()),
        "last_prediction_date": str(calibration_df["date"].max()),
        "last_target_end_date": str(calibration_df["target_end_date"].max()),
        "tickers": sorted(calibration_df["ticker"].unique().tolist())
    }
    restore_calibrator(calibration, config)
    return calibration


def restore_calibrator(calibration, config):

    if not isinstance(calibration, dict) or config.task != "classification":
        raise ValueError("Invalid classification calibration metadata")
    expected = {"method": "sigmoid_logit", "epsilon": CALIBRATION_EPSILON, "feature_column": "logit_probability", "classes": [0, 1], "config": config.to_dict()}

    if any(calibration.get(name) != value for name, value in expected.items()):
        raise ValueError("Incompatible calibration policy or model configuration")

    for name in ("coefficient", "intercept"):
        value = calibration.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Invalid calibration {name}")

    for name in ("rows", "iterations"):
        value = calibration.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid calibration {name}")
    date_fields = ("start_date", "end_date", "first_prediction_date", "last_prediction_date", "last_target_end_date")
    for name in date_fields:
        value = calibration.get(name)
        try:
            if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                raise ValueError("Expected YYYY-MM-DD")
        except ValueError as error:
            raise ValueError(f"Invalid calibration {name}") from error
    start, end = calibration["start_date"], calibration["end_date"]
    first, last = calibration["first_prediction_date"], calibration["last_prediction_date"]
    if not (start <= first <= last < end) or not (last < calibration["last_target_end_date"] <= end):
        raise ValueError("Calibration rows or labels exceed the calibration period")
    tickers = calibration.get("tickers")
    if not isinstance(tickers, list) or not tickers or any(not isinstance(ticker, str) or not ticker.strip() for ticker in tickers):
        raise ValueError("Invalid calibration ticker list")
    calibrator = LogisticRegression(C=float("inf"), solver="lbfgs", max_iter=1000)
    calibrator.classes_ = np.array([0, 1])
    calibrator.coef_ = np.array([[calibration["coefficient"]]], dtype=float)
    calibrator.intercept_ = np.array([calibration["intercept"]], dtype=float)
    calibrator.n_features_in_ = 1
    calibrator.feature_names_in_ = np.array(["logit_probability"], dtype=object)
    calibrator.n_iter_ = np.array([calibration["iterations"]])
    return calibrator


def apply_saved_calibration(predictions_df, metadata, config):
    if "calibration" not in metadata:
        return predictions_df
    calibrator = restore_calibrator(metadata["calibration"], config)
    return apply_calibrator(predictions_df, calibrator, config)
