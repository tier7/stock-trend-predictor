import numpy as np
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