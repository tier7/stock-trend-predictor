from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from services.ml.model_config import ModelConfig

def create_tree_model(config, *, n_estimators=100):
    if config.algorithm not in ("random_forest", "lightgbm"):
        raise ValueError("Expected a Random Forest or LightGBM configuration")

    if type(n_estimators) is not int or n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer")

    if config.algorithm == "random_forest":
        if config.task == "regression":
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=config.random_state)
        elif config.task == "classification":
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=config.random_state)


    elif config.algorithm == "lightgbm":
        if config.task == "classification":
            model = LGBMClassifier(n_estimators=n_estimators, random_state=config.random_state)
        elif config.task == "regression":
            model = LGBMRegressor(n_estimators=n_estimators, random_state=config.random_state)

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
