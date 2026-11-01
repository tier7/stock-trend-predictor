from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from services.ml.dataset_service import prepare_xy
from services.ml.model_config import ModelConfig

def create_tree_model(config, *, n_estimators=100):
    if config.algorithm not in ("random_forest", "lightgbm"):
        raise ValueError("Expected a Random Forest or LightGBM configuration")

    if type(n_estimators) is not int or n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer")

    if config.algorithm == "random_forest":
        model_type = RandomForestClassifier if config.task == "classification" else RandomForestRegressor
    else:
        model_type = LGBMClassifier if config.task == "classification" else LGBMRegressor
    return model_type(n_estimators=n_estimators, random_state=config.random_state)


def fit(train_df, feature_columns, config=None, *, validation_df=None,n_estimators=100, patience=20):
    if config is None:
        config = ModelConfig(algorithm="random_forest", task="classification")
    if validation_df is not None and config.algorithm != "lightgbm":
        raise ValueError("Tree validation is supported for LightGBM only")
    if config.algorithm == "lightgbm" and (type(patience) is not int or patience < 1):
        raise ValueError("Patience must be a positive integer")

    x_train, y_train = prepare_xy(train_df, feature_columns, config)
    model = create_tree_model(config, n_estimators=n_estimators)
    if validation_df is None:
        model.fit(x_train, y_train)
    else:
        x_validation, y_validation = prepare_xy(validation_df, feature_columns, config, require_both_classes=False)
        metric = "binary_logloss" if config.task == "classification" else "l2"
        model.fit(
            x_train, y_train, eval_X=x_validation, eval_y=y_validation,
            eval_metric=metric, callbacks=[early_stopping(stopping_rounds=patience, first_metric_only=True)]
        )
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
