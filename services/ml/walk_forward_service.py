from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from services.ml.dataset_service import prepare_xy
from services.ml.model_config import ModelConfig


RETRAIN_INTERVAL = 20

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


