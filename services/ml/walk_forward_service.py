from services.ml.tree_model_service import create_tree_model, predict
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

    model = create_tree_model(config)

    model.fit(x_train, y_train)
    return model



