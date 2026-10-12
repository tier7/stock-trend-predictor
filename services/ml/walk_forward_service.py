from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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


def train(train_df, feature_columns):
    missing_features = []
    if train_df.empty:
        raise ValueError("Empty train dataframe")

    for feature in feature_columns:
        if feature not in train_df.columns:
            missing_features.append(feature)

    if missing_features:
        raise ValueError("The following features are missing: {}".format(missing_features))

    if train_df[feature_columns].isna().any().any():
        missing_mask = train_df[feature_columns].isna().any()
        missing_columns = missing_mask.index[missing_mask].tolist()
        raise ValueError("Missing values in columns: {}".format(missing_columns))

    if "target" not in train_df.columns:
        raise ValueError("Target column is missing")

    if not train_df["target"].isin([0, 1]).all():
        raise ValueError("Not all target are 0 or 1.")

    if train_df["target"].nunique() != 2:
        raise ValueError("Training data must contain both target classes 0 and 1.")

    x_train = train_df[feature_columns]
    y_train = train_df["target"]
    model = RandomForestClassifier(n_estimators=100, random_state=26)
    model.fit(x_train, y_train)

    return model


def evaluate(results_df):
    if results_df.empty:
        raise ValueError("Empty test dataframe")

    y_test = results_df["target"]
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
