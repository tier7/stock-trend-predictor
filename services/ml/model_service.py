from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from services.ml.dataset_service import FEATURE_COLUMNS, prepare_dataset



def split_dataset_by_date(data, start_date):
    train_df = data[data["date"] < start_date]
    test_df =  data[data["date"] >= start_date]
    return train_df, test_df

def train_and_predict(data,start_date):
    df = prepare_dataset(data)
    train_df, test_df = split_dataset_by_date(df, start_date)
    if len(train_df) < 100:
        raise ValueError("Not enough training data before selected date. Choose a later date.")

    if len(test_df) < 1:
        raise ValueError("No test data from selected date. Choose an earlier date.")

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]

    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]

    model = RandomForestClassifier(n_estimators=100, random_state=26)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions)

    probability_up = model.predict_proba(x_test)[:, 1]

    result_df = test_df[["date", "close", "target"]].copy()
    result_df["prediction"] = predictions
    result_df["probability_up"] = probability_up
    result_df["is_correct"] = result_df["prediction"] == result_df["target"]

    return {
        "accuracy": float(acc),
        "total_predictions": int(len(result_df)),
        "correct_predictions": int(result_df["is_correct"].sum()),
        "predictions": result_df.to_dict(orient="records"),
    }
