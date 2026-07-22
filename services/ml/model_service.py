from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
from services.ml.dataset_service import FEATURE_COLUMNS, prepare_dataset
from services.simulator_service import run_buy_and_hold, run_backtest


RETRAIN_INTERVAL = 20

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

    predictions = []
    probabilities_up = []
    baseline_predictions = []
    for test_position, row_data in enumerate(test_df.iterrows()):
        row_index, test_row = row_data
        if test_position % RETRAIN_INTERVAL == 0:
            current_train_df = df[df["date"] < test_row["date"]]
            x_train = current_train_df[FEATURE_COLUMNS]
            y_train = current_train_df["target"]
            baseline_prediction = int(y_train.mode().iloc[0])
            model = RandomForestClassifier(n_estimators=100,random_state=26)
            model.fit(x_train, y_train)
        current_features = test_row[FEATURE_COLUMNS].to_frame().T.astype(float)
        baseline_predictions.append(baseline_prediction)
        prediction = model.predict(current_features)[0]
        probability_up = model.predict_proba(current_features)[0][1]
        predictions.append(prediction)
        probabilities_up.append(probability_up)

    y_test = test_df["target"]

    acc = accuracy_score(y_test, predictions)
    baseline_acc = accuracy_score(y_test, baseline_predictions)

    tn, fp, fn, tp = confusion_matrix(y_test,predictions,labels=[0, 1]).ravel()
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    result_df = test_df[["date", "close", "next_date", "next_open", "next_close", "target"]].copy()
    result_df["prediction"] = predictions
    result_df["probability_up"] = probabilities_up
    result_df["is_correct"] = result_df["prediction"] == result_df["target"]
    result_df["date"] = result_df["date"].astype(str)
    result_df["next_date"] = result_df["next_date"].astype(str)
    prediction_records = result_df.to_dict(orient="records")

    modelsim = run_backtest(prediction_records)
    bahsim = run_buy_and_hold(prediction_records)

    return {
        "accuracy": float(acc),
        "requested_start_date": start_date,
        "effective_test_start_date": str(result_df["date"].iloc[0]),
        "training_rows": int(len(train_df)),
        "total_predictions": int(len(result_df)),
        "correct_predictions": int(result_df["is_correct"].sum()),
        "predictions": prediction_records,
        "baseline_accuracy": float(baseline_acc),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "model_summary": modelsim,
        "buyandhold_summary": bahsim,

    }
