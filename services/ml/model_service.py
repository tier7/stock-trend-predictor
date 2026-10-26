import pandas as pd
from sklearn.metrics import accuracy_score

from services.ml.dataset_service import FEATURE_COLUMNS, prepare_dataset
from services.ml.walk_forward_service import (
    RETRAIN_INTERVAL,
    build_date_blocks,
    predict,
    train
)
from services.simulator_service import (
    DEFAULT_CONFIDENCE_THRESHOLD, run_buy_and_hold, run_backtest, validate_confidence_threshold,
)
from services.ml.data_preparation_service import split_dataset_by_date
from services.ml.evaluation_service import evaluate


def train_and_predict(data, start_date, end_date, initial_cash, fee_rate, date_blocks=None, *, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    confidence_threshold = validate_confidence_threshold(confidence_threshold)
    if start_date >= end_date:
        raise ValueError("End date must be later than start date.")

    df = prepare_dataset(data)
    train_period_df, test_df = split_dataset_by_date(df, start_date, end_date)
    if len(train_period_df) < 100:
        raise ValueError("Not enough training data before selected date. Choose a later date.")

    if test_df.empty:
        raise ValueError("No test data from selected date. Choose an earlier date.")

    if date_blocks is None:
        date_blocks = build_date_blocks(test_df, RETRAIN_INTERVAL)

    if not date_blocks or any(not block_dates for block_dates in date_blocks):
        raise ValueError("Date blocks must contain non-empty blocks.")

    scheduled_dates = [date for block_dates in date_blocks for date in block_dates]
    if scheduled_dates != sorted(set(scheduled_dates)):
        raise ValueError("Date blocks must contain unique dates in chronological order.")
    if not test_df["date"].isin(scheduled_dates).all():
        raise ValueError("Date blocks do not cover all test dates.")

    prediction_blocks = []
    for block_dates in date_blocks:
        training_date = block_dates[0]
        current_train_df = df[
            (df["date"] < training_date)
            & (df["target_end_date"] <= training_date)
        ]
        current_test_df = test_df[test_df["date"].isin(block_dates)].copy()
        if current_test_df.empty:
            continue

        model = train(current_train_df, FEATURE_COLUMNS)
        current_test_df = predict(model, current_test_df, FEATURE_COLUMNS)
        current_test_df["baseline_prediction"] = int(current_train_df["target"].mode().iloc[0])
        prediction_blocks.append(current_test_df)

    results_df = pd.concat(prediction_blocks, ignore_index=True).sort_values("date")
    return build_backtest_result(
        results_df, start_date, end_date, initial_cash, fee_rate, len(train_period_df),
        confidence_threshold=confidence_threshold,
    )


def build_backtest_result(results_df, start_date, end_date, initial_cash, fee_rate, training_rows,confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    results_df = results_df.sort_values("date")
    model_eval = evaluate(results_df)
    baseline_acc = accuracy_score(results_df["target"], results_df["baseline_prediction"])

    result_df = results_df[
        ["date", "close", "next_date", "next_open", "next_close", "target_end_date",
         "target_end_open", "target", "prediction", "probability_up"]
    ].copy()
    result_df["is_correct"] = result_df["prediction"] == result_df["target"]
    result_df["date"] = result_df["date"].astype(str)
    result_df["target_end_date"] = result_df["target_end_date"].astype(str)
    result_df["next_date"] = result_df["next_date"].astype(str)
    prediction_records = result_df.to_dict(orient="records")

    modelsim = run_backtest(prediction_records, initial_cash, fee_rate, confidence_threshold)
    bahsim = run_buy_and_hold(prediction_records,initial_cash,fee_rate)

    return {
        "accuracy": float(model_eval["accuracy"]),
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "effective_test_start_date": str(result_df["date"].iloc[0]),
        "effective_test_end_date": str(result_df["target_end_date"].iloc[-1]),
        "training_rows": int(training_rows),
        "total_predictions": int(len(result_df)),
        "correct_predictions": int(result_df["is_correct"].sum()),
        "predictions": prediction_records,
        "baseline_accuracy": float(baseline_acc),
        "confusion_matrix": model_eval["confusion_matrix"],
        "precision": float(model_eval["precision"]),
        "recall": float(model_eval["recall"]),
        "f1": float(model_eval["f1"]),
        "model_summary": modelsim,
        "buyandhold_summary": bahsim,

    }
