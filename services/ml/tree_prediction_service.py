import pandas as pd

from services.ml.data_preparation_service import (prepare_prediction_data,prepare_latest_prediction_data,validate_simulation_dates)
from services.ml.model_store_service import load_global_model, get_model_config
from services.ml.model_service import build_backtest_result
from services.ml.tree_model_service import predict
from services.simulator_service import (DEFAULT_CONFIDENCE_THRESHOLD,validate_confidence_threshold)

def _predict_with_loaded_model(
    data, model, metadata, config, start_date, end_date, *,
    ticker=None, prepare_news=True, progress_callback=None,
):
    feature_columns = metadata["feature_columns"]

    test_df = prepare_prediction_data(data,start_date,end_date,config,feature_columns,
        metadata["training_cutoff"],ticker=ticker,prepare_news=prepare_news,progress_callback=progress_callback)


    test_df = predict(model, test_df, feature_columns, config=config)
    if config.task == "classification":
        test_df["baseline_prediction"] = metadata["baseline_prediction"]

    return test_df


def predict_with_global_model(data, model_id, start_date, end_date, model_dir=None, *, ticker=None, prepare_news=True, progress_callback=None):
    validate_simulation_dates(start_date, end_date)

    if data.empty:
        raise ValueError("Empty stock data")

    model, metadata = load_global_model(model_id, model_dir)
    config = get_model_config(metadata)
    test_df = _predict_with_loaded_model(
        data, model, metadata, config, start_date, end_date,
        ticker=ticker, prepare_news=prepare_news,
        progress_callback=progress_callback,
    )
    return test_df, metadata


def simulate_with_global_model(data, model_id, start_date, end_date, initial_cash, fee_rate, model_dir=None,*, prepare_news= True, progress_callback=None,ticker=None, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):

    confidence_threshold = validate_confidence_threshold(confidence_threshold)
    validate_simulation_dates(start_date, end_date)

    if data.empty:
        raise ValueError("Empty stock data")

    model, metadata = load_global_model(model_id, model_dir)
    config = get_model_config(metadata)

    if config.task != "classification" or config.horizon != 1:
        raise ValueError(
            "Obecny symulator obsługuje klasyfikację z horyzontem 1 sesji.")
    results_df = _predict_with_loaded_model(
        data, model, metadata, config, start_date, end_date,
        ticker=ticker, prepare_news=prepare_news,
        progress_callback=progress_callback,
    )
    result = build_backtest_result(
        results_df, start_date, end_date, initial_cash, fee_rate, metadata["training_rows"],
        confidence_threshold=confidence_threshold,
    )

    result["model_type"] = "global"
    result["model_metadata"] = metadata
    return result


def predict_latest_with_global_model(data, model_id, model_dir=None, *, ticker=None, as_of=None, prepare_news=True, progress_callback=None):
    model, metadata = load_global_model(model_id, model_dir)
    config = get_model_config(metadata)
    feature_columns = metadata["feature_columns"]

    latest, prediction_window = prepare_latest_prediction_data(
        data,
        config,
        feature_columns,
        metadata["training_cutoff"],
        ticker=ticker,
        as_of=as_of,
        prepare_news=prepare_news,
        progress_callback=progress_callback)

    forecast_date = str(latest["date"].iloc[0])

    forecast = predict(model, latest, feature_columns, config=config).iloc[0]

    result = {
        "date": forecast_date,
        "task": config.task,
        "horizon": config.horizon,
        "model_metadata": metadata,
    }

    if prediction_window:
        result["execution_date"] = prediction_window["execution_date"]
        result["decision_at"] = prediction_window["decision_at"]
        result["as_of"] = prediction_window["as_of"]
        result["market_open"] = prediction_window["market_open"]

    if config.task == "classification":
        result["prediction"] = int(forecast["prediction"])
        result["probability_up"] = float(forecast["probability_up"])
    else:
        result["predicted_return"] = float(forecast["predicted_return"])

    if prediction_window and as_of is None:
        completed_at = pd.Timestamp.now(tz="UTC")
        market_open = pd.to_datetime(prediction_window["market_open"],utc=True)

        if completed_at >= market_open:
            raise ValueError(
                "The market opened while preparing the forecast; "
                "the upcoming-open decision is no longer available")

    return result