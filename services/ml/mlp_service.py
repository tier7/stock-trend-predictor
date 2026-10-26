import math
from copy import deepcopy

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from services.ml.dataset_service import prepare_xy
from services.ml.mlp_model import NeuralNetwork
from services.ml.model_store_service import load_torch_model, save_torch_model
from services.ml.model_service import build_backtest_result
from services.simulator_service import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    validate_confidence_threshold,
)
from services.ml.model_config import ModelConfig
from services.ml.feature_schema import get_feature_columns
from services.ml.news_feature_service import normalize_training_news_range, NEWS_FEATURE_POLICY
from services.ml.data_preparation_service import (connect_data, split_dataset_by_date, validate_simulation_dates,
                                                  validate_training_cutoff, prepare_prediction_data)
import pandas as pd
from services.ml.data_preparation_service import prepare_latest_prediction_data
from datetime import datetime, timezone
from services.ml.evaluation_service import evaluate, calculate_baseline_loss

def prepare_training_data(tickers, validation_start, training_cutoff, config, *, news_start=None, news_end=None, prepare_news=True):
    validate_training_cutoff(training_cutoff)
    validate_simulation_dates(validation_start, training_cutoff)

    if config.algorithm != "mlp":
        raise ValueError("Invalid algorithm for mlp function")
    if config.scope != "global":
        raise ValueError("Invalid scope for mlp function")

    if "news" in config.feature_groups:
        news_start, news_end = normalize_training_news_range(news_start,news_end,training_cutoff)

    feature_columns = get_feature_columns(config.feature_groups)
    data = connect_data(tickers, horizon=config.horizon, feature_columns=feature_columns, feature_groups=config.feature_groups,news_start=news_start, news_end=news_end, prepare_news=prepare_news)
    train_df, validation_df = split_dataset_by_date(data, validation_start, training_cutoff)

    if train_df.empty or validation_df.empty:
        raise ValueError("train_df and validation_df must not be empty")

    print(f"Tickers: {len(tickers)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    return data, train_df, validation_df

def prepare_tensors(train_df, validation_df, feature_columns, config):

    x_train, y_train = prepare_xy(train_df, feature_columns, config)
    x_validate, y_validate = prepare_xy(validation_df,feature_columns,config,require_both_classes=False)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_validate = scaler.transform(x_validate)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_validate = torch.tensor(x_validate, dtype=torch.float32)

    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    y_validate = torch.tensor(y_validate.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    return x_train, y_train, x_validate, y_validate, scaler

def create_data_loaders(x_train, y_train, x_validate, y_validate, batch_size=256):
    train_dataset = TensorDataset(x_train, y_train)
    validation_dataset = TensorDataset(x_validate, y_validate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader


def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    loss_sum = 0
    sample_count = 0
    for x_batch, y_batch in train_loader:
        pred = model(x_batch)
        loss=loss_fn(pred,y_batch)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x_batch.shape[0]
        sample_count += x_batch.shape[0]
        optimizer.zero_grad()
    return loss_sum / sample_count

def validate_one_epoch(model, validation_loader, loss_fn):
    model.eval()
    loss_sum = 0
    sample_count = 0
    with torch.no_grad():
        for x_batch, y_batch in validation_loader:
            pred = model(x_batch)
            loss = loss_fn(pred,y_batch)
            loss_sum += loss.item() * x_batch.shape[0]
            sample_count += x_batch.shape[0]

    return loss_sum / sample_count

def train_model(model, train_loader, validation_loader=None, epochs=50, learning_rate=0.001, patience=5,*,config):
    if config.algorithm != "mlp":
        raise ValueError("Expected an MLP configuration")
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive")

    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    if config.task == "classification":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs+1):

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        if not math.isfinite(train_loss):
            raise ValueError("train_loss must be finite")

        if validation_loader is not None:
            validation_loss = validate_one_epoch(model, validation_loader, loss_fn)

            if not math.isfinite(validation_loss):
                raise ValueError("validation_loss must be finite")

            history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
            print(f"Epoch {epoch}: "f"train_loss={train_loss:.6f}, "f"validation_loss={validation_loss:.6f}")
            if validation_loss < best_loss:
                epochs_without_improvement = 0
                best_loss = validation_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        else:
            history.append({"epoch": epoch, "train_loss": train_loss})
            print(f"Epoch {epoch}: "f"train_loss={train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, history, best_epoch



def predict(model, data, feature_columns, scaler, config):
    if data.empty:
        raise ValueError("data cannot be empty")
    if config.algorithm != "mlp":
        raise ValueError("Expected an MLP configuration")
    result = data.copy()

    attributes = scaler.transform(data[feature_columns])
    attributes = torch.tensor(attributes, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        outputs = model(attributes)
        if config.task == "classification":
            probabilities = torch.sigmoid(outputs)
            result["probability_up"] = probabilities.cpu().numpy().reshape(-1)
            result["prediction"] = (result["probability_up"] >= 0.5).astype(int)
        if config.task == "regression":
            result["predicted_return"] =  outputs.cpu().numpy().reshape(-1)

    return result


def _predict_with_loaded_model(
    data, model, scaler, metadata, config, start_date, end_date, *,
    ticker=None, prepare_news=True, progress_callback=None,
):
    feature_columns = metadata["feature_columns"]
    test_df = prepare_prediction_data(data,start_date,end_date,config,feature_columns,
        metadata["training_cutoff"],ticker=ticker,prepare_news=prepare_news,progress_callback=progress_callback)

    if config.task == "classification":
        test_df["baseline_prediction"] = metadata["baseline_prediction"]

    return predict(model, test_df, feature_columns, scaler, config)


def predict_with_mlp_model(data,model_id,start_date,end_date,model_dir=None, *,ticker=None,prepare_news=True,progress_callback=None,):
    validate_simulation_dates(start_date, end_date)

    if data.empty:
        raise ValueError("Empty stock data")

    model, scaler, metadata = load_torch_model(model_id, model_dir)
    config = ModelConfig.from_dict(metadata["config"])
    test_df = _predict_with_loaded_model(
        data, model, scaler, metadata, config, start_date, end_date,
        ticker=ticker, prepare_news=prepare_news,
        progress_callback=progress_callback,
    )
    return test_df, metadata


def simulate_with_mlp_model(data, model_id, start_date, end_date, initial_cash, fee_rate, model_dir = None, *, ticker=None, prepare_news=True, progress_callback=None, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    validate_simulation_dates(start_date, end_date)
    confidence_threshold = validate_confidence_threshold(confidence_threshold)

    if data.empty:
        raise ValueError("Empty stock data")

    model, scaler, metadata = load_torch_model(model_id, model_dir)
    config = ModelConfig.from_dict(metadata["config"])

    if config.task != "classification" or config.horizon != 1:
        raise ValueError("Simulation supports classification with horizon=1 only")

    test_df = _predict_with_loaded_model(
        data, model, scaler, metadata, config, start_date, end_date,
        ticker=ticker, prepare_news=prepare_news,
        progress_callback=progress_callback,
    )
    result = build_backtest_result(test_df, start_date, end_date, initial_cash, fee_rate,metadata["training_rows"],confidence_threshold=confidence_threshold)
    result["model_type"] = "mlp"
    result["model_metadata"] = metadata
    return result

def predict_latest_with_mlp_model(data, model_id, model_dir=None, *,ticker=None, as_of=None, prepare_news=True, progress_callback=None):
    model, scaler, metadata = load_torch_model(model_id, model_dir)
    config = ModelConfig.from_dict(metadata["config"])
    feature_columns = metadata["feature_columns"]

    latest, news_window = prepare_latest_prediction_data(
        data,
        config,
        feature_columns,
        metadata["training_cutoff"],
        ticker=ticker,
        as_of=as_of,
        prepare_news=prepare_news,
        progress_callback=progress_callback)

    forecast_date = str(latest["date"].iloc[0])

    forecast = predict(model, latest, feature_columns, scaler, config).iloc[0]

    result = {
        "date": forecast_date,
        "task": config.task,
        "horizon": config.horizon,
        "model_metadata": metadata,
    }

    if news_window:
        result["execution_date"] = news_window["execution_date"]
        result["decision_at"] = news_window["decision_at"]
        result["as_of"] = news_window["news_end"]
        result["market_open"] = news_window["market_open"]

    if config.task == "classification":
        result["prediction"] = int(forecast["prediction"])
        result["probability_up"] = float(forecast["probability_up"])
    else:
        result["predicted_return"] = float(forecast["predicted_return"])

    if news_window and as_of is None:
        completed_at = pd.Timestamp.now(tz="UTC")
        market_open = pd.to_datetime(news_window["market_open"], utc=True)

        if completed_at >= market_open:
            raise ValueError(
                "The market opened while preparing the forecast; "
                "the upcoming-open decision is no longer available")

    return result

def train_mlp_model(tickers,validation_start,training_cutoff,*,config,epochs=50,learning_rate=0.001,patience=5,batch_size=256,news_start=None,news_end=None, prepare_news=True,model_dir=None):
    feature_columns = get_feature_columns(config.feature_groups)
    torch.manual_seed(config.random_state)

    if "news" in config.feature_groups:
        news_start, news_end = normalize_training_news_range(
            news_start,
            news_end,
            training_cutoff)

    data, train_df, validation_df = prepare_training_data(tickers, validation_start, training_cutoff, config,         news_start=news_start,news_end=news_end,prepare_news=prepare_news)
    final_train_df = data[
        (data["date"] < training_cutoff)
        & (data["target_end_date"] <= training_cutoff)
    ].copy()
    missing_tickers = sorted(set(tickers) - set(final_train_df["ticker"]))
    if missing_tickers:
        raise ValueError(
            f"Brak próbek treningowych przed wybraną datą: {missing_tickers}."
        )

    x_train, y_train, x_validation, y_validation, scaler = prepare_tensors(train_df, validation_df, feature_columns, config)
    train_loader, validation_loader = create_data_loaders(x_train, y_train, x_validation, y_validation, batch_size=batch_size)

    model = NeuralNetwork(input_size=len(feature_columns))
    baseline_loss = calculate_baseline_loss(train_df, validation_df, config)
    model, history, best_epoch = train_model(model, train_loader, validation_loader, config=config, epochs=epochs, learning_rate=learning_rate, patience=patience)

    best_validation_loss = min(row["validation_loss"] for row in history)
    validation_results = predict(model,validation_df,feature_columns,scaler,config)
    evaluation = evaluate(validation_results, config)
    print(f"Baseline loss: {baseline_loss:.6f}")
    print(f"MLP validation loss: {best_validation_loss:.6f}")
    print(f"Best epoch: {best_epoch}")

    if config.task == "classification":
        print(f"Validation accuracy: {evaluation['accuracy']:.2%}")
        print(f"Validation log loss: {evaluation['log_loss']:.6f}")
    else:
        print(f"Validation MAE: {evaluation['mae']:.6f}")
        print(f"Validation RMSE: {evaluation['rmse']:.6f}")

    final_scaler = StandardScaler()
    x_final_scaled = final_scaler.fit_transform(final_train_df[feature_columns])
    x_final = torch.tensor(x_final_scaled, dtype=torch.float32)
    y_final = torch.tensor(final_train_df[config.target_column].to_numpy(), dtype=torch.float32).reshape(-1,1)
    final_dataset = TensorDataset(x_final, y_final)
    final_loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=True)

    torch.manual_seed(config.random_state)
    final_model = NeuralNetwork(input_size=len(feature_columns))
    final_model, final_history, _ = train_model(final_model, final_loader, epochs=best_epoch, learning_rate = learning_rate, config=config)

    metadata = {
        "architecture": "mlp",
        "input_size": len(feature_columns),
        "hidden_sizes": [64, 32],
        "feature_columns": list(feature_columns),
        "training_cutoff": training_cutoff,
        "train_tickers": list(tickers),
        "training_rows": len(final_train_df),
        "epochs": best_epoch,
        "selection_history": history,
        "final_history": final_history,
        "config": config.to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": str(torch.__version__),
        "validation_start": validation_start,
        "training_start_date": str(final_train_df["date"].min()),
        "training_end_date": str(final_train_df["date"].max()),
        "last_target_end_date": str(final_train_df["target_end_date"].max()),
        "training_parameters": {
            "optimizer": "Adam",
            "loss": ("BCEWithLogitsLoss" if config.task == "classification" else "MSELoss"),
            "max_epochs": epochs,
            "learning_rate": learning_rate,
            "patience": patience,
            "batch_size": batch_size,
        },
        "validation_rows": len(validation_df),
        "best_validation_loss": float(best_validation_loss),
        "baseline_validation_loss": float(baseline_loss),
    }

    if "news" in config.feature_groups:
        metadata["news_policy"] = NEWS_FEATURE_POLICY.copy()
        metadata["news_start"] = news_start
        metadata["news_end"] = news_end

    if config.task == "classification":
        metadata["baseline_prediction"] = int(final_train_df["target"].mode().iloc[0])
    else:
        metadata["baseline_return"] = float(final_train_df["target_return"].mean())

    saved_metadata = save_torch_model(
        final_model,
        final_scaler,
        metadata,
        model_dir=model_dir
    )

    return saved_metadata
