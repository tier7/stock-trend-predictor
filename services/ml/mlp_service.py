import math
from copy import deepcopy

from sklearn.metrics import log_loss, accuracy_score

from services.ml.global_forest_service import connect_data, validate_simulation_dates
from services.ml.walk_forward_service import split_dataset_by_date
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from services.ml.dataset_service import FEATURE_COLUMNS, prepare_dataset
from services.ml.mlp_model import NeuralNetwork
from services.ml.model_store_service import load_torch_model
from services.ml.model_service import build_backtest_result
from services.simulator_service import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    validate_confidence_threshold,
)

def prepare_training_data(tickers, validation_start, training_cutoff):
    if training_cutoff <= validation_start:
        raise ValueError("training cutoff must be greater than validation_start")

    data = connect_data(tickers)
    train_df, validation_df = split_dataset_by_date(data, validation_start, training_cutoff)

    if train_df.empty or validation_df.empty:
        raise ValueError("train_df and validation_df must not be empty")

    print(f"Tickers: {len(tickers)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    return data, train_df, validation_df

def prepare_tensors(train_df, validation_df, feature_columns):
    x_train = train_df[feature_columns]
    x_validate = validation_df[feature_columns]
    y_train = train_df["target"]
    y_validate = validation_df["target"]

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

def train_model(model, train_loader, validation_loader=None, epochs=50, learning_rate=0.001, patience=5,):
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive")

    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    loss_fn = torch.nn.BCEWithLogitsLoss()
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

def calculate_baseline_loss(train_df, validation_df):
    baseline = train_df["target"].mean()
    probabilities = [baseline] * len(validation_df)

    loss = log_loss(validation_df["target"], probabilities, labels=[0,1])
    return loss


def predict_probabilities(model, data, feature_columns, scaler):
    attributes = data[feature_columns]
    attributes = scaler.transform(attributes)
    attributes = torch.tensor(attributes, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(attributes))
        return probabilities.cpu().numpy().reshape(-1)

def evaluate_model(model, data, feature_columns, scaler):
    probabilities = predict_probabilities(model, data, feature_columns, scaler)
    predictions = (probabilities >= 0.5).astype(int)
    acc_score = accuracy_score(data["target"], predictions)
    loss = log_loss(data["target"], probabilities, labels=[0, 1])

    return {
    "accuracy": acc_score,
    "log_loss": loss,
    "predictions": predictions,
    "probabilities": probabilities,
}


def simulate_with_mlp_model(data, model_id, start_date, end_date, initial_cash, fee_rate,*, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    validate_simulation_dates(start_date, end_date)
    confidence_threshold = validate_confidence_threshold(confidence_threshold)
    model = NeuralNetwork(input_size=len(FEATURE_COLUMNS))
    model, scaler, metadata = load_torch_model(model, model_id)

    if start_date < metadata["training_cutoff"]:
        raise ValueError("start_date must be greater than or equal to training_cutoff")

    prepared = prepare_dataset(data)
    _, test_df = split_dataset_by_date(prepared, start_date, end_date)

    if test_df.empty:
        raise ValueError("test_df is empty")
    test_df = test_df.copy()

    test_df["probability_up"] = predict_probabilities(model, test_df, FEATURE_COLUMNS, scaler)
    test_df["prediction"] = (test_df["probability_up"] >= 0.5).astype(int)
    test_df["baseline_prediction"] = metadata["baseline_prediction"]

    result = build_backtest_result(test_df, start_date, end_date, initial_cash, fee_rate,metadata["training_rows"],confidence_threshold=confidence_threshold)
    result["model_type"] = "mlp"
    result["model_metadata"] = metadata
    return result

