import math
from copy import deepcopy

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from services.ml.dataset_service import prepare_xy
from services.ml.mlp_model import NeuralNetwork


def prepare_loaders(train_df, feature_columns, config, *, validation_df=None, batch_size=256):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    x_train, y_train = prepare_xy(train_df, feature_columns, config)
    scaler = StandardScaler()
    x_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    validation_loader = None
    if validation_df is not None:
        x_validation, y_validation = prepare_xy(validation_df, feature_columns, config, require_both_classes=False)
        x_validation = torch.tensor(scaler.transform(x_validation), dtype=torch.float32)
        y_validation = torch.tensor(y_validation.to_numpy(), dtype=torch.float32).reshape(-1, 1)
        validation_loader = DataLoader(
            TensorDataset(x_validation, y_validation), batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader, scaler


def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    loss_sum = 0
    sample_count = 0
    for x_batch, y_batch in train_loader:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
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
            loss = loss_fn(pred, y_batch)
            loss_sum += loss.item() * x_batch.shape[0]
            sample_count += x_batch.shape[0]
    return loss_sum / sample_count


def train_model(model, train_loader, validation_loader=None, epochs=50, learning_rate=0.001, patience=5, *, config, progress_callback=None):
    if config.algorithm != "mlp":
        raise ValueError("Expected an MLP configuration")
    if type(epochs) is not int or epochs < 1 or type(patience) is not int or patience < 1:
        raise ValueError("epochs and patience must be positive integers")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    loss_fn = torch.nn.BCEWithLogitsLoss() if config.task == "classification" else torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    report = progress_callback or print
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        if not math.isfinite(train_loss):
            raise ValueError("train_loss must be finite")
        row = {"epoch": epoch, "train_loss": train_loss}
        if validation_loader is not None:
            validation_loss = validate_one_epoch(model, validation_loader, loss_fn)
            if not math.isfinite(validation_loss):
                raise ValueError("validation_loss must be finite")
            row["validation_loss"] = validation_loss
            report(f"Epoch {epoch}: train_loss={train_loss:.6f}, validation_loss={validation_loss:.6f}")
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            report(f"Epoch {epoch}: train_loss={train_loss:.6f}")
        history.append(row)
        if validation_loader is not None and epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def fit(train_df, feature_columns, config, *, validation_df=None, epochs=50, learning_rate=0.001, patience=5, batch_size=256, progress_callback=None):
    if config.algorithm != "mlp":
        raise ValueError("Expected an MLP configuration")
    torch.manual_seed(config.random_state)
    train_loader, validation_loader, scaler = prepare_loaders(
        train_df, feature_columns, config, validation_df=validation_df, batch_size=batch_size)
    model = NeuralNetwork(input_size=len(feature_columns))
    model, history, best_epoch = train_model(model, train_loader, validation_loader, epochs=epochs, learning_rate=learning_rate, patience=patience, config=config, progress_callback=progress_callback)
    details = {
        "architecture": "mlp",
        "input_size": len(feature_columns),
        "hidden_sizes": [64, 32],
        "torch_version": str(torch.__version__),
        "history": history,
        "best_epoch": best_epoch,
        "training_parameters": {
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss" if config.task == "classification" else "MSELoss",
            "max_epochs": epochs,
            "learning_rate": learning_rate,
            "patience": patience,
            "batch_size": batch_size
        }
    }
    return model, scaler, details


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
