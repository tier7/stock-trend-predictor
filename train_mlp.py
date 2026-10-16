import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader

from services.ml.dataset_service import FEATURE_COLUMNS
from services.ml.mlp_model import NeuralNetwork
from services.ml.mlp_service import prepare_training_data,prepare_tensors,create_data_loaders,train_model, calculate_baseline_loss, evaluate_model
from services.ml.global_forest_service import train_tickers
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from services.ml.model_store_service import save_torch_model

if __name__ == "__main__":
    torch.manual_seed(26)

    tickers = train_tickers
    validation_start = "2025-01-01"
    training_cutoff = "2026-01-01"

    data, train_df, validation_df = prepare_training_data(tickers, validation_start, training_cutoff)
    x_train, y_train, x_validation, y_validation, scaler = prepare_tensors(train_df, validation_df, FEATURE_COLUMNS)
    train_loader, validation_loader = create_data_loaders(x_train, y_train, x_validation, y_validation)

    model = NeuralNetwork(input_size=len(FEATURE_COLUMNS))
    baseline_loss = calculate_baseline_loss(train_df, validation_df)
    model, history, best_epoch = train_model(model, train_loader, validation_loader)

    best_validation_loss = min(row["validation_loss"] for row in history)
    evaluation = evaluate_model(model, validation_df, FEATURE_COLUMNS, scaler)
    print(f"Baseline loss: {baseline_loss:.6f}")
    print(f"MLP validation loss: {best_validation_loss:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation accuracy: {evaluation['accuracy']:.2%}")
    print(f"Validation log loss: {evaluation['log_loss']:.6f}")

    final_train_df = data[(data["date"] < training_cutoff)& (data["target_end_date"] <= training_cutoff)].copy()
    final_scaler = StandardScaler()
    x_final_scaled = final_scaler.fit_transform(final_train_df[FEATURE_COLUMNS])
    x_final = torch.tensor(x_final_scaled, dtype=torch.float32)
    y_final = torch.tensor(final_train_df["target"].to_numpy(), dtype=torch.float32).reshape(-1,1)
    final_dataset = TensorDataset(x_final, y_final)
    final_loader = DataLoader(final_dataset, batch_size=256, shuffle=True)

    torch.manual_seed(26)
    final_model = NeuralNetwork(input_size=len(FEATURE_COLUMNS))
    final_model, final_history, _ = train_model(final_model, final_loader, epochs=best_epoch)

    metadata = {
        "architecture": "mlp",
        "input_size": len(FEATURE_COLUMNS),
        "hidden_sizes": [64, 32],
        "feature_columns": list(FEATURE_COLUMNS),
        "training_cutoff": training_cutoff,
        "train_tickers": list(tickers),
        "training_rows": len(final_train_df),
        "epochs": best_epoch,
        "selection_history": history,
        "final_history": final_history,
    }

    saved_metadata = save_torch_model(
        final_model,
        final_scaler,
        metadata,
    )

    print(f"Saved model: {saved_metadata['model_id']}")

