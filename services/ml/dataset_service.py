import numpy as np

from services.technical_indicator_service import calculate_all_indicators

FEATURE_COLUMNS = [
    "price_vs_sma_12",
    "price_vs_sma_26",
    "price_vs_ema_12",
    "price_vs_ema_26",
    "macd_vs_price",
    "macd_signal_vs_price",
    "rsi_14",
    "adx_14",
    "atr_14_vs_price",
    "daily_return",
    "volatility_20",
    "momentum_10",
    "volume_vs_avg",
]

def prepare_features(data):
    data = data.copy()
    data = data.sort_values("date")
    data = calculate_all_indicators(data)

    data["price_vs_sma_12"] = data["close"] / data["sma_12"] - 1
    data["price_vs_sma_26"] = data["close"] / data["sma_26"] - 1
    data["price_vs_ema_12"] = data["close"] / data["ema_12"] - 1
    data["price_vs_ema_26"] = data["close"] / data["ema_26"] - 1

    data["atr_14_vs_price"] = data["atr_14"] / data["close"]
    data["macd_vs_price"] = data["macd"] / data["close"]
    data["macd_signal_vs_price"] = data["macd_signal"] / data["close"]
    avg_volume = data["volume"].rolling(window=20).mean()
    data["volume_vs_avg"] = data["volume"] / avg_volume

    return data


def prepare_dataset(data, horizon=1):
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Horizon must be a positive integer")

    data = prepare_features(data)

    data["next_close"] = data["close"].shift(-1)
    data["next_open"] = data["open"].shift(-1)
    data["next_date"] = data["date"].shift(-1)

    data["target_end_open"] = data["open"].shift(-(horizon+1))
    data["target_end_date"] = data["date"].shift(-(horizon+1))

    data = data.dropna(
        subset=FEATURE_COLUMNS + ["next_date", "next_open", "next_close", "target_end_open", "target_end_date"]).copy()

    data["target_return"] = data["target_end_open"] / data["next_open"] - 1
    data["target"] = (data["target_end_open"] > data["next_open"]).astype(int)

    return data

def prepare_latest_features(data, feature_columns):
    if data.empty:
        raise ValueError("Empty data")

    features_df = prepare_features(data)
    latest = features_df.iloc[[-1]].copy()

    if not np.isfinite(latest[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("No complete features for latest session")

    return latest

def prepare_xy(data, feature_columns, config):
    if data.empty:
        raise ValueError("Empty data")
    if not feature_columns:
        raise ValueError("Feature columns cannot be empty")

    required_columns = list(feature_columns) + [config.target_column]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    x = data[feature_columns]
    y = data[config.target_column]

    if (not np.isfinite(x.to_numpy(dtype=float)).all() or not np.isfinite(y.to_numpy(dtype=float)).all()):
        raise ValueError("X and Y must contain only finite values")

    if config.task == "classification":
        if not y.isin([0, 1]).all():
            raise ValueError("y must contain only 0 and 1")

        if y.nunique() != 2:
            raise ValueError("Training data must contain both target classes 0 and 1.")

    return x, y
