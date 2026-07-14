from services.technical_indicator_service import calculate_all_indicators

FEATURE_COLUMNS = [
    "sma_12",
    "sma_26",
    "ema_12",
    "ema_26",
    "macd",
    "macd_signal",
    "rsi_14",
    "adx_14",
    "atr_14",
    "daily_return",
    "volatility_20",
    "momentum_10",
    "volume",
]

def prepare_dataset(data):
    data = data.copy()
    data = data.sort_values("date")
    data = calculate_all_indicators(data)

    data["next_close"] = data["close"].shift(-1)
    data["target"] = (data["next_close"] > data["close"]).astype(int)

    data = data.dropna(subset=FEATURE_COLUMNS + ["next_close", "target"])

    return data