from services.technical_indicator_service import calculate_all_indicators

FEATURE_COLUMNS = [
    "sma_12",
    "sma_26",
    "ema_12",
    "ema_26",
    "price_vs_sma_12",
    "price_vs_sma_26",
    "price_vs_ema_12",
    "price_vs_ema_26",
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

    data["price_vs_sma_12"] = data["close"] / data["sma_12"] - 1
    data["price_vs_sma_26"] = data["close"] / data["sma_26"] - 1
    data["price_vs_ema_12"] = data["close"] / data["ema_12"] - 1
    data["price_vs_ema_26"] = data["close"] / data["ema_26"] - 1

    data["next_close"] = data["close"].shift(-1)
    data["target"] = (data["next_close"] > data["close"]).astype(int)
    data["next_open"] = data["open"].shift(-1)
    data["next_date"] = data["date"].shift(-1)
    data = data.dropna(subset=FEATURE_COLUMNS + ["next_date", "next_open", "next_close", "target"])

    return data