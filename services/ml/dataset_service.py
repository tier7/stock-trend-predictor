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
    """Calculate features without requiring future prices or target labels."""
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


def prepare_dataset(data):
    data = prepare_features(data)

    data["next_close"] = data["close"].shift(-1)
    data["next_open"] = data["open"].shift(-1)
    data["next_date"] = data["date"].shift(-1)

    data["target_end_open"] = data["open"].shift(-2)
    data["target_end_date"] = data["date"].shift(-2)

    data = data.dropna(
        subset=FEATURE_COLUMNS + ["next_date", "next_open", "next_close", "target_end_open", "target_end_date"]).copy()

    data["target"] = (data["target_end_open"] > data["next_open"]).astype(int)

    return data
