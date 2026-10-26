import numpy as np

from services.technical_indicator_service import calculate_all_indicators
from services.ml.feature_schema import (TECHNICAL_FEATURE_COLUMNS as FEATURE_COLUMNS)
from services.ml.feature_schema import get_feature_columns
from services.ml.news_feature_service import attach_preopen_news_features

def prepare_features(data,*, feature_groups=("technical",), ticker=None, news_start=None, news_end=None):
    get_feature_columns(feature_groups)
    data = data.copy()
    data = data.sort_values("date")

    if "technical" in feature_groups:
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
    if "news" in feature_groups:
        if not ticker or news_start is None or news_end is None:
            raise ValueError("News features require ticker, news_start and news_end")

        data = attach_preopen_news_features(
            data,
            ticker,
            news_start,
            news_end,
        )

    return data


def prepare_dataset(data, horizon=1, *, feature_columns=None, feature_groups=("technical",), ticker=None, news_start=None, news_end=None):
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Horizon must be a positive integer")
    if feature_columns is None:
        feature_columns = get_feature_columns(feature_groups)

    data = prepare_features(data, feature_groups=feature_groups, ticker=ticker, news_start=news_start, news_end=news_end)

    data["next_close"] = data["close"].shift(-1)
    data["next_open"] = data["open"].shift(-1)
    data["next_date"] = data["date"].shift(-1)

    data["target_end_open"] = data["open"].shift(-(horizon+1))
    data["target_end_date"] = data["date"].shift(-(horizon+1))

    data = data.dropna(
        subset=list(feature_columns) + ["next_date", "next_open", "next_close", "target_end_open", "target_end_date"]).copy()

    data["target_return"] = data["target_end_open"] / data["next_open"] - 1
    data["target"] = (data["target_end_open"] > data["next_open"]).astype(int)

    return data

def prepare_latest_features(data, feature_columns, *, feature_groups=("technical",), ticker=None, news_start=None, news_end=None):
    if data.empty:
        raise ValueError("Empty data")

    features_df = prepare_features(data, feature_groups=feature_groups, ticker=ticker, news_start=news_start, news_end=news_end)

    latest = features_df.iloc[[-1]].copy()

    if not np.isfinite(latest[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("No complete features for latest session")

    return latest

def prepare_xy(data, feature_columns, config, *, require_both_classes=True):

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

        if require_both_classes and y.nunique() != 2:
            raise ValueError("Training data must contain both target classes 0 and 1.")

    return x, y
