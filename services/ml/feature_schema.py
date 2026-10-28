TECHNICAL_FEATURE_COLUMNS = [
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
    "volume_vs_avg"
]

NEWS_FEATURE_COLUMNS = [
    "news_count",
    "news_positive_mean",
    "news_negative_mean",
    "news_neutral_mean"
]

CALENDAR_FEATURE_COLUMNS = [
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos"
]

MARKET_FEATURE_COLUMNS = [
    "market_return_1d",
    "market_return_5d",
    "market_volatility_20",
    "market_price_vs_sma_20",
    "vix_level",
    "vix_change_1d"
]

SECTOR_FEATURE_COLUMNS = [
    "sector_return_1d",
    "sector_return_5d",
    "relative_strength_5d",
    "relative_strength_20d"
]

FEATURE_GROUPS = {
    "technical": TECHNICAL_FEATURE_COLUMNS,
    "news": NEWS_FEATURE_COLUMNS,
    "calendar": CALENDAR_FEATURE_COLUMNS,
    "market": MARKET_FEATURE_COLUMNS,
    "sector": SECTOR_FEATURE_COLUMNS
}

def get_feature_columns(feature_groups):
    if not feature_groups:
        raise ValueError("Feature groups cannot be empty")

    columns = []

    for group in feature_groups:
        if group not in FEATURE_GROUPS:
            raise ValueError(f"Feature group is not implemented: {group}")

        columns.extend(FEATURE_GROUPS[group])

    return columns