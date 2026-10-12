import numpy as np
import pandas as pd


PRICE_COLUMNS = ["open", "high", "low", "close"]

PRICE_TOLERANCE = 1e-6
VALID_OHLC_SQL = """
    open > 0 AND open < 'Infinity'::real
    AND high > 0 AND high < 'Infinity'::real
    AND low > 0 AND low < 'Infinity'::real
    AND close > 0 AND close < 'Infinity'::real
    AND high + GREATEST(open, high, low, close) * 0.000001 >= GREATEST(open, close)
    AND low - GREATEST(open, high, low, close) * 0.000001 <= LEAST(open, close)
    AND high + GREATEST(open, high, low, close) * 0.000001 >= low
    AND NOT (volume = 0 AND open = high AND open = low AND open = close)
"""


def valid_ohlc_mask(data):
    if data.empty:
        return pd.Series(False, index=data.index, dtype=bool)
    prices = data[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid = pd.Series(np.isfinite(prices.to_numpy(dtype=float)).all(axis=1), index=data.index)
    tolerance = prices.max(axis=1) * PRICE_TOLERANCE
    if "volume" in data.columns:
        flat = prices.eq(prices["close"], axis=0).all(axis=1)
        valid &= ~(pd.to_numeric(data["volume"], errors="coerce").eq(0) & flat)
    return (
        valid
        & prices.gt(0).all(axis=1)
        & (prices["high"] + tolerance >= prices[["open", "close"]].max(axis=1))
        & (prices["low"] - tolerance <= prices[["open", "close"]].min(axis=1))
        & (prices["high"] + tolerance >= prices["low"])
    )


def filter_valid_ohlc(data):
    return data.loc[valid_ohlc_mask(data)].copy()


def mask_invalid_ohlc(data):
    result = data.copy()
    if not result.empty:
        result.loc[~valid_ohlc_mask(result), PRICE_COLUMNS] = np.nan
    return result
