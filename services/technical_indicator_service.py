import pandas as pd

def calc_sma(data, period):
    data = data.copy()
    return data["close"].rolling(window=period).mean()

def calc_ema(data, period):
    data = data.copy()
    return data["close"].ewm(span=period, adjust=False).mean()

def calc_macd(data, slow_period = 26, fast_period = 12, signal_period = 9):
    data = data.copy()
    ema_fast = calc_ema(data, fast_period)
    ema_slow = calc_ema(data, slow_period)

    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calc_fibonacci_retracement(data, period=None):
    data = data.copy()
    if period is not None:
        data = data.tail(period)

    high = data["high"].max()
    low = data["low"].min()
    diff = high - low

    return {
        "23.6%": high - diff * 0.236,
        "38.2%": high - diff * 0.382,
        "50.0%": high - diff * 0.5,
        "61.8%": high - diff * 0.618,
    }


def calc_stochastic_oscillator(data, period=14):
    data = data.copy()
    lowest_low = data["low"].rolling(window=period).min()
    highest_high = data["high"].rolling(window=period).max()

    pk = (data["close"] - lowest_low) / (highest_high - lowest_low) * 100
    return pk

def calc_bollinger_bands(data, k=2, n=20):
    data = data.copy()
    mid_band = calc_sma(data, n)
    rolling_std = data["close"].rolling(window=n).std()
    upper_band = mid_band + k * rolling_std
    lower_band = mid_band - k * rolling_std
    return mid_band, upper_band, lower_band

def calc_rsi(data, period):
    data = data.copy()
    change = data["close"].diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    a = gains.rolling(window=period).mean()
    b = losses.rolling(window=period).mean()

    rs = a / b
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calc_adx(data, period):
    data = data.copy()

    upmove = data["high"].diff()
    downmove = -data["low"].diff()

    pdm = upmove.where((upmove > downmove) & (upmove > 0), 0)
    mdm = downmove.where((downmove > upmove) & (downmove > 0), 0)
    previous_close = data["close"].shift(1)
    tr1 = data["high"] - data["low"]
    tr2 = (data["high"] - previous_close).abs()
    tr3 = (data["low"] - previous_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    pdm_avg = pdm.rolling(window=period).mean()
    mdm_avg = mdm.rolling(window=period).mean()
    tr_avg = tr.rolling(window=period).mean()

    plus_di = 100*pdm_avg/tr_avg
    minus_di = 100*mdm_avg/tr_avg

    dx = 100 * abs(plus_di- minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx

def calc_standard_deviation(data, period):
    data = data.copy()
    return data["close"].rolling(window=period).std()

def calc_ichimoku_cloud(data, conversion_period=9, base_period=26, span_b_period=52, displacement=26):
    data = data.copy()

    conversion_high = data["high"].rolling(window=conversion_period).max()
    conversion_low = data["low"].rolling(window=conversion_period).min()
    tenkan_sen = (conversion_high + conversion_low) / 2

    base_high = data["high"].rolling(window=base_period).max()
    base_low = data["low"].rolling(window=base_period).min()
    kijun_sen = (base_high + base_low) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

    span_b_high = data["high"].rolling(window=span_b_period).max()
    span_b_low = data["low"].rolling(window=span_b_period).min()
    senkou_span_b = ((span_b_high + span_b_low) / 2).shift(displacement)

    chikou_span = data["close"].shift(-displacement)

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def calc_daily_return(data):
    data = data.copy()
    previous_close = data["close"].shift(1)
    dr = (data["close"] - previous_close) / previous_close
    return dr

def calc_volatility(data, period):
    data = data.copy()
    dr = calc_daily_return(data)
    volatility = dr.rolling(window=period).std()
    return volatility

def calc_momentum(data, period):
    data = data.copy()
    previous_close = data["close"].shift(period)
    momentum = data["close"] / previous_close - 1
    return momentum

def calc_atr(data, period):
    data = data.copy()
    previous_close = data["close"].shift(1)

    tr1 = data["high"] - data["low"]
    tr2 = (data["high"] - previous_close).abs()
    tr3 = (data["low"] - previous_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_all_indicators(data):
    result = data.copy()

    result["sma_12"] = calc_sma(data, 12)
    result["sma_26"] = calc_sma(data, 26)
    result["ema_12"] = calc_ema(data, 12)
    result["ema_26"] = calc_ema(data, 26)

    macd, signal, histogram = calc_macd(data)
    result["macd"] = macd
    result["macd_signal"] = signal
    result["macd_histogram"] = histogram

    result["rsi_14"] = calc_rsi(data, 14)
    result["adx_14"] = calc_adx(data, 14)
    result["atr_14"] = calc_atr(data, 14)
    result["daily_return"] = calc_daily_return(data)
    result["volatility_20"] = calc_volatility(data, 20)
    result["momentum_10"] = calc_momentum(data, 10)

    return result

def get_latest_indicators(data):
    indicators = calculate_all_indicators(data)
    latest = indicators.dropna().iloc[-1]

    return {
        "sma_12": round(latest["sma_12"], 2),
        "sma_26": round(latest["sma_26"], 2),
        "ema_12": round(latest["ema_12"], 2),
        "ema_26": round(latest["ema_26"], 2),
        "rsi_14": round(latest["rsi_14"], 2),
        "macd": round(latest["macd"], 2),
        "macd_signal": round(latest["macd_signal"], 2),
        "adx_14": round(latest["adx_14"], 2),
        "atr_14": round(latest["atr_14"], 2),
        "daily_return": round(latest["daily_return"] * 100, 2),
        "volatility_20": round(latest["volatility_20"] * 100, 2),
        "momentum_10": round(latest["momentum_10"] * 100, 2),
    }

