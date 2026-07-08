import pandas as pd

def resample_data(data,interval):
    if interval == "1d":
        return data.copy()

    name_map = {"1w": "W-FRI", "1mo":"ME"}

    if interval not in name_map:
        return data.copy()

    data["date"] = pd.to_datetime(data["date"])
    data=data.set_index("date")

    resampled = data.resample(name_map[interval]).agg(
        {"ticker": "last",
         "interval": "last",
         "open": "first",
         "high": "max",
         "low": "min",
         "close": "last",
         "volume": "sum",
         "dividend": "sum",
        "stock_splits": "sum"}
    )

    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    resampled = resampled.reset_index()
    resampled["date"] = resampled["date"].dt.strftime("%Y-%m-%d")
    resampled["interval"] = interval

    return resampled