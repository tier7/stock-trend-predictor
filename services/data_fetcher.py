import yfinance as yf
import pandas as pd
from services.stock_data_quality_service import filter_valid_ohlc

def fetch_stock_data(ticker, interval="1d", period="max", start_date=None):
    print("FETCH PARAMS:")
    print("ticker:", ticker)
    print("interval:", interval)
    print("period:", period)
    print("start_date:", start_date)

    try:
        if start_date is None:
            data = yf.download(ticker, period=period, interval=interval, multi_level_index=False, actions=True,progress=False, repair=True)
        else:
            data = yf.download(ticker, start=start_date, interval=interval, multi_level_index=False, actions=True,progress=False, repair=True)

    except Exception as error:
        print(f"Error while downloading data: {ticker}: {error}")
        return pd.DataFrame()

    if data is None or data.empty:
        print(F"No data found for {ticker}")
        return pd.DataFrame()

    data = data.reset_index()
    data["Ticker"] = ticker
    data["interval"] = interval

    data = data.rename(columns={
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Ticker": "ticker",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits"
    })

    data = data[["ticker", "interval", "date", "open", "high", "low", "close", "volume", "dividends", "stock_splits"]]
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")
    valid_data = filter_valid_ohlc(data)
    if len(valid_data) != len(data):
        print(f"Skipped {len(data) - len(valid_data)} invalid OHLC rows for {ticker}.")
    return valid_data


def fetch_company_name(ticker):
    ticker = ticker.upper()
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return (info.get("longName") or info.get("shortName") or ticker)

    except Exception as error:
        print(f"Could not fetch company name for {ticker}: {error}")
        return None


def fetch_market_metadata(ticker):
    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty")

    try:
        metadata = yf.Ticker(ticker).get_history_metadata()
    except Exception as error:
        raise ValueError(f"{ticker}: could not fetch market metadata") from error

    required_fields = (
        "exchangeName",
        "exchangeTimezoneName",
        "instrumentType",
        "currency")

    missing_fields = [
        field for field in required_fields
        if not metadata.get(field)
    ]

    if missing_fields:
        raise ValueError(f"{ticker}: missing market metadata: {missing_fields}")

    return {field: metadata[field] for field in required_fields}


if __name__ == "__main__":
    for ticker in ("AAPL", "ALE.WA", "MC.PA"):
        print(ticker, fetch_market_metadata(ticker))