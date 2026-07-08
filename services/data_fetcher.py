import yfinance as yf

def fetch_stock_data(ticker, interval="1d", period="max", start_date=None):
    print("FETCH PARAMS:")
    print("ticker:", ticker)
    print("interval:", interval)
    print("period:", period)
    print("start_date:", start_date)
    if start_date is None:
        data = yf.download(ticker, period=period, interval=interval, multi_level_index=False, actions=True,progress=False)
    else:
        data = yf.download(ticker, start=start_date, interval=interval, multi_level_index=False, actions=True,progress=False)
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
    return data


def fetch_company_name(ticker):
    ticker = ticker.upper()
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return (info.get("longName") or info.get("shortName") or ticker)

    except Exception:
        return ticker


if __name__ == "__main__":
    df = fetch_stock_data("AAPL")
    print(df.head())
    print(df.tail())