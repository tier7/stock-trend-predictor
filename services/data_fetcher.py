import yfinance as yf

def fetch_stock_data(ticker):
    data = yf.download(ticker, period="max", interval="1d", multi_level_index=False, actions=True, progress=False)
    data = data.reset_index()
    data["Ticker"] = ticker

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
    data = data[["ticker", "date", "open", "high", "low", "close", "volume", "dividends", "stock_splits"]]
    return data

df = fetch_stock_data("AAPL")
print(df.head())
print(df.tail())