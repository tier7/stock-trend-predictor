from services.data_fetcher import fetch_stock_data
from services.db import get_connection, save_stock_data
from datetime import datetime, timedelta


def get_next_day(datestr):
    date = datetime.strptime(datestr, "%Y-%m-%d")
    next_day = date + timedelta(days=1)
    return next_day.strftime("%Y-%m-%d")

def ensure_stock_data(ticker, interval="1d"):
    conn = get_connection()
    cur = conn.cursor()
    ticker = ticker.upper()
    cur.execute("SELECT date FROM stock_prices WHERE ticker = ? AND interval = ? ORDER BY date DESC LIMIT 1", (ticker,interval))
    row = cur.fetchone()
    conn.close()

    if row is None:
        print(f"No stock data for {ticker}. Downloading max period data")
        data = fetch_stock_data(ticker=ticker, interval=interval, period="max")
        if data.empty:
            print(f"No valid data found for {ticker}")
            return False
        save_stock_data(data)
        print(f"Saved {len(data)} records for {ticker}")

        return True

    last_date = row[0]
    if last_date == datetime.today().strftime('%Y-%m-%d'):
        print(f"Stock data for {ticker} already up to date")
        return True

    next_day = get_next_day(last_date)
    print(f"Updating {ticker} from {next_day}")
    data = fetch_stock_data(ticker=ticker,interval=interval, start_date=next_day)

    if data.empty:
        print(f"No new data for {ticker}")
        return True

    save_stock_data(data)
    print(f"Saved {len(data)} records for {ticker}")
    return True


if __name__ == "__main__":
    ensure_stock_data("AAPL")