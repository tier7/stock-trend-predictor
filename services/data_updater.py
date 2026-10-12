from services.data_fetcher import fetch_stock_data
from services.db import get_connection, save_stock_data
from datetime import datetime, timedelta
from services.stock_data_quality_service import VALID_OHLC_SQL


REFRESH_LOOKBACK_DAYS = 30


def get_next_day(datestr):
    date = datetime.strptime(datestr, "%Y-%m-%d")
    next_day = date + timedelta(days=1)
    return next_day.strftime("%Y-%m-%d")

def ensure_stock_data(ticker, interval="1d"):
    conn = get_connection()
    cur = conn.cursor()
    ticker = ticker.upper()
    cur.execute(
        f"SELECT MAX(date), MAX(date) FILTER (WHERE {VALID_OHLC_SQL}) "
        "FROM stock_prices WHERE ticker = %s AND interval = %s", (ticker, interval),
    )
    row = cur.fetchone()
    conn.close()

    if row is None or row[0] is None:
        print(f"No stock data for {ticker}. Downloading max period data")
        data = fetch_stock_data(ticker=ticker, interval=interval, period="max")
        if data.empty:
            print(f"No valid data found for {ticker}")
            return False
        save_stock_data(data)
        print(f"Saved {len(data)} records for {ticker}")

        return True

    last_date = row[0]
    last_valid_date = row[1]
    if last_valid_date == datetime.today().strftime('%Y-%m-%d'):
        print(f"Stock data for {ticker} already up to date")
        return True

    # Yahoo may correct a previously downloaded session later. Revisit recent rows
    # instead of starting after the newest row, which would preserve bad candles.
    refresh_start = (
        datetime.strptime(last_date, "%Y-%m-%d") - timedelta(days=REFRESH_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")
    print(f"Updating {ticker} from {refresh_start}")
    data = fetch_stock_data(ticker=ticker,interval=interval, start_date=refresh_start)

    if data.empty:
        print(f"No new data for {ticker}")
        return last_valid_date is not None

    save_stock_data(data)
    print(f"Saved {len(data)} records for {ticker}")
    return True


if __name__ == "__main__":
    ensure_stock_data("AAPL")
