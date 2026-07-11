import pandas as pd
from dotenv import load_dotenv
import os
import psycopg

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")

    conn = psycopg.connect(DATABASE_URL)
    return conn

def create_tables():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS stock_prices (
    ticker TEXT NOT NULL,
    interval TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume BIGINT NOT NULL,
    dividends REAL default 0,
    stock_splits REAL default 0,
        
    PRIMARY KEY (ticker, interval, date))
    ''')

    cur.execute('''CREATE TABLE IF NOT EXISTS companies (
    ticker TEXT PRIMARY KEY,
    name TEXT NOT NULL)
    ''')

    conn.commit()
    conn.close()

def save_stock_data(data):
    conn = get_connection()
    cur = conn.cursor()

    rows = data[[
        "ticker",
        "interval",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dividends",
        "stock_splits"
    ]].values.tolist()
    cur.executemany("""INSERT INTO stock_prices (ticker, interval, date, open, high, low, close, volume, dividends, stock_splits)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, interval, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            dividends = EXCLUDED.dividends,
            stock_splits = EXCLUDED.stock_splits
    """, rows)

    conn.commit()
    conn.close()

def get_stock_data(ticker, interval="1d", years=5):
    conn = get_connection()
    ticker = ticker.upper()

    if years is None:
        query = "SELECT * FROM stock_prices WHERE ticker = %s AND interval = %s ORDER BY date ASC"
        data = pd.read_sql_query(query, conn, params=(ticker, interval))

    else:
        query = """SELECT * FROM stock_prices WHERE ticker = %s AND interval = %s
              AND date >= to_char(CURRENT_DATE - (%s * INTERVAL '1 year'), 'YYYY-MM-DD') ORDER BY date ASC"""
        data = pd.read_sql_query(query, conn, params=(ticker, interval, years))

    conn.close()
    return data

def get_recent_stock_prices(ticker, interval="1d", limit=7):
    conn = get_connection()
    ticker = ticker.upper()

    query = """
        SELECT date, close
        FROM stock_prices
        WHERE ticker = %s AND interval = %s
        ORDER BY date DESC
        LIMIT %s
    """

    data = pd.read_sql_query(query, conn, params=(ticker, interval, limit))
    conn.close()

    if data.empty:
        return data

    return data.sort_values("date").reset_index(drop=True)

def get_company_name_from_db(ticker):
    conn = get_connection()
    cur = conn.cursor()

    ticker = ticker.upper()
    cur.execute("SELECT name from companies WHERE ticker = %s", (ticker,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    return row[0]

def save_company_data(ticker, name):
    conn = get_connection()
    cur = conn.cursor()
    ticker = ticker.upper()

    cur.execute("INSERT INTO companies VALUES (%s, %s)", (ticker, name))

    conn.commit()
    conn.close()


def get_latest_stock_summary(ticker, interval="1d"):
    conn = get_connection()

    ticker = ticker.upper()

    query = """
        SELECT date, close, volume
        FROM stock_prices
        WHERE ticker = %s AND interval = %s
        ORDER BY date DESC
        LIMIT 2
    """

    data = pd.read_sql_query(query, conn, params=(ticker, interval))
    conn.close()
    if data.empty:
        return None

    latest = data.iloc[0]
    last_price = float(latest["close"])
    volume = int(latest["volume"])
    last_date = latest["date"]

    previous_price = None
    price_change = None
    price_change_percent = None
    trend = "-"

    if len(data) > 1:
        previous = data.iloc[1]
        previous_price = float(previous["close"])

        price_change = last_price - previous_price

        if previous_price != 0:
            price_change_percent = (price_change / previous_price) * 100

    return {
        "last_price": last_price,
        "volume": volume,
        "last_date": last_date,
        "previous_price": previous_price,
        "price_change": price_change,
        "price_change_percent": price_change_percent,
    }

if __name__ == "__main__":
    create_tables()
