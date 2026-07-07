import sqlite3
from data_fetcher import *
import pandas as pd

DB_NAME = "stocks.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    return conn

def create_table():
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
    volume INTEGER NOT NULL,
    dividends REAL default 0,
    stock_splits REAL default 0,
        
    PRIMARY KEY (ticker, interval, date))
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
    cur.executemany("""INSERT OR REPLACE INTO stock_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows)

    conn.commit()
    cur.close()

def get_stock_data(ticker, interval="1d", years=5):
    conn = get_connection()
    ticker = ticker.upper()

    if years is None:
        query = "SELECT * FROM stock_prices WHERE ticker = ? AND interval = ? ORDER BY date ASC"
    else:
        query = "SELECT * FROM stock_prices WHERE ticker = ? AND interval = ? AND date >= date('now', ?) ORDER BY date ASC"
    data = pd.read_sql_query(query, conn, params=(ticker, interval, f"-{years} years"))

    conn.close()
    return data

if __name__ == "__main__":
    create_table()
    print("Table stock_prices created")
    data = fetch_stock_data("AAPL")
    save_stock_data(data)

