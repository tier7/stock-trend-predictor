import sqlite3
import services.data_fetcher
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_NAME = BASE_DIR / "stocks.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
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
    volume INTEGER NOT NULL,
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
    cur.executemany("""INSERT OR REPLACE INTO stock_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows)

    conn.commit()
    conn.close()

def get_stock_data(ticker, interval="1d", years=5):
    conn = get_connection()
    ticker = ticker.upper()

    if years is None:
        query = "SELECT * FROM stock_prices WHERE ticker = ? AND interval = ? ORDER BY date ASC"
        data = pd.read_sql_query(query, conn, params=(ticker, interval))

    else:
        query = "SELECT * FROM stock_prices WHERE ticker = ? AND interval = ? AND date >= date('now', ?) ORDER BY date ASC"
        data = pd.read_sql_query(query, conn, params=(ticker, interval, f"-{years} years"))

    conn.close()
    return data

def get_company_name_from_db(ticker):
    conn = get_connection()
    cur = conn.cursor()

    ticker = ticker.upper()
    cur.execute("SELECT name from companies WHERE ticker = ?", (ticker,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    return row[0]

def save_company_data(ticker, name):
    conn = get_connection()
    cur = conn.cursor()
    ticker = ticker.upper()

    cur.execute("INSERT INTO companies VALUES (?, ?)", (ticker, name))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_tables()