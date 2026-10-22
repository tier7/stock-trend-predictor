import pandas as pd
from dotenv import load_dotenv
import os
import psycopg
from services.stock_data_quality_service import VALID_OHLC_SQL, filter_valid_ohlc
from psycopg.rows import dict_row

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

    cur.execute('''CREATE TABLE IF NOT EXISTS news_articles (
    article_id BIGINT NOT NULL,
    headline TEXT NOT NULL,
    symbols TEXT[],
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (article_id, updated_at)
    )
    ''')


    cur.execute('''CREATE TABLE IF NOT EXISTS news_sentiments (
    article_id BIGINT NOT NULL,
    article_updated_at TIMESTAMPTZ NOT NULL,
    model_name TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    positive_probability DOUBLE PRECISION NOT NULL,
    negative_probability DOUBLE PRECISION NOT NULL,
    neutral_probability DOUBLE PRECISION NOT NULL,
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (
        article_id, article_updated_at, model_name, model_revision
    ),

    FOREIGN KEY (article_id, article_updated_at)
        REFERENCES news_articles (article_id, updated_at)
    )
    ''')

    cur.execute("""CREATE TABLE IF NOT EXISTS news_imports (
        ticker TEXT NOT NULL,
        start_at TIMESTAMPTZ NOT NULL,
        end_at TIMESTAMPTZ NOT NULL,
        fetched_count INTEGER NOT NULL CHECK (fetched_count >= 0),
        completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

        PRIMARY KEY (ticker, start_at, end_at),
        CHECK (start_at < end_at)
        )
    """)

    conn.commit()
    conn.close()

def save_stock_data(data):
    data = filter_valid_ohlc(data)
    if data.empty:
        return
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
    cur = conn.cursor()
    ticker = ticker.upper()

    if years is None:
        query = "SELECT * FROM stock_prices WHERE ticker = %s AND interval = %s ORDER BY date ASC"
        cur.execute(query, (ticker, interval))

    else:
        query = """SELECT * FROM stock_prices WHERE ticker = %s AND interval = %s
              AND date >= to_char(CURRENT_DATE - (%s * INTERVAL '1 year'), 'YYYY-MM-DD') ORDER BY date ASC"""
        cur.execute(query, (ticker, interval, years))

    columns = [column.name for column in cur.description]
    data = pd.DataFrame(cur.fetchall(), columns=columns)

    conn.close()
    return data

def get_recent_stock_prices(ticker, interval="1d", limit=7):
    conn = get_connection()
    ticker = ticker.upper()

    query = f"""
        SELECT date, close
        FROM stock_prices
        WHERE ticker = %s AND interval = %s
          AND ({VALID_OHLC_SQL})
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

    query = f"""
        SELECT date, close, volume
        FROM stock_prices
        WHERE ticker = %s AND interval = %s
          AND ({VALID_OHLC_SQL})
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

def save_news(articles):
    if len(articles) <= 0:
        return 0
    conn = get_connection()
    cur = conn.cursor()

    rows = []
    for article in articles:
        row = (
            article["id"],
            article["headline"],
            article.get("symbols") or [],
            article["created_at"],
            article["updated_at"],
        )
        rows.append(row)

    cur.executemany("""INSERT INTO news_articles (article_id, headline, symbols, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (article_id, updated_at) DO NOTHING""", rows)
    inserted_count = cur.rowcount
    conn.commit()
    conn.close()

    return inserted_count

def get_news(ticker):
    conn = get_connection()
    cur = conn.cursor(row_factory=dict_row)

    cur.execute(f"""SELECT article_id, headline, symbols, created_at, updated_at
    FROM news_articles
    WHERE %s = ANY(symbols)
    ORDER BY created_at, article_id, updated_at""", (ticker.upper(),))

    articles = cur.fetchall()

    conn.close()

    return articles


def save_sentiment(results, model_name, model_revision):
    if len(results) <= 0:
        return 0
    conn = get_connection()
    cur = conn.cursor()

    rows = []
    for result in results:
        row = (
            result["article_id"],
            result["updated_at"],
            model_name,
            model_revision,
            result["sentiment"]["positive"],
            result["sentiment"]["negative"],
            result["sentiment"]["neutral"],
        )
        rows.append(row)

    cur.executemany("""INSERT INTO news_sentiments (article_id, article_updated_at, model_name, model_revision, positive_probability, negative_probability, neutral_probability)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (article_id, article_updated_at, model_name, model_revision) DO NOTHING""", rows)
    inserted_count = cur.rowcount
    conn.commit()
    conn.close()

    return inserted_count

def get_news_without_sentiment(ticker, model_name, model_revision):
    conn = get_connection()
    cur = conn.cursor(row_factory=dict_row)
    cur.execute(f"""
    SELECT n.article_id, n.headline, n.symbols, n.created_at, n.updated_at
    FROM news_articles AS n
    WHERE %s = ANY(n.symbols)
    AND NOT EXISTS (
    SELECT 1
    FROM news_sentiments AS s
    WHERE s.article_id = n.article_id
    AND s.article_updated_at = n.updated_at
    AND s.model_name = %s
    AND s.model_revision = %s
    )
    ORDER BY n.created_at, n.article_id, n.updated_at  
    """, (ticker.upper(), model_name, model_revision))

    articles = cur.fetchall()

    conn.close()

    return articles

def get_news_with_sentiment(ticker, model_name, model_revision):
    conn = get_connection()
    cur = conn.cursor(row_factory=dict_row)
    cur.execute("""SELECT
    n.article_id, n.headline, n.symbols, n.created_at, n.updated_at,
    s.positive_probability, s.negative_probability, s.neutral_probability
    FROM news_articles AS n
    JOIN news_sentiments AS s
    ON s.article_id = n.article_id
    AND s.article_updated_at = n.updated_at
    WHERE %s = ANY(n.symbols)
    AND s.model_name = %s
    AND s.model_revision = %s
    ORDER BY n.created_at, n.article_id, n.updated_at""", (ticker.upper(), model_name, model_revision))

    articles = cur.fetchall()
    conn.close()
    return articles

def is_news_imported(ticker, start_date, end_date):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""SELECT 1 FROM news_imports WHERE ticker = %s AND start_at = %s AND end_at = %s""", (ticker.upper(),start_date, end_date))

    import_exists = cur.fetchone() is not None
    conn.close()

    return import_exists


def get_news_imports(ticker, start_date, end_date):
    """Read completed import blocks in one connection, without downloading news."""
    conn = get_connection()
    try:
        cur = conn.cursor(row_factory=dict_row)
        cur.execute("""
            SELECT start_at, end_at
            FROM news_imports
            WHERE ticker = %s AND start_at < %s AND end_at > %s
            ORDER BY start_at, end_at
        """, (ticker.upper(), end_date, start_date))
        return cur.fetchall()
    finally:
        conn.close()


def save_news_import(ticker, start_date, end_date, fetched_count):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""INSERT INTO news_imports (ticker, start_at, end_at, fetched_count)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (ticker, start_at, end_at) DO NOTHING""", (ticker.upper(), start_date, end_date, fetched_count)
    )

    inserted_count = cur.rowcount
    conn.commit()
    conn.close()
    return inserted_count

if __name__ == "__main__":
    create_tables()
