import os

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from sympy.codegen.ast import Raise


from services.ml.finbert_service import load_finbert, analyze_sentiment, MODEL_NAME, MODEL_REVISION
from services.db import create_tables, save_news, get_news, save_sentiment, get_news_without_sentiment, get_news_with_sentiment, is_news_imported, save_news_import

load_dotenv()

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
news_session = requests.Session()
news_session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )),
)

news_session = requests.Session()
news_session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )),
)

def fetch_news(ticker, start_date, end_date, page_token=None):
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError("Missing Alpaca API credentials")
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key
    }

    params = {
        "symbols": ticker,
        "start": start_date,
        "end": end_date,
        "limit": 50,
        "sort": "asc",
    }
    if page_token is not None:
        params["page_token"] = page_token

    response = news_session.get(NEWS_URL, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def fetch_all_news(ticker, start_date, end_date):
    articles = []
    page_token = None
    while True:
        page = fetch_news(ticker, start_date, end_date, page_token=page_token)
        articles.extend(page["news"])
        page_token = page.get("next_page_token")
        if not page_token:
            break

    return articles


def classify_news_relevance(article, ticker):
    headline = article.get("headline")
    tickers = set(article.get("symbols") or [])


    if not isinstance(headline, str) or not headline.strip():
        return {
            "use_for_model": False,
            "reason": "empty_headline"
        }
    elif not tickers:
        return {
            "use_for_model": False,
            "reason": "missing_symbols"
        }
    elif ticker not in tickers:
        return {
            "use_for_model": False,
            "reason": "ticker_not_mentioned"
        }
    elif ticker in tickers and len(tickers) == 1:
        return {
            "use_for_model": True,
            "reason": "single_ticker"
        }
    else:
        return {
            "use_for_model": False,
            "reason": "multiple_tickers"
        }

def import_news_range(ticker, start_date, end_date):
    ticker = ticker.upper()
    is_imported = is_news_imported(ticker, start_date, end_date)
    if is_imported:
        return {
            "status": "skipped",
            "fetched_count": None,
            "inserted_count": 0
        }
    articles = fetch_all_news(ticker, start_date, end_date)
    inserted_count = save_news(articles)
    save_news_import(ticker, start_date, end_date, len(articles))
    return {
        "status": "completed",
        "fetched_count": len(articles),
        "inserted_count": inserted_count
    }

def build_month_blocks(start_date, end_date):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    if start >= end:
        raise ValueError("Start date must be before end date")

    ranges = []
    current = start

    while current < end:
        next_month = (current+pd.offsets.MonthBegin(1)).normalize()
        period_end = min(next_month, end)
        ranges.append({
            "start_date": current.isoformat(),
            "end_date": period_end.isoformat(),
        })

        current = period_end
    return ranges

def score_pending_news(ticker, tokenizer, model):
    ticker = ticker.upper()
    pending = get_news_without_sentiment(ticker, MODEL_NAME, MODEL_REVISION)
    results = []
    inserted_count = 0
    for article in pending:
        if classify_news_relevance(article, ticker)["use_for_model"]:
            sentiment = analyze_sentiment(article["headline"], tokenizer, model)
            results.append({
                "article_id": article["article_id"],
                "updated_at": article["updated_at"],
                "sentiment": sentiment,
            })

            if len(results) >= 100:
                inserted_count += save_sentiment(results, MODEL_NAME, MODEL_REVISION)
                results.clear()
                print(ticker, "Zapisano ocen:", inserted_count, flush=True)

    inserted_count += save_sentiment(results, MODEL_NAME, MODEL_REVISION)
    return inserted_count
if __name__ == "__main__":
    create_tables()

    tickers = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
        "META", "TSLA", "AMD", "INTC", "ORCL",
        "JPM", "BAC", "GS", "V", "MA", "BRK-B",
        "JNJ", "PFE", "LLY", "UNH", "ABBV",
        "XOM", "CVX", "SHEL", "NEE",
        "KO", "PEP", "MCD", "SBUX", "WMT",
        "COST", "PG", "NKE", "MC.PA",
        "DIS", "NFLX", "CMCSA", "VZ", "T", "TMUS",
        "BA", "AIR.PA", "LMT", "CAT", "DE",
        "UNP", "FDX", "UPS", "O", "AMT"
    ]
    periods = build_month_blocks(
        "2021-01-01T00:00:00Z",
        "2025-01-01T00:00:00Z",
    )

    tokenizer, model = load_finbert()

    for ticker in tickers:
        for period in periods:
            news = import_news_range(ticker, period['start_date'], period['end_date'])
            print(ticker, period['start_date'], period['end_date'], news)
        inserted_count = score_pending_news(ticker, tokenizer, model)
        print(ticker, "Zapisano ocen:", inserted_count)


