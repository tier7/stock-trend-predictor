import os

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from sympy.codegen.ast import Raise

from services.ml.finbert_service import load_finbert, analyze_sentiment, MODEL_NAME, MODEL_REVISION
from services.db import create_tables, save_news, get_news, save_sentiment, get_news_without_sentiment, get_news_with_sentiment, is_news_imported, save_news_import

load_dotenv()

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

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

    response = requests.get(NEWS_URL, headers=headers, params=params, timeout=30)
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


if __name__ == "__main__":
    '''create_tables()
    ticker = "AAPL"

    ranges = build_month_blocks(
        "2023-01-01T00:00:00Z",
        "2023-04-01T00:00:00Z",
    )

    for period in ranges:
        result = import_news_range(
            ticker,
            period["start_date"],
            period["end_date"],
        )
        print(ticker, period["start_date"][:7], result)
    ticker = "AAPL"
    scored_articles = get_news_with_sentiment(
        ticker, MODEL_NAME, MODEL_REVISION
    )

    print("Wiadomości z ocenami:", len(scored_articles))

    if scored_articles:
        print(scored_articles[0])
    '''
    ticker = "AAPL"
    articles = get_news_without_sentiment(ticker, MODEL_NAME, MODEL_REVISION)
    results = []
    accepted_articles = []

    for article in articles:
        relevance = classify_news_relevance(article, ticker)
        if relevance["use_for_model"]:
            accepted_articles.append(article)
    if not accepted_articles:
        print("No news to classify")
    else:
        tokenizer, model = load_finbert()
        for article in accepted_articles:
            sentiment = analyze_sentiment(article["headline"], tokenizer, model)
            results.append({
                "article_id": article["article_id"],
                "updated_at": article["updated_at"],
                "sentiment": sentiment,
            })
            print(article["headline"])
            print(sentiment)

    inserted = save_sentiment(results, MODEL_NAME, MODEL_REVISION)
    print("Zapisano ocen:", inserted)


