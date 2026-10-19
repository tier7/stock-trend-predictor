import os
import requests
from dotenv import load_dotenv
from services.ml.finbert_service import load_finbert, analyze_sentiment, MODEL_NAME, MODEL_REVISION
from services.db import create_tables, save_news, get_news, save_sentiment, get_news_without_sentiment, get_news_with_sentiment

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


if __name__ == "__main__":
    ticker = "AAPL"
    scored_articles = get_news_with_sentiment(
        ticker, MODEL_NAME, MODEL_REVISION
    )

    print("Wiadomości z ocenami:", len(scored_articles))

    if scored_articles:
        print(scored_articles[0])
    '''ticker = "AAPL"
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
        for article in accepted_articles:
            tokenizer, model = load_finbert()
            sentiment = analyze_sentiment(article["headline"], tokenizer, model)
            results.append({
                "article_id": article["article_id"],
                "updated_at": article["updated_at"],
                "sentiment": sentiment,
            })
            print(article["headline"])
            print(sentiment)

    inserted = save_sentiment(results, MODEL_NAME, MODEL_REVISION)
    print("Zapisano ocen:", inserted)'''


