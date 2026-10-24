import os

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

from services.ml.finbert_service import FinbertAnalyzer, MODEL_NAME, MODEL_REVISION
from services.db import (
    create_tables,
    get_news_imports,
    get_news_without_sentiment,
    save_news,
    save_news_import,
    save_sentiment,
)


class NewsPreparationError(RuntimeError):
    pass


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

def fetch_news(ticker, start_date, end_date, page_token=None):
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise NewsPreparationError("Missing Alpaca API credentials")
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

    try:
        response = news_session.get(
            NEWS_URL,
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        raise NewsPreparationError(
            f"{ticker}: Cannot download news from Alpaca."
        ) from error

    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("news"), list)
    ):
        raise NewsPreparationError(
            f"{ticker}: Alpaca returned invalid format of news."
        )

    return payload

def fetch_all_news(ticker, start_date, end_date):
    articles = []
    page_token = None
    seen_tokens = set()
    while True:
        page = fetch_news(ticker, start_date, end_date, page_token=page_token)
        articles.extend(page["news"])
        page_token = page.get("next_page_token")
        if page_token is None or page_token == "":
            break
        if not isinstance(page_token, str) or page_token in seen_tokens:
            raise NewsPreparationError(
                f"{ticker}: Alpaca returned invalid news pagination."
            )
        seen_tokens.add(page_token)

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


def find_missing_news_ranges(imports, start_date, end_date):
    range_start = pd.to_datetime(start_date, utc=True)
    range_end = pd.to_datetime(end_date, utc=True)

    if pd.isna(range_start) or pd.isna(range_end):
        raise ValueError("Range dates cannot be missing")

    if range_start >= range_end:
        raise ValueError("Start date must be before end date")

    imported_ranges = []

    for row in imports:
        import_start = pd.to_datetime(row["start_at"], utc=True)
        import_end = pd.to_datetime(row["end_at"], utc=True)
        completed_at = pd.to_datetime(row["completed_at"], utc=True)

        if import_end > completed_at:
            continue

        import_start = max(import_start, range_start)
        import_end = min(import_end, range_end)

        if import_start < import_end:
            imported_ranges.append((import_start, import_end))

    missing_ranges = []
    covered_until = range_start

    for import_start, import_end in sorted(imported_ranges):
        if import_start > covered_until:
            missing_ranges.append({
                "start_date": covered_until.isoformat(),
                "end_date": import_start.isoformat()})

        covered_until = max(covered_until, import_end)

        if covered_until >= range_end:
            break

    if covered_until < range_end:
        missing_ranges.append({
            "start_date": covered_until.isoformat(),
            "end_date": range_end.isoformat(),
        })

    return missing_ranges


def import_news_range(
    ticker, start_date, end_date, *, as_of=None, refresh=False,
    progress_callback=None,
):
    ticker = ticker.upper()


    range_start = pd.to_datetime(start_date, utc=True)
    range_end = pd.to_datetime(end_date, utc=True)

    now = pd.Timestamp.now(tz="UTC")
    import_until = (
        now if as_of is None
        else pd.to_datetime(as_of, utc=True))

    if (
        pd.isna(range_start)
        or pd.isna(range_end)
        or pd.isna(import_until)):
        raise ValueError("Import dates cannot be missing")

    if range_start >= range_end:
        raise ValueError("Start date must be before end date")

    if import_until > now:
        raise ValueError("as_of cannot be in the future")

    range_end = min(range_end, import_until)

    if range_start >= range_end:
        raise ValueError("No import range before as_of")

    start_date = range_start.isoformat()
    end_date = range_end.isoformat()

    if refresh:
        ranges_to_import = [{
            "start_date": start_date,
            "end_date": end_date}]
    else:
        imports = get_news_imports(ticker, start_date, end_date)
        ranges_to_import = find_missing_news_ranges(imports, start_date, end_date)

    if not ranges_to_import:
        if progress_callback:
            progress_callback(f"{ticker}: Zakres wiadomości jest już w bazie.")
        return {
            "status": "skipped",
            "fetched_count": None,
            "inserted_count": 0,
            "start_date": start_date,
            "end_date": end_date
        }

    fetched_count = 0
    inserted_count = 0

    for missing_range in ranges_to_import:
        periods = build_month_blocks(missing_range["start_date"],missing_range["end_date"])

        for period in periods:
            period_start = period["start_date"]
            period_end = period["end_date"]

            if progress_callback:
                progress_callback(
                    f"{ticker}: Pobieranie wiadomości {period_start} — {period_end}."
                )

            articles = fetch_all_news(
                ticker, period_start, period_end)

            inserted_count += save_news(articles)
            save_news_import(
                ticker,
                period_start,
                period_end,
                len(articles))

            fetched_count += len(articles)
            if progress_callback:
                progress_callback(
                    f"{ticker}: Zakończono blok; pobrano wiadomości: {len(articles)}."
                )

    return {
        "status": "completed",
        "fetched_count": fetched_count,
        "inserted_count": inserted_count,
        "start_date": start_date,
        "end_date": end_date
    }

def build_month_blocks(start_date, end_date):
    start = pd.to_datetime(start_date, utc=True)
    end = pd.to_datetime(end_date, utc=True)

    if pd.isna(start) or pd.isna(end):
        raise ValueError("Range dates cannot be missing")

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

def score_pending_news(
    ticker, analyzer, *, start_date=None, end_date=None,
    progress_callback=None,
):
    ticker = ticker.upper()
    pending = get_news_without_sentiment(ticker, MODEL_NAME, MODEL_REVISION, start_date=start_date, end_date=end_date)
    results = []
    inserted_count = 0
    for article in pending:
        if classify_news_relevance(article, ticker)["use_for_model"]:
            try:
                sentiment = analyzer.analyze(article["headline"])
            except (OSError, RuntimeError) as error:
                raise NewsPreparationError(
                    f"{ticker}: Cannot calculate FinBERT score for {article['article_id']}."
                ) from error
            results.append({
                "article_id": article["article_id"],
                "updated_at": article["updated_at"],
                "sentiment": sentiment,
            })

            if len(results) >= 100:
                inserted_count += save_sentiment(results, MODEL_NAME, MODEL_REVISION)
                results.clear()
                if progress_callback:
                    progress_callback(
                        f"{ticker}: Zapisano ocen FinBERT: {inserted_count}."
                    )

    inserted_count += save_sentiment(results, MODEL_NAME, MODEL_REVISION)
    return inserted_count

def ensure_news_data(ticker, start_date, end_date, *,analyzer=None,refresh=False,as_of=None,progress_callback=None):
    ticker = ticker.upper()

    if progress_callback:
        progress_callback(f"{ticker}: Przygotowywanie wiadomości.")

    import_result = import_news_range(
        ticker,
        start_date,
        end_date,
        as_of=as_of,
        refresh=refresh,
        progress_callback=progress_callback,
    )

    if analyzer is None:
        analyzer = FinbertAnalyzer()

    if progress_callback:
        progress_callback(f"{ticker}: Sprawdzanie brakujących ocen FinBERT.")

    sentiments_inserted = score_pending_news(
        ticker,
        analyzer,
        start_date=import_result["start_date"],
        end_date=import_result["end_date"],
        progress_callback=progress_callback,
    )

    if progress_callback:
        progress_callback(f"{ticker}: Zapisano ocen FinBERT: {sentiments_inserted}.")

    return {
        "ticker": ticker,
        "start_date": import_result["start_date"],
        "end_date": import_result["end_date"],
        "import": import_result,
        "sentiments_inserted": sentiments_inserted
    }
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

    analyzer = FinbertAnalyzer()
    import_as_of = pd.Timestamp.now(tz="UTC")

    for ticker in tickers:
        result = ensure_news_data(
            ticker,
            periods[0]["start_date"],
            periods[-1]["end_date"],
            analyzer=analyzer,
            as_of=import_as_of,
            progress_callback=print,
        )
        print(ticker, result)


