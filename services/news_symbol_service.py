import pandas as pd


NEWS_SYMBOL_POLICY = {
    "aliases": {"META": ["META", "FB"], "BRK-B": ["BRK-B", "BRK"]},
    "meta_symbol_start": "2022-06-09T00:00:00Z",
    "article_date": "created_at",
    "missing_meta_date": "reject"
}


def normalize_news_ticker(ticker):
    ticker = ticker.strip().upper()
    for canonical, aliases in NEWS_SYMBOL_POLICY["aliases"].items():
        if ticker in aliases:
            return canonical
    return ticker


def get_news_symbols(ticker, as_of=None):
    ticker = normalize_news_ticker(ticker)
    if ticker == "META" and as_of is not None:
        timestamp = pd.to_datetime(as_of, utc=True)
        if pd.isna(timestamp):
            raise ValueError("News symbol date cannot be missing")
        if timestamp < pd.Timestamp(NEWS_SYMBOL_POLICY["meta_symbol_start"]):
            return ["FB"]
    return list(NEWS_SYMBOL_POLICY["aliases"].get(ticker, [ticker]))
