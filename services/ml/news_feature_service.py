import pandas as pd
from services.db import (get_news_imports,get_news_without_sentiment,get_news_with_sentiment,)
from services.ml.finbert_service import MODEL_NAME, MODEL_REVISION
from services.news_fetcher import classify_news_relevance, find_missing_news_ranges
from services.ml.feature_schema import NEWS_FEATURE_COLUMNS
from services.ml.prediction_time_service import (DEFAULT_MARKET_CALENDAR,NEWS_MINUTES_BEFORE_OPEN,build_market_schedule,build_preopen_schedule)
NEWS_FEATURE_POLICY = {
    "source": "alpaca",
    "model_name": MODEL_NAME,
    "model_revision": MODEL_REVISION,
    "calendar_name": DEFAULT_MARKET_CALENDAR,
    "minutes_before_open": NEWS_MINUTES_BEFORE_OPEN,
    "window": "previous_close_to_preopen",
    "relevance_filter": "single_ticker",
}

def normalize_training_news_range(news_start, news_end, training_cutoff):
    if news_start is None or news_end is None:
        raise ValueError("News range cannot be missing")

    range_start = pd.to_datetime(news_start, utc=True)
    range_end = pd.to_datetime(news_end, utc=True)
    cutoff = pd.to_datetime(training_cutoff, utc=True)


    if pd.isna(range_start) or pd.isna(range_end) or pd.isna(cutoff):
        raise ValueError("Range dates cannot be missing")

    if range_start >= range_end:
        raise ValueError("Start range cannot be higher than end range")

    now = pd.Timestamp.now(tz="UTC")
    range_end = min(range_end, cutoff, now)
    if range_start >= range_end:
        raise ValueError("Start range cannot be higher than end range")

    return range_start.isoformat(), range_end.isoformat()

def prepare_news_dataframe(articles):
    data = pd.DataFrame(articles)
    if data.empty:
        return data

    data["created_at"] = pd.to_datetime(data["created_at"], utc=True)
    data["updated_at"] = pd.to_datetime(data["updated_at"], utc=True)
    data["available_at"] = data[["created_at", "updated_at"]].max(axis=1)

    return data.sort_values("available_at")


def aggregate_news_by_session(assigned_news):
    if assigned_news.empty:
        return pd.DataFrame(columns=["session_date"] + NEWS_FEATURE_COLUMNS)

    unique_news = (assigned_news.sort_values("available_at").drop_duplicates(["session_date", "article_id"], keep="last"))

    features = unique_news.groupby("session_date").agg(
        news_count=("article_id", "size"),
        news_positive_mean=("positive_probability", "mean"),
        news_negative_mean=("negative_probability", "mean"),
        news_neutral_mean=("neutral_probability", "mean"),
    ).reset_index()

    return features


def load_scored_news_range(ticker, start_date, end_date):
    ticker = ticker.upper()
    range_start = pd.to_datetime(start_date, utc=True)
    range_end = pd.to_datetime(end_date, utc=True)

    if pd.isna(range_start) or pd.isna(range_end):
        raise ValueError("Range dates cannot be missing")

    if range_start >= range_end:
        raise ValueError("Start date must be before end date")

    imports = get_news_imports(ticker, range_start, range_end)

    missing_ranges = find_missing_news_ranges(imports, range_start, range_end)

    if missing_ranges:
        first_missing = missing_ranges[0]
        raise ValueError(
            f"{ticker}: Missing news coverage between "
            f"{first_missing['start_date']} and "
            f"{first_missing['end_date']}")


    pending = get_news_without_sentiment(ticker, MODEL_NAME, MODEL_REVISION, start_date=range_start, end_date=range_end)

    for article in pending:
        if classify_news_relevance(article, ticker)["use_for_model"]:
            raise ValueError(f"{ticker}: First calculate the missing FinBERT scores")

    articles = get_news_with_sentiment(ticker, MODEL_NAME, MODEL_REVISION)
    accepted_articles = []
    for article in articles:
        if classify_news_relevance(article, ticker)["use_for_model"]:
            accepted_articles.append(article)

    news_df = prepare_news_dataframe(accepted_articles)

    if not news_df.empty:
        news_df = news_df.loc[
            (news_df["available_at"] >= range_start)
            & (news_df["available_at"] < range_end)].copy()

    return news_df



def assign_news_to_preopen_sessions(news_df, sessions):
    if news_df.empty:
        return news_df.copy()
    assigned = pd.merge_asof(
        news_df.sort_values("available_at"),
        sessions[["session_date", "previous_close", "decision_at"]].sort_values("decision_at"),
        left_on="available_at",
        right_on="decision_at",
        direction="forward",
    )
    # A story after today's cutoff must not leak into tomorrow's overnight window.
    return assigned.loc[
        assigned["available_at"].gt(assigned["previous_close"])
        & assigned["available_at"].le(assigned["decision_at"])
    ].copy()

def build_preopen_news_features(news_df,schedule,start_date,end_date,minutes_before_open=NEWS_FEATURE_POLICY["minutes_before_open"]):
    range_start = pd.to_datetime(start_date, utc=True)
    range_end = pd.to_datetime(end_date, utc=True)

    if range_start >= range_end:
        raise ValueError("Start date must be before end date")

    sessions = build_preopen_schedule(schedule, minutes_before_open)

    sessions = sessions.loc[
        (sessions["previous_close"] >= range_start)
        & (sessions["decision_at"] < range_end)
    ].copy()

    if sessions.empty:
        raise ValueError("No fully covered news windows")

    news = news_df.copy()
    if not news.empty:
        news = news.loc[
            (news["available_at"] >= range_start)
            & (news["available_at"] < range_end)
        ]

    assigned = assign_news_to_preopen_sessions(news, sessions)
    aggregated = aggregate_news_by_session(assigned)

    result = sessions[
        ["session_date", "feature_date", "execution_date", "decision_at"]
    ].merge(
        aggregated,
        on="session_date",
        how="left",
        validate="one_to_one",
    )

    result = result.rename(columns={"feature_date": "date"})

    result[NEWS_FEATURE_COLUMNS] = (
        result[NEWS_FEATURE_COLUMNS].astype(float).fillna(0)
    )
    result["news_count"] = result["news_count"].astype(int)

    return result[
        ["date", "execution_date", "decision_at"] + NEWS_FEATURE_COLUMNS
    ].sort_values("date").reset_index(drop=True)



def attach_preopen_news_features(data,ticker,start_date,end_date,minutes_before_open=NEWS_FEATURE_POLICY["minutes_before_open"]):
    if data.empty:
        raise ValueError("Empty stock data")

    range_start = pd.to_datetime(start_date, utc=True)
    range_end = pd.to_datetime(end_date, utc=True)

    news_df = load_scored_news_range(
        ticker,
        range_start.isoformat(),
        range_end.isoformat(),
    )

    calendar_start = (
        range_start - pd.DateOffset(days=14)
    ).date().isoformat()
    calendar_end = (
        range_end + pd.DateOffset(days=14)
    ).date().isoformat()

    schedule = build_market_schedule(calendar_start, calendar_end)

    news_features = build_preopen_news_features(
        news_df,
        schedule,
        range_start,
        range_end,
        minutes_before_open=minutes_before_open,
    )

    prices = data.copy()
    prices["date"] = pd.to_datetime(
        prices["date"]
    ).dt.strftime("%Y-%m-%d")

    return prices.merge(
        news_features,
        on="date",
        how="left",
        validate="one_to_one",
    )


