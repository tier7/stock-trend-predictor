import pandas as pd
import pandas_market_calendars as mcal
from services.db import get_news_with_sentiment, get_stock_data
from services.ml.finbert_service import MODEL_NAME, MODEL_REVISION
from services.db import is_news_imported, get_news_without_sentiment
from services.news_fetcher import build_month_blocks, classify_news_relevance

NEWS_FEATURE_COLUMNS = [
    "news_count",
    "news_positive_mean",
    "news_negative_mean",
    "news_neutral_mean",
]

def prepare_news_dataframe(articles):
    data = pd.DataFrame(articles)
    if data.empty:
        return data

    data["created_at"] = pd.to_datetime(data["created_at"], utc=True)
    data["updated_at"] = pd.to_datetime(data["updated_at"], utc=True)
    data["available_at"] = data[["created_at", "updated_at"]].max(axis=1)

    return data.sort_values("available_at")

def build_market_schedule(start_date, end_date, calendar_name="NYSE"):
    calendar = mcal.get_calendar(calendar_name)

    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    return schedule

def assign_news_to_sessions(news_df, schedule):
    if news_df.empty:
        return news_df.copy()

    sessions = schedule.reset_index(names="session_date")

    assigned = pd.merge_asof(
        news_df.sort_values("available_at"),
        sessions.sort_values("market_close"),
        left_on="available_at",
        right_on="market_close",
        direction="forward"
    )

    if assigned["session_date"].isna().any():
        raise ValueError("Calendar ends too early")

    return assigned


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

def merge_news_features(stock_df, news_features):
    prices = stock_df.copy()
    features = news_features.copy()

    prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")
    features["date"] = pd.to_datetime(features["session_date"]).dt.strftime("%Y-%m-%d")

    merged = prices.merge(
        features[["date"] + NEWS_FEATURE_COLUMNS],
        on="date",
        how="left",
        validate="one_to_one",
    )

    merged[NEWS_FEATURE_COLUMNS] = merged[NEWS_FEATURE_COLUMNS].fillna(0)
    merged["news_count"] = merged["news_count"].astype(int)

    return merged
if __name__ == "__main__":
    ticker = "AAPL"
    range_start = pd.Timestamp("2023-01-01", tz="UTC")
    range_end = pd.Timestamp("2023-04-01", tz="UTC")
    for period in build_month_blocks("2023-01-01T00:00:00Z","2023-04-01T00:00:00Z"):
        if not is_news_imported(ticker, period["start_date"], period["end_date"]):
            raise ValueError(f"No import: {period['start_date']}")

    pending = get_news_without_sentiment(ticker, MODEL_NAME, MODEL_REVISION)

    for article in pending:
        available_at = pd.to_datetime(max(article["created_at"], article["updated_at"]),utc=True,)

        if not (range_start <= available_at < range_end):
            continue

        relevance = classify_news_relevance(article, ticker)
        if relevance["use_for_model"]:
            raise ValueError("Firstly calculate relevance of news")

    articles = get_news_with_sentiment(ticker, MODEL_NAME, MODEL_REVISION)
    news_df = prepare_news_dataframe(articles)
    if not news_df.empty:
        news_df = news_df.loc[(news_df["available_at"] >= range_start) & (news_df["available_at"] < range_end)].copy()


    schedule = build_market_schedule("2022-12-23", "2023-04-07")
    assigned = assign_news_to_sessions(news_df, schedule)
    news_features = aggregate_news_by_session(assigned)

    prices = get_stock_data(ticker, years=None)
    dates = pd.to_datetime(prices["date"])
    prices = prices.loc[
        dates.between("2023-01-04", "2023-03-31")
    ].copy()

    combined = merge_news_features(prices, news_features)

    print("Sesje:", len(combined))
    print("Sesje z newsami:", (combined["news_count"] > 0).sum())
    print("Sesje bez zaakceptowanych newsów:", (combined["news_count"] == 0).sum())
    print("Artykuły użyte w sesjach:", combined["news_count"].sum())

    print(combined[["date", "close"] + NEWS_FEATURE_COLUMNS].to_string(index=False))

    test_news = pd.DataFrame({
        "available_at": pd.to_datetime([
            "2023-01-06T20:59:00Z",  # piątek przed zamknięciem
            "2023-01-06T21:01:00Z",  # piątek po zamknięciu
            "2023-01-08T12:00:00Z",  # niedziela
        ], utc=True),
    })

    test_assigned = assign_news_to_sessions(test_news, schedule)

    print(test_assigned[
              ["available_at", "session_date", "market_close"]
          ].to_string(index=False))