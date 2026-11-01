import pandas as pd
from services.ml.finbert_service import MODEL_NAME, MODEL_REVISION
from services.ml.prediction_time_service import (DEFAULT_MARKET_CALENDAR,NEWS_MINUTES_BEFORE_OPEN,build_market_schedule,build_preopen_schedule)
from services.ml.feature_schema import NEWS_BASE_FEATURE_COLUMNS, NEWS_FEATURE_COLUMNS

NEWS_HISTORY_SESSIONS = 5

NEWS_FEATURE_POLICY = {
    "source": "alpaca",
    "model_name": MODEL_NAME,
    "model_revision": MODEL_REVISION,
    "calendar_name": DEFAULT_MARKET_CALENDAR,
    "minutes_before_open": NEWS_MINUTES_BEFORE_OPEN,
    "window": "previous_close_to_preopen",
    "relevance_filter": "single_ticker",
    "sentiment_score": "positive_mean_minus_negative_mean",
    "sentiment_5d": "news_count_weighted_mean_over_5_sessions",
    "sentiment_change": "current_minus_previous_when_both_have_news",
    "history_sessions": NEWS_HISTORY_SESSIONS,
    "version": 2
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
        return pd.DataFrame(columns=["session_date"] + NEWS_BASE_FEATURE_COLUMNS)

    unique_news = (assigned_news.sort_values("available_at").drop_duplicates(["session_date", "article_id"], keep="last"))

    features = unique_news.groupby("session_date").agg(
        news_count=("article_id", "size"),
        news_positive_mean=("positive_probability", "mean"),
        news_negative_mean=("negative_probability", "mean"),
        news_neutral_mean=("neutral_probability", "mean")).reset_index()

    return features


def assign_news_to_preopen_sessions(news_df, sessions):
    if news_df.empty:
        return news_df.copy()
    assigned = pd.merge_asof(
        news_df.sort_values("available_at"),
        sessions[["session_date", "previous_close", "decision_at"]].sort_values("decision_at"),
        left_on="available_at",
        right_on="decision_at",
        direction="forward")

    return assigned.loc[assigned["available_at"].gt(assigned["previous_close"]) & assigned["available_at"].le(assigned["decision_at"])].copy()

def get_news_history_start(start_date,end_date,minutes_before_open=NEWS_MINUTES_BEFORE_OPEN):
    range_start = pd.to_datetime(start_date, utc=True)
    range_end = pd.to_datetime(end_date, utc=True)

    if pd.isna(range_start) or pd.isna(range_end) or range_start >= range_end:
        raise ValueError("Invalid news range")

    schedule = build_market_schedule((range_start - pd.Timedelta(days=45)).date().isoformat(),(range_start + pd.Timedelta(days=14)).date().isoformat())
    sessions = build_preopen_schedule(schedule, minutes_before_open)
    matches = sessions.index[
        sessions["previous_close"].ge(range_start) & sessions["decision_at"].lt(range_end)]

    if len(matches) == 0 or matches[0] < NEWS_HISTORY_SESSIONS - 1:
        raise ValueError("Not enough previous news sessions")

    return sessions.iloc[matches[0] - (NEWS_HISTORY_SESSIONS - 1)]["previous_close"].isoformat()


def build_preopen_news_features(news_df,schedule,start_date,end_date,minutes_before_open=NEWS_FEATURE_POLICY["minutes_before_open"], history_start=None):
    range_start = pd.to_datetime(start_date, utc=True)
    range_end = pd.to_datetime(end_date, utc=True)

    load_start = (range_start if history_start is None else pd.to_datetime(history_start, utc=True))

    if pd.isna(load_start) or load_start > range_start:
        raise ValueError("Invalid news history start")

    if range_start >= range_end:
        raise ValueError("Start date must be before end date")

    sessions = build_preopen_schedule(schedule, minutes_before_open)

    sessions = sessions.loc[(sessions["previous_close"] >= load_start) & (sessions["decision_at"] < range_end)].copy()

    if sessions.empty:
        raise ValueError("No fully covered news windows")

    news = news_df.copy()
    if not news.empty:
        news = news.loc[(news["available_at"] >= load_start) & (news["available_at"] < range_end)]

    assigned = assign_news_to_preopen_sessions(news, sessions)
    aggregated = aggregate_news_by_session(assigned)

    result = sessions[["session_date", "feature_date", "execution_date", "decision_at", "previous_close"]].merge(aggregated,on="session_date",how="left",validate="one_to_one")

    result = result.rename(columns={"feature_date": "date"})

    result[NEWS_BASE_FEATURE_COLUMNS] = (result[NEWS_BASE_FEATURE_COLUMNS].astype(float).fillna(0))
    result["news_count"] = result["news_count"].astype(int)

    count = result["news_count"]
    score = result["news_positive_mean"] - result["news_negative_mean"]
    result["news_count_5d"] = result["news_count"].rolling(5).sum()
    result["news_active_sessions_5d"] = ((result["news_count"] > 0).rolling(5).sum())


    result["news_sentiment_5d"] = ((score * result["news_count"]).rolling(5).sum() / result["news_count_5d"])
    result.loc[result["news_count_5d"] == 0, "news_sentiment_5d"] = 0

    result["news_sentiment_change"] = score - score.shift(1)
    result.loc[(result["news_count"] == 0) | (result["news_count"].shift(1) == 0), "news_sentiment_change"] = 0
    result = result.loc[result["previous_close"] >= range_start]

    return result[["date", "execution_date", "decision_at"] + NEWS_FEATURE_COLUMNS].sort_values("date").reset_index(drop=True)
