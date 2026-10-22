import pandas as pd
import pandas_market_calendars as mcal
from services.db import get_news_with_sentiment, get_stock_data
from services.ml.finbert_service import MODEL_NAME, MODEL_REVISION
from services.db import get_news_imports, get_news_without_sentiment
from services.news_fetcher import build_month_blocks, classify_news_relevance
from services.ml.dataset_service import FEATURE_COLUMNS, prepare_dataset

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

    merged[NEWS_FEATURE_COLUMNS] = merged[NEWS_FEATURE_COLUMNS].astype(float).fillna(0)
    merged["news_count"] = merged["news_count"].astype(int)

    return merged


def load_scored_news_range(ticker, start_date, end_date):
    ticker = ticker.upper()
    range_start = pd.to_datetime(start_date, utc=True)
    range_end = pd.to_datetime(end_date, utc=True)

    if range_start >= range_end:
        raise ValueError("Start date must be before end date")

    imports = get_news_imports(ticker, range_start, range_end)
    imported_ranges = {
        (pd.to_datetime(row["start_at"], utc=True), pd.to_datetime(row["end_at"], utc=True))
        for row in imports
    }
    periods = build_month_blocks(range_start.isoformat(), range_end.isoformat())
    for period in periods:
        bounds = (
            pd.to_datetime(period["start_date"], utc=True),
            pd.to_datetime(period["end_date"], utc=True),
        )
        if bounds not in imported_ranges:
            raise ValueError(f"{ticker}: No import: {period['start_date']}")

    pending = get_news_without_sentiment(ticker, MODEL_NAME, MODEL_REVISION)

    for article in pending:
        available_at = pd.to_datetime(max(article["created_at"], article["updated_at"]),utc=True,)

        if not (range_start <= available_at < range_end):
            continue

        relevance = classify_news_relevance(article, ticker)
        if relevance["use_for_model"]:
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


def prepare_dataset_with_news(ticker, start_date, end_date):
    ticker = ticker.upper()
    range_start = pd.to_datetime(start_date, utc=True)
    range_end = pd.to_datetime(end_date, utc=True)
    news_df = load_scored_news_range(ticker, start_date, end_date)

    calendar_start = (range_start - pd.DateOffset(days=14)).date().isoformat()
    calendar_end = (range_end + pd.DateOffset(days=14)).date().isoformat()

    schedule = build_market_schedule(calendar_start, calendar_end)
    schedule["previous_close"] = schedule["market_close"].shift(1)

    complete_sessions = schedule.loc[
        (schedule["previous_close"] >= range_start)
        & (schedule["market_close"] < range_end)].copy()

    if complete_sessions.empty:
        raise ValueError("No fully covered market sessions")

    assigned = assign_news_to_sessions(news_df, schedule)
    news_features = aggregate_news_by_session(assigned)

    prices = get_stock_data(ticker, years=None)

    if prices.empty:
        raise ValueError(f"No stock data for {ticker}")

    prepared = prepare_dataset(prices)

    complete_dates = complete_sessions.index.strftime("%Y-%m-%d")
    price_dates = pd.to_datetime(prepared["date"]).dt.strftime("%Y-%m-%d")

    prepared = prepared.loc[
        price_dates.isin(complete_dates)
    ].copy()

    if prepared.empty:
        raise ValueError(f"No prepared stock data for {ticker} in this range")

    combined = merge_news_features(prepared, news_features)
    combined["ticker"] = ticker

    return combined


def build_preopen_schedule(schedule, minutes_before_open=5, horizon=1):
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Horizon must be a positive integer")

    if not isinstance(minutes_before_open, int) or minutes_before_open <= 0:
        raise ValueError("minutes_before_open must be a positive integer")
    sessions = schedule.sort_index().copy()
    sessions.index.name = "session_date"
    dates = pd.Series(sessions.index.strftime("%Y-%m-%d"), index=sessions.index)
    sessions["execution_date"] = dates
    sessions["feature_date"] = dates.shift(1)
    sessions["previous_close"] = sessions["market_close"].shift(1)
    sessions["close_window_start"] = sessions["market_close"].shift(2)
    sessions["decision_at"] = sessions["market_open"].map(
        lambda opening: opening - pd.DateOffset(minutes=minutes_before_open)
    )
    sessions["target_end_at"] = sessions["market_open"].shift(-horizon)
    sessions["expected_target_end_date"] = dates.shift(-horizon)
    if (sessions["decision_at"] <= sessions["previous_close"]).any():
        raise ValueError("Decision must fall after the previous market close")
    return sessions.reset_index()


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


def build_news_timing_dataset(prepared_prices, news_df, schedule, start_date, end_date,minutes_before_open=5, horizon=1):

    range_start = pd.to_datetime(start_date, utc=True)
    range_end = pd.to_datetime(end_date, utc=True)
    if range_start >= range_end:
        raise ValueError("Start date must be before end date")
    sessions = build_preopen_schedule(schedule, minutes_before_open, horizon=horizon)
    complete = sessions.loc[
        sessions["close_window_start"].ge(range_start)
        & sessions["decision_at"].ge(range_start)
        & sessions["decision_at"].lt(range_end)
        & sessions["target_end_at"].lt(range_end)
    ].copy()
    if complete.empty:
        raise ValueError("No fully covered paired market sessions")

    news = news_df.copy()
    if not news.empty:
        news = news.loc[news["available_at"].ge(range_start) & news["available_at"].lt(range_end)]
    close_assigned = assign_news_to_sessions(news, schedule)
    preopen_assigned = assign_news_to_preopen_sessions(news, sessions)

    prices = prepared_prices.copy()
    for column in ["date", "next_date", "target_end_date"]:
        prices[column] = pd.to_datetime(prices[column]).dt.strftime("%Y-%m-%d")
    prices["execution_date"] = prices["next_date"]
    combined = prices.merge(complete, on="execution_date", how="inner", validate="one_to_one")
    if combined.empty:
        raise ValueError("No stock rows for the paired market sessions")
    # A missing price session would silently turn a one-session target into a longer horizon.
    if not (combined["date"].eq(combined["feature_date"])
            & combined["target_end_date"].eq(combined["expected_target_end_date"])).all():
        raise ValueError("Stock dates do not match adjacent NYSE sessions")
    if (combined[["next_open", "target_end_open"]] <= 0).any().any():
        raise ValueError("Entry and target prices must be positive")

    for timing, assigned, join_date in [
        ("close", close_assigned, "feature_date"),
        ("preopen", preopen_assigned, "execution_date"),
    ]:
        features = aggregate_news_by_session(assigned)
        if assigned.empty:
            features["latest_available_at"] = pd.Series(dtype="datetime64[ns, UTC]")
        else:
            latest = assigned.groupby("session_date")["available_at"].max()
            features["latest_available_at"] = features["session_date"].map(latest)
        features[join_date] = pd.to_datetime(features["session_date"]).dt.strftime("%Y-%m-%d")
        renames = {column: f"{timing}_{column}" for column in NEWS_FEATURE_COLUMNS}
        renames["latest_available_at"] = f"{timing}_latest_available_at"
        features = features.rename(columns=renames)
        combined = combined.merge(
            features[[join_date] + list(renames.values())],
            on=join_date, how="left", validate="many_to_one",
        )
        columns = [f"{timing}_{column}" for column in NEWS_FEATURE_COLUMNS]
        combined[columns] = combined[columns].astype(float).fillna(0)
        combined[f"{timing}_news_count"] = combined[f"{timing}_news_count"].astype(int)
    return combined.sort_values("decision_at").reset_index(drop=True)


def prepare_dataset_with_news_timing(ticker, start_date, end_date, minutes_before_open=5,
                                     schedule=None, horizon=1):
    ticker = ticker.upper()
    news_df = load_scored_news_range(ticker, start_date, end_date)
    prices = get_stock_data(ticker, years=None)
    if prices.empty:
        raise ValueError(f"No stock data for {ticker}")
    if schedule is None:
        start = pd.to_datetime(start_date, utc=True) - pd.DateOffset(days=14)
        end = pd.to_datetime(end_date, utc=True) + pd.DateOffset(days=14)
        schedule = build_market_schedule(start.date().isoformat(), end.date().isoformat())
    combined = build_news_timing_dataset(
        prepare_dataset(prices, horizon=horizon), news_df, schedule, start_date, end_date, minutes_before_open, horizon=horizon
    )
    combined["ticker"] = ticker
    return combined

if __name__ == "__main__":
    feature_columns = FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS

    for ticker in ["AAPL", "MSFT"]:
        data = prepare_dataset_with_news(
            ticker,
            "2023-01-01",
            "2023-04-01",
        )

        print(ticker,"Wiersze:", len(data),"Braki:", data[feature_columns + ["target"]].isna().sum().sum())
