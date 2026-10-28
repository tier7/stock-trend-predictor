import pandas as pd
import pandas_market_calendars as mcal

DEFAULT_MARKET_CALENDAR = "NYSE"
NEWS_MINUTES_BEFORE_OPEN = 5
MARKET_CLOSE_BUFFER_MINUTES = 1

def build_market_schedule(start_date, end_date, calendar_name=DEFAULT_MARKET_CALENDAR):
    calendar = mcal.get_calendar(calendar_name)

    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    return schedule

def build_preopen_schedule(schedule, minutes_before_open=NEWS_MINUTES_BEFORE_OPEN):
    if not isinstance(minutes_before_open, int) or minutes_before_open <= 0:
        raise ValueError("minutes_before_open must be a positive integer")
    sessions = schedule.sort_index().copy()
    sessions.index.name = "session_date"
    dates = pd.Series(sessions.index.strftime("%Y-%m-%d"), index=sessions.index)
    sessions["execution_date"] = dates
    sessions["feature_date"] = dates.shift(1)
    sessions["previous_close"] = sessions["market_close"].shift(1)
    sessions["decision_at"] = sessions["market_open"].map(
        lambda opening: opening - pd.DateOffset(minutes=minutes_before_open)
    )

    if (sessions["decision_at"] <= sessions["previous_close"]).any():
        raise ValueError("Decision must fall after the previous market close")
    return sessions.reset_index()

def get_latest_prediction_window(feature_date, feature_groups, as_of=None):
    now = pd.Timestamp.now(tz="UTC")
    snapshot_at = now if as_of is None else pd.to_datetime(as_of, utc=True)

    if pd.isna(snapshot_at) or snapshot_at > now:
        raise ValueError("as_of must be a valid time and not in the future")

    feature_date = pd.Timestamp(feature_date)
    if pd.isna(feature_date):
        raise ValueError("Missing feature date")

    feature_date = feature_date.strftime("%Y-%m-%d")
    calendar_end = pd.Timestamp(feature_date) + pd.DateOffset(days=14)

    schedule = build_market_schedule(feature_date, calendar_end)
    sessions = build_preopen_schedule(schedule)
    matching = sessions.loc[sessions["feature_date"].eq(feature_date)]
    if matching.empty:
        raise ValueError("No matching sessions found")

    session = matching.iloc[0]
    uses_news = "news" in feature_groups
    decision_at = session["previous_close"]

    if uses_news:
        decision_at = session["decision_at"]
    if "sector" in feature_groups:
        sector_ready_at = session["previous_close"] + pd.Timedelta(minutes=MARKET_CLOSE_BUFFER_MINUTES)
        decision_at = max(decision_at, sector_ready_at)
    if "market" in feature_groups:
        vix_schedule = build_market_schedule(feature_date, feature_date, calendar_name="CBOE_Index_Options")

        if vix_schedule.empty:
            raise ValueError("No VIX session for the feature date")

        market_ready_at = max(session["previous_close"], vix_schedule["market_close"].iloc[0]) + pd.Timedelta(minutes=MARKET_CLOSE_BUFFER_MINUTES)
        decision_at = max(decision_at, market_ready_at)
    if snapshot_at < decision_at or (uses_news and snapshot_at == decision_at):
        raise ValueError(f"Forecast data is not ready yet: {decision_at}")
    if snapshot_at >= session["market_open"]:
        raise ValueError("The entry session has already opened")


    return {
        "execution_date": session["execution_date"],
        "previous_close": session["previous_close"].isoformat(),
        "decision_at": decision_at.isoformat(),
        "as_of": snapshot_at.isoformat(),
        "market_open": session["market_open"].isoformat()
    }
