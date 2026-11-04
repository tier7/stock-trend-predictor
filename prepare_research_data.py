import argparse
import json
import os
from pathlib import Path
from time import sleep
from uuid import uuid4
import pandas as pd
from services.data_fetcher import fetch_stock_data
from services.db import create_tables, get_news_imports, get_news_with_sentiment, get_news_without_sentiment, get_stock_data, save_stock_data
from services.ml.evaluation_universe import ALL_TICKERS, EVALUATION_PERIODS, SECTOR_ASSIGNMENTS
from services.ml.finbert_service import FinbertAnalyzer, MODEL_NAME, MODEL_REVISION
from services.ml.market_feature_service import MARKET_FEATURE_POLICY
from services.ml.news_feature_service import get_news_history_start
from services.ml.prediction_time_service import build_market_schedule
from services.ml.sector_feature_service import resolve_sector_benchmarks
from services.news_fetcher import NewsPreparationError, build_month_blocks, classify_news_relevance, find_missing_news_ranges, import_news_range, score_pending_news
from services.news_symbol_service import normalize_news_ticker
from services.stock_data_quality_service import filter_valid_ohlc


def inspect_prices(ticker, expected_dates, start_date, end_date):
    data = get_stock_data(ticker, interval="1d", years=None)
    if not data.empty:
        data = data.loc[data["date"].ge(start_date) & data["date"].lt(end_date)]
    valid = filter_valid_ohlc(data)
    dates = set(valid["date"]) if not valid.empty else set()
    missing = sorted(set(expected_dates) - dates)
    return {
        "complete": not missing,
        "expected_sessions": len(expected_dates),
        "valid_sessions": len(set(expected_dates) & dates),
        "invalid_rows": len(data) - len(valid),
        "missing_sessions": len(missing),
        "missing_dates_sample": missing[:20]
    }


def inspect_news(ticker, start_date, end_date):
    missing = find_missing_news_ranges(get_news_imports(ticker, start_date, end_date), start_date, end_date)
    pending = get_news_without_sentiment(ticker, MODEL_NAME, MODEL_REVISION, start_date=start_date, end_date=end_date)
    eligible_pending = sum(classify_news_relevance(article, ticker)["use_for_model"] for article in pending)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    scored = get_news_with_sentiment(ticker, MODEL_NAME, MODEL_REVISION)
    eligible_scored = sum(
        start <= max(pd.to_datetime(article["created_at"], utc=True), pd.to_datetime(article["updated_at"], utc=True)) < end
        and classify_news_relevance(article, ticker)["use_for_model"] for article in scored)
    return {
        "imports_complete": not missing,
        "sentiments_complete": not missing and eligible_pending == 0,
        "missing_ranges": missing,
        "pending_candidate_versions": len(pending),
        "eligible_without_sentiment": eligible_pending,
        "eligible_scored_versions": eligible_scored
    }


def save_report(report, path):
    report["updated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    current_path = Path(report.get("report_path", path))
    fallback = path.with_name(f"{path.stem}.{uuid4().hex[:8]}{path.suffix}")
    for index, destination in enumerate((current_path, fallback)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        report["report_path"] = str(destination)
        temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for attempt in range(5):
            try:
                temporary.replace(destination)
            except PermissionError:
                if attempt == 4:
                    if index == 1:
                        raise
                    break
                sleep(0.1 * 2 ** attempt)
            else:
                if index == 1:
                    print(f"Report file locked. Continuing with report: {destination}", flush=True)
                return destination


def prepare_research_data(phase, tickers, report_path):
    tickers = list(dict.fromkeys(normalize_news_ticker(ticker) for ticker in tickers))
    if not tickers or set(tickers) - set(ALL_TICKERS):
        raise ValueError("Choose tickers from the approved evaluation universe")
    train_start = EVALUATION_PERIODS["train_start"]
    end_date = EVALUATION_PERIODS["test_end"]
    price_start = (pd.Timestamp(train_start) - pd.DateOffset(months=3)).date().isoformat()
    news_start = get_news_history_start(train_start, end_date)
    news_end = pd.to_datetime(end_date, utc=True).isoformat()
    sectors = resolve_sector_benchmarks(tickers, SECTOR_ASSIGNMENTS)
    sources = {ticker: "NYSE" for ticker in tickers}
    sources.update({benchmark: "NYSE" for benchmark in sorted(set(sectors.values()))})
    sources[MARKET_FEATURE_POLICY["benchmark_ticker"]] = MARKET_FEATURE_POLICY["benchmark_calendar"]
    sources[MARKET_FEATURE_POLICY["volatility_ticker"]] = MARKET_FEATURE_POLICY["volatility_calendar"]
    expected = {
        name: build_market_schedule(price_start, (pd.Timestamp(end_date) - pd.Timedelta(days=1)).date().isoformat(), calendar_name=name).index.strftime("%Y-%m-%d").tolist()
        for name in set(sources.values())
    }
    report = {
        "status": "running",
        "phase": phase,
        "pid": os.getpid(),
        "started_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "periods": dict(EVALUATION_PERIODS),
        "tickers": tickers,
        "price_sources": sources,
        "price_start": price_start,
        "news_start": news_start,
        "end": news_end,
        "sentiment_model": {"name": MODEL_NAME, "revision": MODEL_REVISION},
        "dataset_complete": False,
        "prices": {},
        "news": {},
        "sentiments": {}
    }

    def progress(message):
        report["current"] = message
        save_report(report, report_path)
        print(message, flush=True)

    try:
        if phase != "check":
            create_tables()
        progress(f"Preparing {len(tickers)} companies and {len(sources)} price sources; phase={phase}.")
        if phase in ("all", "prices"):
            for index, (ticker, calendar) in enumerate(sources.items(), 1):
                progress(f"Prices {index}/{len(sources)}: {ticker}.")
                state = inspect_prices(ticker, expected[calendar], price_start, end_date)
                if not state["complete"]:
                    for attempt in range(3):
                        data = fetch_stock_data(ticker, start_date=price_start, end_date=end_date)
                        if not data.empty:
                            data = data.loc[data["date"].ge(price_start) & data["date"].lt(end_date)]
                            save_stock_data(data)
                        state = inspect_prices(ticker, expected[calendar], price_start, end_date)
                        if state["complete"]:
                            break
                        if attempt < 2:
                            progress(f"{ticker}: Retrying missing price sessions, attempt {attempt + 2}/3.")
                            sleep(2 ** (attempt + 1))
                report["prices"][ticker] = state
                save_report(report, report_path)
                if not state["complete"]:
                    raise ValueError(f"{ticker}: Missing {state['missing_sessions']} price sessions")
        if phase in ("all", "news"):
            for index, ticker in enumerate(tickers, 1):
                progress(f"News {index}/{len(tickers)}: {ticker}.")
                report["news"][ticker] = import_news_range(ticker, news_start, news_end, progress_callback=progress)
                save_report(report, report_path)
        if phase in ("all", "sentiments"):
            analyzer = FinbertAnalyzer()
            periods = build_month_blocks(news_start, news_end)
            for index, ticker in enumerate(tickers, 1):
                report["sentiments"][ticker] = {"inserted_count": 0, "completed_blocks": 0}
                for period in periods:
                    start = period["start_date"]
                    end = period["end_date"]
                    if find_missing_news_ranges(get_news_imports(ticker, start, end), start, end):
                        raise ValueError(f"{ticker}: Import news before scoring {start} to {end}")
                    progress(f"Sentiments {index}/{len(tickers)}: {ticker}, {start} to {end}.")
                    inserted = score_pending_news(ticker, analyzer, start_date=start, end_date=end, progress_callback=progress)
                    report["sentiments"][ticker]["inserted_count"] += inserted
                    report["sentiments"][ticker]["completed_blocks"] += 1
                    save_report(report, report_path)
        if phase in ("all", "check"):
            for ticker, calendar in sources.items():
                progress(f"Checking prices: {ticker}.")
                report["prices"][ticker] = inspect_prices(ticker, expected[calendar], price_start, end_date)
            for ticker in tickers:
                progress(f"Checking news and sentiments: {ticker}.")
                report["news"][ticker] = inspect_news(ticker, news_start, news_end)
            report["dataset_complete"] = all(state["complete"] for state in report["prices"].values()) and all(state["sentiments_complete"] for state in report["news"].values())
            report["status"] = "completed" if report["dataset_complete"] else "incomplete"
        else:
            report["status"] = "completed"
        report["finished_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        save_report(report, report_path)
        print(json.dumps({"status": report["status"], "phase": phase, "dataset_complete": report["dataset_complete"], "report": report.get("report_path", str(report_path))}, ensure_ascii=False), flush=True)
        return report
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error_type"] = type(error).__name__
        if isinstance(error, (ValueError, NewsPreparationError)):
            report["error"] = str(error)
        try:
            save_report(report, report_path)
        except OSError as report_error:
            report["report_save_error"] = type(report_error).__name__
            print(f"Cannot save failure report: {type(report_error).__name__}", flush=True)
        print(json.dumps({"status": report["status"], "error_type": report["error_type"], "current": report.get("current"), "report": report.get("report_path", str(report_path))}, ensure_ascii=False), flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description="Uzupełnij ceny, newsy i oceny FinBERT dla zatwierdzonego protokołu badania.")
    parser.add_argument("--phase", choices=("all", "prices", "news", "sentiments", "check"), default="all")
    parser.add_argument("--tickers", nargs="+", default=list(ALL_TICKERS))
    parser.add_argument("--report", type=Path, default=Path("reports/data_preparation/status.json"))
    arguments = parser.parse_args()
    try:
        report = prepare_research_data(arguments.phase, arguments.tickers, arguments.report.resolve())
    except ValueError as error:
        parser.error(str(error))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
