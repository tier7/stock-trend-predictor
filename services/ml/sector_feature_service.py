import json
from pathlib import Path

import pandas as pd

from services.ml.feature_schema import SECTOR_FEATURE_COLUMNS
from services.ml.market_feature_service import prepare_market_prices
from services.ml.prediction_time_service import (build_market_schedule, DEFAULT_MARKET_CALENDAR, MARKET_CLOSE_BUFFER_MINUTES)


SECTOR_BENCHMARKS = {
    "communication_services": "XLC",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "energy": "XLE",
    "financials": "XLF",
    "health_care": "XLV",
    "industrials": "XLI",
    "information_technology": "XLK",
    "materials": "XLB",
    "real_estate": "XLRE",
    "utilities": "XLU"
}

SECTOR_FEATURE_POLICY = {
    "version": 1,
    "profile": "us_equities",
    "source": "yfinance",
    "assignment_method": "explicit_reference_benchmark",
    "benchmarks_by_sector": SECTOR_BENCHMARKS.copy(),
    "calendar_name": DEFAULT_MARKET_CALENDAR,
    "close_buffer_minutes": MARKET_CLOSE_BUFFER_MINUTES,
    "price_basis": "database_close",
    "sector_return_windows": [1, 5],
    "relative_strength_windows": [5, 20],
    "relative_strength_definition": "stock_return_minus_benchmark_return",
    "alignment": "same_session_date",
    "missing_values": "keep_nan"
}

def resolve_sector_benchmarks(tickers, sector_assignments=None, *, saved_benchmarks=None):
    if sector_assignments is None:
        sector_assignments = {}

    if saved_benchmarks is None:
        saved_benchmarks = {}

    if (not isinstance(sector_assignments, dict) or not isinstance(saved_benchmarks, dict)):
        raise ValueError("Sector mappings must be dictionaries")


    benchmarks = {}
    for ticker in tickers:
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Invalid ticker")

        ticker = ticker.strip().upper()
        benchmark = saved_benchmarks.get(ticker)

        if (ticker in saved_benchmarks and benchmark not in SECTOR_BENCHMARKS.values()):
            raise ValueError(f"{ticker}: unsupported saved benchmark")
        if ticker in sector_assignments:
            sector = sector_assignments[ticker]
            if not isinstance(sector, str) or sector not in SECTOR_BENCHMARKS:
                raise ValueError(f"{ticker}: provide supported sector")
            requested = SECTOR_BENCHMARKS[sector]
            if benchmark is not None and benchmark != requested:
                raise ValueError(f"{ticker}: assignment conflicts with saved benchmark")

            benchmark = requested

        if benchmark is None:
            raise ValueError(f"{ticker}: provide an explicit sector assignment")
        benchmarks[ticker] = benchmark

    return benchmarks

def load_sector_assignments(path):
    if path is None:
        return None

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Nie można wczytać mapy sektorów {path}: {error}") from error

    if not isinstance(data, dict) or not data:
        raise ValueError("Mapa sektorów musi być niepustym obiektem JSON: ticker -> sektor")

    assignments = {}
    for ticker, sector in data.items():
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            raise ValueError("Mapa sektorów zawiera pusty ticker")
        if normalized_ticker in assignments:
            raise ValueError(f"Powtórzony ticker w mapie sektorów: {normalized_ticker}")
        assignments[normalized_ticker] = sector

    resolve_sector_benchmarks(assignments, assignments)
    return assignments

def build_sector_features(stock_data, benchmark_data):
    stock = (prepare_market_prices(stock_data, "Stock")).set_index("date")["close"]
    benchmark = (prepare_market_prices(benchmark_data, "Benchmark")).set_index("date")["close"]

    schedule = build_market_schedule((min(stock.index.min(), benchmark.index.min())), (max(stock.index.max(), benchmark.index.max())))
    sessions = schedule.index.rename("date")

    if not stock.index.isin(sessions).all() or not benchmark.index.isin(sessions).all():
        raise ValueError("Prices contains non session dates")
    stock = stock.reindex(sessions)
    benchmark = benchmark.reindex(sessions)

    features = pd.DataFrame(index=sessions)

    features["sector_return_1d"] = benchmark / benchmark.shift(1) - 1
    features["sector_return_5d"] = benchmark / benchmark.shift(5) - 1

    features["relative_strength_5d"] = (stock / stock.shift(5) - 1) - features["sector_return_5d"]
    features["relative_strength_20d"] = (stock / stock.shift(20) - 1) - (benchmark / benchmark.shift(20) - 1)

    features = features.reset_index()[["date"] + SECTOR_FEATURE_COLUMNS]
    return features


def extract_sector_benchmarks(data):
    required_columns = ["ticker", "sector_benchmark"]
    missing_columns = [column for column in required_columns if column not in data.columns]

    if data.empty or missing_columns:
        raise ValueError(
            "Training data must contain: tickers, sector benchmarks")

    pairs = data[required_columns].drop_duplicates().copy()
    pairs["ticker"] = pairs["ticker"].str.strip().str.upper()

    if pairs.isna().any().any() or pairs["ticker"].eq("").any():
        raise ValueError("Missing ticker or sector benchmark")

    if not pairs["sector_benchmark"].isin(SECTOR_BENCHMARKS.values()).all():
        raise ValueError("Unsupported sector benchmark")

    pairs = pairs.drop_duplicates()

    if pairs["ticker"].duplicated().any():
        raise ValueError("Each ticker must have one sector benchmark")

    return pairs.set_index("ticker")["sector_benchmark"].to_dict()
