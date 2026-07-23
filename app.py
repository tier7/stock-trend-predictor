from math import isfinite
from flask import Flask, render_template, request, redirect, url_for, jsonify
from services.db import create_tables, get_stock_data, get_latest_stock_summary, get_recent_stock_prices
from services.data_updater import ensure_stock_data
from services.resample_service import resample_data
from services.company_service import ensure_stock_name
from services.technical_indicator_service import get_latest_indicators, calculate_all_indicators
from services.ml.model_service import train_and_predict
import re



app = Flask(__name__)

FEATURED_STOCKS = (
    {"ticker": "AAPL", "name": "Apple"},
    {"ticker": "NVDA", "name": "Nvidia"},
    {"ticker": "GOOGL", "name": "Google"},
    {"ticker": "MSFT", "name": "Microsoft"},
    {"ticker": "AMZN", "name": "Amazon"},
    {"ticker": "META", "name": "Meta"},
)


def is_valid_ticker_format(ticker):
    return re.match(r"^[A-Z0-9.\-]{1,15}$", ticker) is not None

def build_sparkline_points(values, width=184, height=56, padding=4):
    if not values:
        values = [1, 1]
    elif len(values) == 1:
        values = [values[0], values[0]]

    min_value = min(values)
    max_value = max(values)
    value_range = max_value - min_value
    x_step = (width - padding * 2) / (len(values) - 1)
    points = []

    for index, value in enumerate(values):
        x = padding + index * x_step
        if value_range == 0:
            y = height / 2
        else:
            y = height - padding - ((value - min_value) / value_range) * (height - padding * 2)
        points.append(f"{x:.1f},{y:.1f}")

    return " ".join(points)

def get_featured_stocks():
    featured_stocks = []

    for stock in FEATURED_STOCKS:
        ensure_stock_data(stock["ticker"], "1d")
        recent_prices = get_recent_stock_prices(stock["ticker"], "1d", 7)
        closes = []

        if not recent_prices.empty:
            for value in recent_prices["close"].tolist():
                close = float(value)
                if isfinite(close):
                    closes.append(close)

        change_percent = None
        if len(closes) > 1 and closes[0] != 0:
            change_percent = ((closes[-1] - closes[0]) / closes[0]) * 100

        if change_percent is None:
            trend_class = "is-flat"
            change_label = "brak danych"
        elif change_percent >= 0:
            trend_class = "is-up"
            change_label = f"+{change_percent:.1f}%"
        else:
            trend_class = "is-down"
            change_label = f"{change_percent:.1f}%"

        featured_stocks.append({
            "ticker": stock["ticker"],
            "name": stock["name"],
            "change_label": change_label,
            "sparkline_points": build_sparkline_points(closes),
            "trend_class": trend_class,
        })

    return featured_stocks

@app.route("/", methods=["GET", "POST"])
def index():
    error = request.args.get("error")
    featured_stocks = get_featured_stocks()
    if request.method == "POST":
        ticker = (request.form.get("ticker") or "").upper().strip()

        if not ticker:
            return render_template("index.html", error="No ticker provided", featured_stocks=featured_stocks)
        if not is_valid_ticker_format(ticker):
            return render_template("index.html", error="Invalid ticker format", featured_stocks=featured_stocks)

        return redirect(url_for("stock", ticker=ticker))

    return render_template("index.html", error=error, featured_stocks=featured_stocks)


@app.route("/stock/<ticker>")
def stock(ticker):
    ticker = ticker.upper().strip()
    if not is_valid_ticker_format(ticker):
        return redirect(url_for("index", error="Invalid ticker format"))

    has_data = ensure_stock_data(ticker, "1d")
    if not has_data:
        return redirect(url_for("index", error="No data found for ticker " + ticker))


    company_name = ensure_stock_name(ticker)
    summary = get_latest_stock_summary(ticker, "1d")

    stock_info = {
        "ticker": ticker,
        "name": company_name,
        "last_price": round(summary["last_price"], 2),
        "volume": summary["volume"],
        "last_date": summary["last_date"],
        "price_change": round(summary["price_change"], 2) if summary["price_change"] is not None else "-",
        "price_change_percent": round(summary["price_change_percent"], 2) if summary["price_change_percent"] is not None else "-",
        "prediction": "-"
    }
    stock_indicators = get_latest_indicators(get_stock_data(ticker, "1d", None))
    for indicator in stock_indicators:
        stock_indicators[indicator] = round(stock_indicators[indicator], 2)
    #chart_html = create_candlestick_chart(df, ticker)
    return render_template("stock.html", data=stock_info, indicators=stock_indicators)

@app.route("/api/stock/<ticker>/candles")
def stock_candles_api(ticker):
    ticker = ticker.upper()
    success = ensure_stock_data(ticker, "1d")
    if not success:
        return jsonify({"error": f"No data found for ticker {ticker}"}), 404

    df = get_stock_data(ticker, "1d", None)
    if df.empty:
        return jsonify({"error": f"No data found for ticker {ticker}"}), 404

    interval = request.args.get("interval", "1d")
    df = resample_data(df,interval)

    candles_df = df[["date", "open", "high", "low", "close"]].copy()
    candles_df = candles_df.rename(columns={"date": "time"})

    candles_df["open"] = candles_df["open"].astype(float)
    candles_df["high"] = candles_df["high"].astype(float)
    candles_df["low"] = candles_df["low"].astype(float)
    candles_df["close"] = candles_df["close"].astype(float)

    candles = candles_df.to_dict(orient="records")
    return jsonify(candles)

@app.route("/api/stock/<ticker>/indicators")
def stock_indicators_api(ticker):
    ticker = ticker.upper()
    success = ensure_stock_data(ticker, "1d")
    if not success:
        return jsonify({"error": f"No data found for ticker {ticker}"}), 404

    df = get_stock_data(ticker, "1d", None)
    if df.empty:
        return jsonify({"error": f"No data found for ticker {ticker}"}), 404

    interval = request.args.get("interval", "1d")
    df = resample_data(df, interval)
    indicators = calculate_all_indicators(df)

    indicator_columns = [
        "sma_12",
        "sma_26",
        "ema_12",
        "ema_26",
        "bollinger_mid",
        "bollinger_upper",
        "bollinger_lower",
    ]
    result = {}

    for column in indicator_columns:
        series_df = indicators[["date", column]].dropna().copy()
        series_df = series_df.rename(columns={
            "date": "time",
            column: "value"
        })
        series_df["time"] = series_df["time"].astype(str)
        series_df["value"] = series_df["value"].astype(float)

        result[column] = series_df.to_dict(orient="records")

    return jsonify(result)

@app.route("/api/stock/<ticker>/prediction")
def stock_prediction_api(ticker):
    ticker = ticker.upper()
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if not start_date or not end_date:
        return jsonify({"error": "Missing start or end date"}), 400

    success = ensure_stock_data(ticker, "1d")
    if not success:
        return jsonify({"error": f"No data found for ticker {ticker}"}), 404

    df = get_stock_data(ticker, "1d", None)
    if df.empty:
        return jsonify({"error": f"No data found for ticker {ticker}"}), 404

    try:
        predictions = train_and_predict(df, start_date, end_date)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(predictions)



if __name__ == "__main__":
    create_tables()
    app.run(debug=True, host="0.0.0.0", port=5001)
