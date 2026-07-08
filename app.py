from flask import Flask, render_template, request, redirect, url_for, jsonify
from services.db import get_stock_data
from services.data_updater import ensure_stock_data
from services.resample_service import resample_data
from services.company_service import ensure_stock_name
import re

app = Flask(__name__)


def is_valid_ticker_format(ticker):
    return re.match(r"^[A-Z0-9.\-]{1,15}$", ticker) is not None

@app.route("/", methods=["GET", "POST"])
def index():
    error = request.args.get("error")
    if request.method == "POST":
        ticker = request.form.get("ticker")
        ticker = ticker.upper().strip()

        if not ticker:
            return render_template("index.html", error="No ticker provided")
        if not is_valid_ticker_format(ticker):
            return render_template("index.html", error="Invalid ticker format")

        return redirect(url_for("stock", ticker=ticker))

    return render_template("index.html", error=error)


@app.route("/stock/<ticker>")
def stock(ticker):
    ticker = ticker.upper().strip()
    if not is_valid_ticker_format(ticker):
        return redirect(url_for("index", error="Invalid ticker format"))

    has_data = ensure_stock_data(ticker, "1d")
    if not has_data:
        return redirect(url_for("index", error="No data found for ticker " + ticker))


    company_name = ensure_stock_name(ticker)

    sample_data = {
        "ticker": ticker,
        "name": company_name,
        "last_price": "-",
        "volume": "-",
        "trend": "-",
        "prediction": "-"
    }
    #chart_html = create_candlestick_chart(df, ticker)
    return render_template("stock.html", data=sample_data)

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

if __name__ == "__main__":
    app.run(debug=True)