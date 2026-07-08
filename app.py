from flask import Flask, render_template, request, redirect, url_for, jsonify
from services.db import get_stock_data
from services.data_updater import ensure_stock_data
from services.resample_service import resample_data
from services.chart_service import create_candlestick_chart
app = Flask(__name__)

COMPANIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla",
    "NVDA": "Nvidia"
}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form.get("ticker")
        return redirect(url_for("stock", ticker=ticker))

    return render_template("index.html", companies=COMPANIES)


@app.route("/stock/<ticker>")
def stock(ticker):
    company_name = COMPANIES.get(ticker, "Nieznana spółka")

    ensure_stock_data(ticker, "1d")
    df = get_stock_data(ticker, "1d", 5)

    sample_data = {
        "ticker": ticker,
        "name": company_name,
        "last_price": 189.25,
        "volume": 12345678,
        "trend": "wzrostowy",
        "prediction": "cena prawdopodobnie wzrośnie"
    }
    #chart_html = create_candlestick_chart(df, ticker)
    return render_template("stock.html", data=sample_data)

@app.route("/api/stock/<ticker>/candles")
def stock_candles_api(ticker):
    ticker = ticker.upper()
    ensure_stock_data(ticker, "1d")
    df = get_stock_data(ticker, "1d", None)

    if df.empty:
        return jsonify([])
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