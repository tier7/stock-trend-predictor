from flask import Flask, render_template, request, redirect, url_for
from services.db import get_stock_data
from services.data_updater import ensure_stock_data
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
    chart_html = create_candlestick_chart(df, ticker)
    return render_template("stock.html",data=sample_data,chart_html=chart_html)


if __name__ == "__main__":
    app.run(debug=True)