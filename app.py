from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
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
    sample_data = {
        "ticker": ticker,
        "name": company_name,
        "last_price": 189.25,
        "volume": 12345678,
        "trend": "wzrostowy",
        "prediction": "cena prawdopodobnie wzrośnie"
    }
    data = yf.download(ticker, period="10d", interval="1d")
    print(data)
    return render_template("stock.html", data=sample_data)


if __name__ == "__main__":
    app.run(debug=True)