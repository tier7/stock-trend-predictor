import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


BASE_URL = "http://localhost:5001"
START_DATE = "2025-10-01"
END_DATE = "2026-07-22"

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
]

def evaluate_ticker(ticker):
    query = urlencode({
        "start_date": START_DATE,
        "end_date": END_DATE,
    })

    url = f"{BASE_URL}/api/stock/{ticker}/prediction?{query}"

    with urlopen(url, timeout=600) as response:
        data = json.load(response)

    model_summary = data["model_summary"]
    buy_and_hold_summary = data["buyandhold_summary"]

    return {
        "ticker": ticker,
        "accuracy": data["accuracy"],
        "baseline": data["baseline_accuracy"],
        "precision": data["precision"],
        "recall": data["recall"],
        "f1": data["f1"],
        "correct": data["correct_predictions"],
        "total": data["total_predictions"],
        "model_max_drawdown": model_summary["max_drawdown"],
        "buy_hold_max_drawdown": buy_and_hold_summary["max_drawdown"],

        "strategy_return": model_summary["total_return"],
        "buy_hold_return": buy_and_hold_summary["total_return"],
        "return_difference": (
                model_summary["total_return"]
                - buy_and_hold_summary["total_return"]
        ),
        "final_value": model_summary["final_value"],
        "transactions": model_summary["total_transactions"],
    }

def format_percent(value):
    return f"{value * 100:.2f}%"

def print_results(results):
    print()
    print(
        f"{'Ticker':<8}"
        f"{'Accuracy':>12}"
        f"{'Baseline':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Correct':>12}"
        f"{'Model DD':>12}"
        f"{'B&H DD':>12}"

    )

    for result in results:
        correct = f"{result['correct']}/{result['total']}"

        print(
            f"{result['ticker']:<8}"
            f"{format_percent(result['accuracy']):>12}"
            f"{format_percent(result['baseline']):>12}"
            f"{format_percent(result['precision']):>12}"
            f"{format_percent(result['recall']):>12}"
            f"{format_percent(result['f1']):>12}"
            f"{correct:>12}"
            f"{format_percent(result['model_max_drawdown']):>12}"
            f"{format_percent(result['buy_hold_max_drawdown']):>12}"
        )
def print_backtest_results(results):
    print()
    print("Wyniki symulatora")
    print(
        f"{'Ticker':<8}"
        f"{'Model':>12}"
        f"{'Buy and hold':>12}"
        f"{'Difference':>12}"
        f"{'Final value':>14}"
        f"{'Transactions':>14}"
    )

    for result in results:
        print(
            f"{result['ticker']:<8}"
            f"{format_percent(result['strategy_return']):>12}"
            f"{format_percent(result['buy_hold_return']):>12}"
            f"{format_percent(result['return_difference']):>12}"
            f"{result['final_value']:>14.2f}"
            f"{result['transactions']:>14}"
        )
if __name__ == "__main__":
    results = []
    for ticker in TICKERS:
        print(f"Testowanie {ticker}...")

        try:
            result = evaluate_ticker(ticker)
            results.append(result)
        except (HTTPError, URLError, TimeoutError) as error:
            print(f"Nie udało się przetestować {ticker}: {error}")

    print_results(results)
    print_backtest_results(results)