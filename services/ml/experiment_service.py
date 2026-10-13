
from services.db import get_stock_data
from services.ml.global_forest_service import simulate_with_global_model
from services.ml.model_service import train_and_predict
from services.ml.model_store_service import get_active_global_model
import pandas as pd

def build_result_row(ticker, model_type, result):
    return {
        "ticker": ticker,
        "model_type": model_type,
        "start_date": result["effective_test_start_date"],
        "end_date": result["effective_test_end_date"],
        "confidence_threshold": result["model_summary"]["confidence_threshold"],
        "accuracy": result["accuracy"],
        "baseline_accuracy": result["baseline_accuracy"],
        "model_return": result["model_summary"]["total_return"],
        "model_drawdown": result["model_summary"]["max_drawdown"],
        "buy_hold_return": result["buyandhold_summary"]["total_return"],
        "buy_hold_drawdown": result["buyandhold_summary"]["max_drawdown"],
        "excess_return": (result["model_summary"]["total_return"]- result["buyandhold_summary"]["total_return"]),
        "model_exposure": result["model_summary"]["market_exposure"],
        "transactions": result["model_summary"]["total_transactions"],
    }

def run_simulation(ticker, model_type, start_date, end_date, global_model_id=None, initial_cash=10000, fee_rate=0.001, confidence_threshold=0.6):
    data = get_stock_data(ticker,years=None)
    if data.empty:
        raise ValueError("No data")

    if model_type == "global":
        if global_model_id is None:
            metadata = get_active_global_model()
            global_model_id = metadata["model_id"]
        result = simulate_with_global_model(
            data,
            model_id=global_model_id,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            confidence_threshold=confidence_threshold,
        )
    elif model_type == "local":
        result = train_and_predict(
            data,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            confidence_threshold=confidence_threshold,
        )
    else:
        raise ValueError("model_type must be either 'global' or 'local'")

    row = build_result_row(ticker, model_type, result)
    row["model_id"] = global_model_id if model_type == "global" else None
    row["initial_cash"] = initial_cash
    row["fee_rate"] = fee_rate
    return row

def run_experiments(tickers, model_types, periods, global_model_id=None, initial_cash=10000, fee_rate = 0.001, confidence_threshold=0.6):
    rows = []
    if "global" in model_types and global_model_id is None:
        metadata = get_active_global_model()
        global_model_id = metadata["model_id"]

    for period in periods:
        for ticker in tickers:
            for model_type in model_types:
                row = run_simulation(ticker, model_type, period["start_date"], period["end_date"], global_model_id, initial_cash, fee_rate, confidence_threshold)
                row["period"] = period["name"]
                rows.append(row)

    return rows

def summarize_results(report):
    summary = []

    for model_type, group in report.groupby("model_type"):
        row = {
            "model_type": model_type,
            "experiments": len(group),
            "wins_vs_buy_hold": len(group[(group["excess_return"] > 0)]),
            "profitable_runs": len(group[(group["model_return"] > 0)]),
            "mean_excess_return": group["excess_return"].mean(),
            "mean_drawdown": group["model_drawdown"].mean(),
        }
        summary.append(row)

    return pd.DataFrame(summary)