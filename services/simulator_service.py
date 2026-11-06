from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from services.ml.evaluation_service import evaluate
from services.strategy_service import should_enter_position
from dataclasses import asdict, replace

def calculate_max_drawdown(equity_history, initial_cash):
    max_wallet_value = initial_cash
    max_drawdown = 0
    for trade in equity_history:
        wallet_value = float(trade["value"])
        if wallet_value > max_wallet_value:
            max_wallet_value = trade["value"]
        drawdown = (max_wallet_value - wallet_value) / max_wallet_value
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

def calculate_cagr(equity_history, initial_value, final_value):
    start_date = date.fromisoformat(equity_history[0]["date"])
    end_date = date.fromisoformat(equity_history[-1]["date"])
    days = (end_date - start_date).days

    if days <= 0:
        raise ValueError("CAGR requires at least two different dates.")

    years = days / 365.25
    cagr = (final_value / initial_value) ** (1 / years) - 1
    return float(cagr)

def days_between(start_date, end_date):
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    return (end - start).days

def build_closed_trades(trades):
    closed_trades = []
    current_buy = None

    for trade in trades:
        if trade["type"] == "BUY":
            current_buy = trade
        elif trade["type"] == "SELL" and current_buy is not None:
            shares = trade["shares"]
            buy_value = current_buy["price"] * shares
            buy_cost = buy_value + current_buy["fee"]
            sell_value = trade["price"] * shares
            sell_proceeds = sell_value - trade["fee"]
            profit = sell_proceeds - buy_cost
            return_rate = profit / buy_cost

            closed_trade = {
                "buy_date": current_buy["date"],
                "sell_date": trade["date"],
                "buy_price": current_buy["price"],
                "sell_price": trade["price"],
                "shares": trade["shares"],
                "buy_fee": current_buy["fee"],
                "sell_fee": trade["fee"],
                "total_fees": trade["fee"] + current_buy["fee"],
                "profit": profit,
                "return_rate": return_rate,
                "holding_days": days_between(current_buy["date"], trade["date"])
            }
            closed_trades.append(closed_trade)
            current_buy = None
    return closed_trades

def calculate_trades_stats(closed_trades):
    profitable_trades = 0
    losing_trades = 0

    for trade in closed_trades:
        profit = float(trade["profit"])

        if profit > 0:
            profitable_trades += 1
        else:
            losing_trades += 1

    total_trades = len(closed_trades)
    if total_trades > 0:
        win_rate = profitable_trades / total_trades
    else:
        win_rate = 0.0

    return profitable_trades, losing_trades, win_rate

def prepare_simulation_prices(prices_df, start_date, end_date):
    if prices_df.empty:
        raise ValueError("Simulation prices cannot be empty")

    required_columns = ["date", "open", "close"]
    missing_columns = [column for column in required_columns if column not in prices_df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    prices = prices_df[required_columns].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="raise").dt.normalize()

    if prices["date"].isna().any():
        raise ValueError("Missing price dates")

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("Invalid simulation date range")

    prices = prices.loc[prices["date"].between(start, end)].sort_values("date").copy()

    if prices.empty:
        raise ValueError("No prices in the simulation range")
    if prices["date"].duplicated().any():
        raise ValueError("Duplicate price dates")

    price_columns = ["open", "close"]
    prices[price_columns] = prices[price_columns].apply(pd.to_numeric, errors="raise")
    values = prices[price_columns].to_numpy(dtype=float)

    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be finite and positive")
    prices = prices.set_index("date")
    if start not in prices.index or end not in prices.index:
        raise ValueError("Missing first or last simulation session")

    return prices

def calculate_buy(cash, open_price, strategy_config):
    budget = cash * strategy_config.position_fraction
    execution_price = open_price * (1 + strategy_config.slippage_rate)

    trade_value = budget / (1+strategy_config.fee_rate)
    fee = trade_value * strategy_config.fee_rate
    shares = trade_value / execution_price

    result = {
        "cash_remaining": cash-budget,
        "shares": shares,
        "price": execution_price,
        "fee": fee
    }

    return result

def calculate_sell(shares, open_price, strategy_config):
    execution_price = open_price * (1 - strategy_config.slippage_rate)

    gross_value = shares*execution_price
    fee = gross_value * strategy_config.fee_rate
    proceeds = gross_value - fee

    result = {
        "proceeds": proceeds,
        "price": execution_price,
        "fee": fee
    }
    return result

def run_backtest(predictions_df, prices_df, model_config, strategy_config, initial_cash=10000):
    initial_cash = float(initial_cash)

    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("Initial cash must be finite and positive")
    if predictions_df.empty:
        raise ValueError("No predictions available for backtest")
    if "ticker" in predictions_df and predictions_df["ticker"].nunique() > 1:
        raise ValueError("Simulate one ticker at a time")

    prediction_column = ("probability_up" if model_config.task == "classification" else "predicted_return")

    required_columns = ["next_date", "target_end_date", prediction_column]
    missing_columns = [column for column in required_columns if column not in predictions_df.columns]

    if missing_columns:
        raise ValueError(f"Missing prediction columns: {missing_columns}")

    predictions = predictions_df.copy()

    for column in ("next_date", "target_end_date"):
        predictions[column] = pd.to_datetime(predictions[column], errors="raise").dt.normalize()

    if predictions[["next_date", "target_end_date"]].isna().any().any():
        raise ValueError("Missing prediction dates")
    if predictions["next_date"].duplicated().any():
        raise ValueError("Duplicate predictions for the same session")

    prices = prepare_simulation_prices(prices_df,predictions["next_date"].min(),predictions["target_end_date"].max())
    entry_positions = prices.index.get_indexer(predictions["next_date"])
    exit_positions = prices.index.get_indexer(predictions["target_end_date"])

    if (entry_positions < 0).any() or (exit_positions < 0).any():
        raise ValueError("Missing entry or exit session prices")
    if (exit_positions - entry_positions != model_config.horizon).any():
        raise ValueError("Prediction dates do not match the model horizon")

    predictions = predictions.set_index("next_date")

    cash = initial_cash
    shares = 0.0
    exit_date = None
    trades = []
    equity_history = []
    invested_sessions = 0

    for session_date, session in prices.iterrows():
        trade_date = session_date.date().isoformat()
        open_price = float(session["open"])

        if shares > 0 and session_date == exit_date:
            sale = calculate_sell(shares, open_price, strategy_config)
            cash += sale["proceeds"]

            trades.append({
                "type": "SELL",
                "date": trade_date,
                "price": sale["price"],
                "shares": shares,
                "fee": sale["fee"]})

            shares = 0.0
            exit_date = None

        if shares == 0 and cash > 0 and session_date in predictions.index:
            prediction = predictions.loc[session_date]

            if should_enter_position(prediction, model_config, strategy_config):
                purchase = calculate_buy(cash, open_price, strategy_config)

                cash = purchase["cash_remaining"]
                shares = purchase["shares"]
                exit_date = prediction["target_end_date"]

                trades.append({
                    "type": "BUY",
                    "date": trade_date,
                    "price": purchase["price"],
                    "shares": shares,
                    "fee": purchase["fee"]})

        if shares > 0:
            invested_sessions += 1

        wallet_value = cash + shares * float(session["close"])

        equity_history.append({
            "date": trade_date,
            "value": float(wallet_value)})

    if shares > 0:
        raise ValueError("Simulation ended before the planned exit")

    closed_trades = build_closed_trades(trades)
    profitable, losing, win_rate = calculate_trades_stats(closed_trades)
    final_value = float(cash)

    return {
        "initial_cash": initial_cash,
        "strategy_config": asdict(strategy_config),
        "confidence_threshold": strategy_config.confidence_threshold,
        "horizon": model_config.horizon,
        "task": model_config.task,
        "final_value": final_value,
        "total_return": final_value / initial_cash - 1,
        "position_open": False,
        "total_transactions": len(trades),
        "trades": trades,
        "equity_history": equity_history,
        "max_drawdown": calculate_max_drawdown(equity_history, initial_cash),
        "market_exposure": invested_sessions / (len(prices) - 1),
        "cagr": calculate_cagr(equity_history, initial_cash, final_value),
        "closed_trades": closed_trades,
        "profitable_trades": profitable,
        "losing_trades": losing,
        "win_rate": win_rate
    }

def run_buy_and_hold(prices, strategy_config, initial_cash=10000):
    if len(prices) < 2:
        raise ValueError("Buy and Hold requires minimum two sessions")

    initial_cash = float(initial_cash)
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("Initial cash must be finite and positive")

    buy_hold_config = replace(strategy_config, position_fraction=1.0)

    purchase = calculate_buy(initial_cash,float(prices.iloc[0]["open"]),buy_hold_config)
    cash = purchase["cash_remaining"]
    shares = purchase["shares"]
    sale = calculate_sell(shares,float(prices.iloc[-1]["open"]),buy_hold_config)
    final_value = float(cash + sale["proceeds"])
    first_date = prices.index[0].date().isoformat()
    last_date = prices.index[-1].date().isoformat()

    trades = [
        {
            "type": "BUY",
            "date": first_date,
            "price": purchase["price"],
            "shares": shares,
            "fee": purchase["fee"]
        },
        {
            "type": "SELL",
            "date": last_date,
            "price": sale["price"],
            "shares": shares,
            "fee": sale["fee"]
        }
    ]

    equity_history = [{"date": session_date.date().isoformat(),"value": float(cash + shares * session["close"])}
        for session_date, session in prices.iloc[:-1].iterrows()]

    equity_history.append({
        "date": last_date,
        "value": final_value})

    return {
        "initial_cash": initial_cash,
        "final_value": final_value,
        "total_return": final_value / initial_cash - 1,
        "position_open": False,
        "buy_price": purchase["price"],
        "final_price": sale["price"],
        "buy_fee": purchase["fee"],
        "sell_fee": sale["fee"],
        "total_transactions": len(trades),
        "trades": trades,
        "equity_history": equity_history,
        "max_drawdown": calculate_max_drawdown(equity_history, initial_cash),
        "market_exposure": 1.0,
        "cagr": calculate_cagr(equity_history, initial_cash, final_value)
    }


def build_backtest_result(results_df, prices_df, model_config, strategy_config, start_date, end_date, initial_cash, training_rows):

    if results_df.empty:
        raise ValueError("No predictions for simulation")

    result_df = results_df.sort_values("date").copy()
    model_evaluation = evaluate(result_df, model_config)

    prices = prepare_simulation_prices(prices_df, pd.to_datetime(result_df["next_date"], errors="raise").min(), pd.to_datetime(result_df["target_end_date"], errors="raise").max())
    model_summary = run_backtest(result_df, prices_df, model_config, strategy_config, initial_cash)
    buy_hold_summary = run_buy_and_hold(prices, strategy_config, initial_cash)
    metrics = {
        name: value if isinstance(value, dict) else float(value)
        for name, value in model_evaluation.items()
        if name not in ("predictions", "probability_up")}

    if model_config.task == "classification":
        result_df["is_correct"] = (result_df["prediction"] == result_df["target"])

    reported_columns = [
        "ticker", "date", "close",
        "next_date", "next_open", "next_close",
        "target_end_date", "target_end_open",
        "target", "target_return",
        "prediction", "probability_up", "predicted_return", "is_correct"]

    records_df = result_df[[column for column in reported_columns if column in result_df.columns]].copy()

    for column in ("date", "next_date", "target_end_date"):
        records_df[column] = pd.to_datetime(records_df[column], errors="raise").dt.strftime("%Y-%m-%d")

    result = {
        "task": model_config.task,
        "horizon": model_config.horizon,
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "effective_test_start_date": prices.index[0].date().isoformat(),
        "effective_test_end_date": prices.index[-1].date().isoformat(),
        "training_rows": int(training_rows),
        "total_predictions": int(len(result_df)),
        "predictions": records_df.to_dict(orient="records"),
        "model_summary": model_summary,
        "buyandhold_summary": buy_hold_summary}

    result.update(metrics)

    if model_config.task == "classification":
        result["correct_predictions"] = int(result_df["is_correct"].sum())
        result["baseline_accuracy"] = None

        if "baseline_prediction" in result_df.columns:
            result["baseline_accuracy"] = float(accuracy_score(result_df["target"],result_df["baseline_prediction"]))

    return result
