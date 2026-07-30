from datetime import date


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

def run_backtest(predictions, initial_cash=10000, fee_rate=0.001):
    if not predictions:
        raise ValueError("No predictions available for backtest.")
    cash = initial_cash
    shares = 0
    trades = []
    wallet_value = cash
    max_wallet_value = wallet_value
    equity_history = []
    dropdown = 0
    days_in_market = 0
    cagr = 0

    for prediction in predictions:
        signal = prediction["prediction"]
        price = float(prediction["next_open"])
        trade_date = prediction["next_date"]
        valuation_price = float(prediction["next_close"])


        if signal == 1 and cash > 0:
            trade_value = cash / (1 + fee_rate)
            fee = trade_value * fee_rate
            shares = trade_value / price
            cash = 0

            trades.append({"type": "BUY","date": trade_date,"price": price,"shares": shares,"fee": fee})

        elif signal == 0 and shares > 0:
            sold_shares = shares
            gross_value = sold_shares * price
            fee = gross_value * fee_rate
            cash = gross_value - fee
            shares = 0

            trades.append({"type": "SELL","date": trade_date,"price": price,"shares": sold_shares,"fee": fee})

        if shares > 0:
            wallet_value = cash + shares * valuation_price * (1 - fee_rate)
            days_in_market += 1
        else:
            wallet_value = cash
        equity_history.append({"date":trade_date, "value": wallet_value })

    position_open = shares > 0
    final_value = equity_history[-1]["value"]
    total_return = final_value / initial_cash - 1
    max_drawdown = calculate_max_drawdown(equity_history, initial_cash)
    market_exposure = days_in_market/len(predictions)
    cagr = calculate_cagr(equity_history, initial_cash, final_value)

    closed_trades = build_closed_trades(trades)
    profitable_trans, unprofitable_trans, winrate = calculate_trades_stats(closed_trades)

    return {
        "initial_cash": float(initial_cash),
        "final_value": float(final_value),
        "total_return": float(total_return),
        "position_open": position_open,
        "total_transactions": len(trades),
        "trades": trades,
        "equity_history": equity_history,
        "max_drawdown": max_drawdown,
        "market_exposure": market_exposure,
        "cagr": cagr,
        "closed_trades": closed_trades,
        "profitable_trades": profitable_trans,
        "losing_trades": unprofitable_trans,
        "win_rate": winrate,
    }

def run_buy_and_hold(predictions, initial_cash=10000, fee_rate=0.001):
    if not predictions:
        raise ValueError("No predictions available for buy and hold.")

    buy_price = float(predictions[0]["next_open"])
    final_price = float(predictions[-1]["next_close"])

    trade_value = initial_cash / (1 + fee_rate)
    buy_fee = trade_value * fee_rate
    shares = trade_value / buy_price

    gross_value = shares * final_price
    sell_fee = gross_value * fee_rate
    final_value = gross_value - sell_fee
    total_return = final_value / initial_cash - 1

    equity_history = []
    for prediction in predictions:
        valuation_price = float(prediction["next_close"])
        value = shares * valuation_price * (1 - fee_rate)

        equity_history.append({
            "date": prediction["next_date"],
            "value": value,
        })
    max_drawdown = calculate_max_drawdown(equity_history,initial_cash,)
    cagr = calculate_cagr(equity_history, initial_cash, final_value)
    return {
        "initial_cash": float(initial_cash),
        "final_value": float(final_value),
        "total_return": float(total_return),
        "buy_price": buy_price,
        "final_price": final_price,
        "buy_fee": float(buy_fee),
        "sell_fee": float(sell_fee),
        "max_drawdown": max_drawdown,
        "cagr": cagr,
        "equity_history": equity_history
    }




