
def run_backtest(predictions, initial_cash=10000, fee_rate=0.001):
    if not predictions:
        raise ValueError("No predictions available for backtest.")
    cash = initial_cash
    shares = 0
    trades = []
    wallet_value = cash
    equity_history = []

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
        else:
            wallet_value = cash
        equity_history.append({"date":trade_date, "value": wallet_value })

    position_open = shares > 0
    final_value = equity_history[-1]["value"]
    total_return = final_value / initial_cash - 1

    return {
        "initial_cash": float(initial_cash),
        "final_value": float(final_value),
        "total_return": float(total_return),
        "position_open": position_open,
        "total_transactions": len(trades),
        "trades": trades,
        "equity_history": equity_history,
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

    return {
        "initial_cash": float(initial_cash),
        "final_value": float(final_value),
        "total_return": float(total_return),
        "buy_price": buy_price,
        "final_price": final_price,
        "buy_fee": float(buy_fee),
        "sell_fee": float(sell_fee),
    }




