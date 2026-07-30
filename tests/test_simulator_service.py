import pytest

from services.simulator_service import calculate_cagr, calculate_max_drawdown, run_backtest, run_buy_and_hold


def test_calculate_max_drawdown_from_highest_peak():
    equity_history = [
        {"date": "2026-01-01", "value": 10000},
        {"date": "2026-01-02", "value": 11000},
        {"date": "2026-01-03", "value": 8800},
        {"date": "2026-01-04", "value": 9900},
    ]

    result = calculate_max_drawdown(equity_history=equity_history,initial_cash=10000)
    assert result == pytest.approx(0.20)

def test_calculate_max_drawdown_returns_zero_when_value_only_increases():
    equity_history = [
        {"date": "2026-01-01", "value": 10000},
        {"date": "2026-01-02", "value": 10500},
        {"date": "2026-01-03", "value": 12000},
    ]

    result = calculate_max_drawdown(equity_history=equity_history,initial_cash=10000)
    assert result == 0

def test_calculate_cagr_for_two_year_investment():
    equity_history = [
        {"date": "2024-01-01", "value": 10000},
        {"date": "2026-01-01", "value": 12100},
    ]

    result = calculate_cagr(equity_history=equity_history,initial_value=10000,final_value=12100)
    assert result == pytest.approx(0.10, abs=0.0001)

def test_calculate_cagr_raises_error_for_identical_dates():
    equity_history = [
        {"date": "2026-01-01", "value": 10000},
        {"date": "2026-01-01", "value": 11000},
    ]

    with pytest.raises(ValueError, match="CAGR requires at least two different dates"):
        calculate_cagr(equity_history=equity_history,initial_value=10000, final_value=11000)

def test_run_backtest_buys_then_sells_with_profit():
    predictions = [
        {
            "prediction": 1,
            "next_date": "2026-01-01",
            "next_open": 100,
            "next_close": 110,
        },
        {
            "prediction": 0,
            "next_date": "2026-01-02",
            "next_open": 120,
            "next_close": 120,
        },
    ]

    result = run_backtest(predictions=predictions,initial_cash=10000,fee_rate=0)

    assert result["final_value"] == pytest.approx(12000)
    assert result["total_return"] == pytest.approx(0.20)
    assert result["total_transactions"] == 2
    assert result["position_open"] is False
    assert result["market_exposure"] == pytest.approx(0.50)
    assert result["trades"][0]["type"] == "BUY"
    assert result["trades"][1]["type"] == "SELL"
    assert len(result["closed_trades"]) == 1

    closed_trade = result["closed_trades"][0]
    assert closed_trade["profit"] == pytest.approx(2000)
    assert closed_trade["return_rate"] == pytest.approx(0.20)
    assert closed_trade["holding_days"] == 1
    assert closed_trade["total_fees"] == 0


def test_run_backtest_applies_transaction_fees():
    predictions = [
        {
            "prediction": 1,
            "next_date": "2026-01-01",
            "next_open": 100,
            "next_close": 110,
        },
        {
            "prediction": 0,
            "next_date": "2026-01-02",
            "next_open": 120,
            "next_close": 120,
        },
    ]

    result_without_fee = run_backtest(predictions=predictions,initial_cash=10000,fee_rate=0)
    result_with_fee = run_backtest(predictions=predictions,initial_cash=10000,fee_rate=0.001)

    assert result_with_fee["final_value"] < result_without_fee["final_value"]
    assert result_with_fee["trades"][0]["fee"] > 0
    assert result_with_fee["trades"][1]["fee"] > 0
    closed_trade = result_with_fee["closed_trades"][0]

    assert closed_trade["total_fees"] == pytest.approx(result_with_fee["trades"][0]["fee"]+ result_with_fee["trades"][1]["fee"])
    assert closed_trade["profit"] == pytest.approx(result_with_fee["final_value"]- result_with_fee["initial_cash"])

def test_run_backtest_with_empty_predictions():
    predictions = []
    with pytest.raises(ValueError, match="No predictions available for backtest"):
        run_backtest(predictions=predictions,initial_cash=10000,fee_rate=0)

def test_run_backtest_with_open_position_at_end():
    predictions = [
        {
            "prediction": 1,
            "next_date": "2026-01-01",
            "next_open": 100,
            "next_close": 110,
        },
        {
            "prediction": 1,
            "next_date": "2026-01-02",
            "next_open": 120,
            "next_close": 125,
        },
    ]

    result = run_backtest(predictions=predictions,initial_cash=10000,fee_rate=0)

    assert result["position_open"] is True
    assert result["total_transactions"] == 1
    assert result["trades"][0]["type"] == "BUY"
    assert result["closed_trades"] == []

def test_run_buy_and_hold_returns_expected_profit():
    predictions = [
        {
            "prediction": 0,
            "next_date": "2026-01-01",
            "next_open": 100,
            "next_close": 110,
        },
        {
            "prediction": 0,
            "next_date": "2026-01-02",
            "next_open": 115,
            "next_close": 120,
        },
    ]

    result = run_buy_and_hold(predictions=predictions,initial_cash=10000,fee_rate=0)

    assert result["buy_price"] == pytest.approx(100)
    assert result["final_price"] == pytest.approx(120)
    assert result["final_value"] == pytest.approx(12000)
    assert result["total_return"] == pytest.approx(0.20)
    assert len(result["equity_history"]) == 2

def test_run_buy_and_hold_with_empty_predictions():
    predictions = []

    with pytest.raises(ValueError, match="No predictions available for buy and hold"):
        run_buy_and_hold(predictions=predictions,initial_cash=10000,fee_rate=0)
