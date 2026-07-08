from services.db import get_connection, get_company_name_from_db, save_company_data
from services.data_fetcher import fetch_company_name


def ensure_stock_name(ticker):
    ticker = ticker.upper()
    name = get_company_name_from_db(ticker)

    if name is not None:
        return name

    print(f"No data for {ticker} in companies table")

    name = fetch_company_name(ticker)

    if name is None:
        return ticker

    save_company_data(ticker, name)

    return name