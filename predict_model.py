import argparse
import json
from pathlib import Path

from services.data_updater import ensure_stock_data
from services.db import get_stock_data
from services.ml.data_preparation_service import validate_simulation_dates
from services.ml.model_service import predict_history, predict_latest
from services.ml.model_store_service import get_model_config, validate_model_id
from services.ml.sector_feature_service import load_sector_assignments
from services.news_fetcher import NewsPreparationError


def main():
    parser = argparse.ArgumentParser(description="Uruchom zapisany model RF, LightGBM, MLP lub hybrydę bez ponownego treningu.")
    parser.add_argument("--model-id", required=True, help="Identyfikator wypisany po zapisaniu modelu.")
    parser.add_argument("--ticker", required=True, type=str.upper, help="Spółka, dla której powstanie prognoza.")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--latest", action="store_true", help="Prognoza dla ostatniego wiersza danych przed następnym otwarciem.")
    modes.add_argument("--start", metavar="YYYY-MM-DD", help="Początek historycznego okresu prognoz.")
    parser.add_argument("--end", metavar="YYYY-MM-DD", help="Koniec okresu historycznego; wymagany razem z --start.")
    parser.add_argument("--sector-map", metavar="PATH", help="Plik JSON z przypisaniami sektorów, potrzebny dla nowych spółek przy grupie sector.")
    parser.add_argument("--prepared-news-only", action="store_true", help="Korzystaj wyłącznie z zapisanych newsów i ocen FinBERT.")
    parser.add_argument("--model-dir", help="Opcjonalny katalog zapisanych modeli.")
    parser.add_argument("--output", metavar="PATH", help="Zapisz najnowszą prognozę jako JSON lub historyczne prognozy jako CSV.")

    arguments = parser.parse_args()
    try:
        sector_assignments = load_sector_assignments(arguments.sector_map)
        ticker = arguments.ticker.strip()
        if not ticker:
            raise ValueError("Ticker nie może być pusty")
        if arguments.latest and arguments.end is not None:
            raise ValueError("--end można podać wyłącznie razem z --start")
        if not arguments.latest:
            if arguments.end is None:
                raise ValueError("Prognozy historyczne wymagają --start i --end")
            validate_simulation_dates(arguments.start, arguments.end)

        validate_model_id(arguments.model_id)

        if not ensure_stock_data(ticker, "1d"):
            raise ValueError(f"Brak danych cenowych dla {ticker}")
        data = get_stock_data(ticker, years=None)
        if data.empty:
            raise ValueError(f"Brak danych {ticker} w bazie")

        if arguments.latest:
            result = predict_latest(data, arguments.model_id, model_dir=arguments.model_dir, ticker=ticker, prepare_news=not arguments.prepared_news_only, progress_callback=print, sector_assignments=sector_assignments)
            metadata = result.pop("model_metadata")
            result["model_id"] = metadata["model_id"]
            result["ticker"] = ticker
            payload = json.dumps(result, ensure_ascii=False, indent=2)
            print(payload)
            if arguments.output:
                Path(arguments.output).write_text(payload + "\n", encoding="utf-8")
        else:
            results, metadata = predict_history(data, arguments.model_id, arguments.start, arguments.end, model_dir=arguments.model_dir, ticker=ticker, prepare_news=not arguments.prepared_news_only, progress_callback=print, sector_assignments=sector_assignments)
            config = get_model_config(metadata)
            columns = ["date", "next_date", "target_end_date", config.target_column]
            if "sector_benchmark" in results.columns:
                columns.append("sector_benchmark")
            if config.task == "classification":
                columns.extend(["probability_up", "prediction"])
            else:
                columns.append("predicted_return")
            report = results[columns].copy()
            report.insert(0, "ticker", ticker)
            report.insert(0, "model_id", metadata["model_id"])
            print(f"Liczba prognoz: {len(report)}; podgląd pierwszych 10 wierszy:")
            print(report.head(10).to_string(index=False))
            if arguments.output:
                report.to_csv(arguments.output, index=False, encoding="utf-8")

        if arguments.output:
            print(f"Zapisano wynik: {arguments.output}")
    except NewsPreparationError as error:
        parser.exit(1, f"Nie udało się przygotować newsów: {error}\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
