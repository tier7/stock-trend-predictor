
import argparse

from services.ml.global_forest_service import train_global_model
from services.ml.model_config import ModelConfig
from services.ml.feature_schema import FEATURE_GROUPS
from services.news_fetcher import NewsPreparationError

def main():
    parser = argparse.ArgumentParser(description="Wytrenuj i zapisz gotowy model globalny.")
    parser.add_argument(
        "--cutoff", required=True, metavar="YYYY-MM-DD",
        help="Data graniczna danych treningowych.",
    )

    parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        default="classification",
        help="Kierunek albo stopa zwrotu.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Horyzont prognozy w sesjach giełdowych.",
    )

    parser.add_argument(
        "--feature-groups",
        nargs="+",
        choices=tuple(FEATURE_GROUPS),
        default=["technical"],
        help="Grupy cech używane przez model.",
    )
    parser.add_argument(
        "--news-start",
        metavar="YYYY-MM-DD",
        help="Początek zakresu newsów.",
    )
    parser.add_argument(
        "--news-end",
        metavar="YYYY-MM-DD",
        help="Koniec zakresu newsów, bez wskazanego dnia.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        type=str.upper,
        help="Lista spółek treningowych.",
    )
    parser.add_argument(
        "--prepared-news-only",
        action="store_true",
        help="Korzystaj wyłącznie z zapisanych newsów i ocen FinBERT.",
    )



    arguments = parser.parse_args()
    try:
        config = ModelConfig(
            algorithm="random_forest",
            task=arguments.task,
            horizon=arguments.horizon,
            feature_groups=tuple(arguments.feature_groups)
        )

        metadata = train_global_model(
            arguments.cutoff,
            progress_callback=print,
            config=config,
            news_start=arguments.news_start,
            news_end=arguments.news_end,
            training_tickers=arguments.tickers,
            prepare_news = not arguments.prepared_news_only,
        )

    except NewsPreparationError as error:
        parser.exit(1, f"News preparation failed: {error}\n")
    except ValueError as error:
        parser.error(str(error))
    print(f"Zapisano model: {metadata['model_id']}")
    print(f"Zadanie: {config.task}")
    print(f"Horyzont: {config.horizon} sesji")
    print(f"Data graniczna treningu: {metadata['training_cutoff']}")
    print(f"Grupy cech: {', '.join(config.feature_groups)}")
    print(f"Liczba spółek: {len(metadata['train_tickers'])}")



if __name__ == "__main__":
    main()
